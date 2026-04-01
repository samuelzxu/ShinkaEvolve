#!/usr/bin/env python3
"""
claude-evolve: Evolutionary code optimization orchestrator for Claude Code.

A standalone script that manages population state, parent selection,
evaluation, and archive management. Claude Code acts as the mutation
operator — no LLM API keys required.

Usage:
    python evolve.py init --initial <path> --eval-cmd <cmd> [--language python]
    python evolve.py select [--strategy weighted]
    python evolve.py evaluate --candidate <path>
    python evolve.py update --candidate-dir <dir> --parent-id <id> --mutation-type <type> --description <desc>
    python evolve.py status
    python evolve.py leaderboard [--top 10]
"""

import argparse
import difflib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stable_sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def median_absolute_deviation(values: list[float]) -> float:
    med = median(values)
    deviations = [abs(v - med) for v in values]
    return median(deviations)


STATE_FILE = "state.json"
PROGRAMS_DIR = "programs"

# ---------------------------------------------------------------------------
# Default model pool — (model, effort) combinations
# Emulates ShinkaEvolve's multi-model ensemble using Claude Code's
# available models and thinking effort levels.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_POOL = [
    {"model": "opus",   "effort": "max",    "label": "opus/max"},
    {"model": "opus",   "effort": "high",   "label": "opus/high"},
    {"model": "opus",   "effort": "medium", "label": "opus/medium"},
    {"model": "sonnet", "effort": "high",   "label": "sonnet/high"},
    {"model": "sonnet", "effort": "medium", "label": "sonnet/medium"},
    {"model": "sonnet", "effort": "low",    "label": "sonnet/low"},
    {"model": "haiku",  "effort": "medium", "label": "haiku/medium"},
    {"model": "haiku",  "effort": "low",    "label": "haiku/low"},
]


def default_model_stats() -> list[dict]:
    """Initialize bandit stats for each model in the pool."""
    return [
        {
            "label": m["label"],
            "model": m["model"],
            "effort": m["effort"],
            "total_pulls": 0,
            "total_reward": 0.0,       # Asymmetric: only positive improvements
            "best_score_delta": 0.0,
            "n_correct": 0,
            "n_incorrect": 0,
        }
        for m in DEFAULT_MODEL_POOL
    ]


# Default bandit hyperparameters
BANDIT_MIN_PULLS = 4       # K retries per arm before judging
BANDIT_EXPLORATION_C = 1.41  # UCB exploration coefficient
BANDIT_EPSILON = 0.2        # ε-greedy exploration rate


# ---------------------------------------------------------------------------
# Asymmetric UCB1 Model Selection (bandit)
# Ported from ShinkaEvolve's AsymmetricUCB in shinka/llm/prioritization.py
# ---------------------------------------------------------------------------

def select_model_ucb1(
    model_stats: list[dict],
    total_generations: int,
    min_pulls: int = BANDIT_MIN_PULLS,
    exploration_c: float = BANDIT_EXPLORATION_C,
    epsilon: float = BANDIT_EPSILON,
) -> dict:
    """
    Asymmetric UCB1 bandit with min-pulls warmup and ε-exploration.

    Mirrors ShinkaEvolve's AsymmetricUCB:
    - Warm-up phase: round-robin until each arm has min_pulls (K=4)
    - Asymmetric reward: only positive improvements count (max(0, delta))
    - ε-exploration: with probability ε, pick a random arm
    - UCB1 exploitation: argmax(mean_reward + c * sqrt(ln(N) / n_i))

    Returns the selected model_stats entry.
    """
    # Phase 1: Warm-up — ensure every arm gets min_pulls before exploiting
    under_min = [m for m in model_stats if m["total_pulls"] < min_pulls]
    if under_min:
        # Pick the arm with fewest pulls (round-robin effect)
        min_count = min(m["total_pulls"] for m in under_min)
        least_pulled = [m for m in under_min if m["total_pulls"] == min_count]
        return random.choice(least_pulled)

    # Phase 2: ε-exploration — random arm with probability ε
    if random.random() < epsilon:
        return random.choice(model_stats)

    # Phase 3: UCB1 exploitation
    # argmax( mean_reward + c * sqrt(ln(N) / n_i) )
    n_total = max(total_generations, 1)
    ln_n = math.log(n_total)

    best_ucb = -float("inf")
    best_arm = model_stats[0]

    for m in model_stats:
        n_i = m["total_pulls"]
        mean_reward = m["total_reward"] / n_i
        exploration_bonus = exploration_c * math.sqrt(ln_n / n_i)
        ucb_value = mean_reward + exploration_bonus

        if ucb_value > best_ucb:
            best_ucb = ucb_value
            best_arm = m

    return best_arm


def get_base_dir() -> Path:
    """Return the claude-evolve base directory (where this script lives)."""
    return Path(__file__).resolve().parent


def load_state() -> dict:
    state_path = get_base_dir() / STATE_FILE
    if not state_path.exists():
        print("Error: No state.json found. Run 'evolve.py init' first.", file=sys.stderr)
        sys.exit(1)
    with open(state_path) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    state_path = get_base_dir() / STATE_FILE
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def next_program_id(state: dict) -> str:
    """Generate the next program ID based on existing programs."""
    existing = [p["id"] for p in state["programs"]]
    max_num = -1
    for pid in existing:
        try:
            num = int(pid.split("_")[1])
            if num > max_num:
                max_num = num
        except (IndexError, ValueError):
            pass
    return f"prog_{max_num + 1:04d}"


def get_program_dir(prog_id: str) -> Path:
    return get_base_dir() / PROGRAMS_DIR / prog_id


def get_program_by_id(state: dict, prog_id: str) -> dict | None:
    for p in state["programs"]:
        if p["id"] == prog_id:
            return p
    return None


# ---------------------------------------------------------------------------
# Embeddings (local model for code dedup)
# Uses intfloat/multilingual-e5-large via sentence-transformers
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_SIMILARITY_THRESHOLD = 0.97  # Cosine similarity threshold for dedup
EMBEDDINGS_FILE = "embeddings.json"  # Stored separately (large vectors)

_embedding_model = None  # Lazy-loaded singleton


def get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except ImportError:
            print("Warning: sentence-transformers not installed. "
                  "Run: pip install sentence-transformers", file=sys.stderr)
            return None
    return _embedding_model


def compute_embedding(code: str) -> list[float] | None:
    """Compute embedding vector for a code string."""
    model = get_embedding_model()
    if model is None:
        return None
    # E5 models expect "query: " or "passage: " prefix
    vec = model.encode(f"passage: {code}", normalize_embeddings=True)
    return vec.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_embeddings() -> dict[str, list[float]]:
    """Load embeddings from disk."""
    path = get_base_dir() / EMBEDDINGS_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Save embeddings to disk."""
    path = get_base_dir() / EMBEDDINGS_FILE
    with open(path, "w") as f:
        json.dump(embeddings, f)


def compute_and_store_embedding(prog_id: str, code: str) -> list[float] | None:
    """Compute embedding for a program and store it."""
    block = extract_evolve_block(code)
    embedding = compute_embedding(block)
    if embedding:
        embeddings = load_embeddings()
        embeddings[prog_id] = embedding
        save_embeddings(embeddings)
    return embedding


def find_most_similar_by_embedding(
    candidate_embedding: list[float],
    exclude_ids: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Find most similar programs by embedding cosine similarity."""
    embeddings = load_embeddings()
    exclude = exclude_ids or set()
    similarities = []
    for pid, emb in embeddings.items():
        if pid in exclude:
            continue
        sim = cosine_similarity(candidate_embedding, emb)
        similarities.append((pid, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


# ---------------------------------------------------------------------------
# Prompt Evolution (co-evolving system prompts alongside code)
# Ported from ShinkaEvolve's SystemPromptEvolver + SystemPromptDatabase
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_ARCHIVE_SIZE = 5
DEFAULT_PROMPT_EVOLUTION_INTERVAL = 10  # Evolve prompt every N generations
DEFAULT_PROMPT_UCB_C = 1.0
DEFAULT_PROMPT_EPSILON = 0.1
DEFAULT_PROMPT_MIN_EVALS = 3  # Min programs before using fitness


def default_prompt_archive(task_sys_msg: str | None) -> list[dict]:
    """Create initial prompt archive with the default task system message."""
    if not task_sys_msg:
        return []
    return [{
        "id": "prompt_0000",
        "prompt_text": task_sys_msg,
        "name": "initial",
        "description": "Default task system prompt",
        "parent_id": None,
        "generation": 0,
        "patch_type": "init",
        "program_count": 0,
        "correct_program_count": 0,
        "total_percentile": 0.0,
        "fitness": 0.5,  # Optimistic prior
        "program_scores": [],
        "in_archive": True,
    }]


def next_prompt_id(state: dict) -> str:
    """Generate the next prompt ID."""
    prompts = state.get("prompt_archive", [])
    max_num = -1
    for p in prompts:
        try:
            num = int(p["id"].split("_")[1])
            if num > max_num:
                max_num = num
        except (IndexError, ValueError):
            pass
    return f"prompt_{max_num + 1:04d}"


def select_prompt_ucb(
    prompt_archive: list[dict],
    min_evals: int = DEFAULT_PROMPT_MIN_EVALS,
    exploration_c: float = DEFAULT_PROMPT_UCB_C,
    epsilon: float = DEFAULT_PROMPT_EPSILON,
) -> dict:
    """UCB selection for system prompts. Matches ShinkaEvolve's prompt_dbase.py."""
    if not prompt_archive:
        return {"prompt_text": None, "id": None}

    # Epsilon-greedy exploration
    if random.random() < epsilon:
        return random.choice(prompt_archive)

    total_evals = sum(p.get("correct_program_count", 0) for p in prompt_archive)
    if total_evals == 0:
        return random.choice(prompt_archive)

    best_ucb = -float("inf")
    best_prompt = prompt_archive[0]

    for p in prompt_archive:
        n = p.get("correct_program_count", 0)
        if n < min_evals:
            # Optimistic prior for under-evaluated prompts
            ucb = 1.0 + exploration_c
        else:
            exploitation = p.get("fitness", 0.5)
            exploration = exploration_c * math.sqrt(math.log(max(total_evals, 1)) / n)
            ucb = exploitation + exploration

        if ucb > best_ucb:
            best_ucb = ucb
            best_prompt = p

    return best_prompt


def update_prompt_fitness(state: dict, prompt_id: str, score: float, correct: bool):
    """Update prompt fitness based on a program's evaluation result."""
    if not prompt_id:
        return
    prompts = state.get("prompt_archive", [])
    # Compute percentile against all correct programs
    correct_scores = [
        p["combined_score"] for p in state["programs"]
        if p.get("correct", False)
    ]
    if not correct_scores:
        percentile = 0.5
    else:
        rank = sum(1 for s in correct_scores if s <= score)
        percentile = rank / len(correct_scores)

    for p in prompts:
        if p["id"] == prompt_id:
            p["program_count"] = p.get("program_count", 0) + 1
            p["program_scores"] = p.get("program_scores", [])
            p["program_scores"].append(score)
            if correct:
                p["correct_program_count"] = p.get("correct_program_count", 0) + 1
                p["total_percentile"] = p.get("total_percentile", 0.0) + percentile
                cc = p["correct_program_count"]
                p["fitness"] = p["total_percentile"] / cc if cc > 0 else 0.5
            break


def should_evolve_prompt(state: dict) -> bool:
    """Check if prompt evolution should trigger."""
    config = state["config"]
    if not config.get("evolve_prompts", False):
        return False
    interval = config.get("prompt_evolution_interval", DEFAULT_PROMPT_EVOLUTION_INTERVAL)
    gen = state["generation"]
    last_evo = state.get("last_prompt_evolution_gen", 0)
    return interval > 0 and gen > 0 and (gen - last_evo) >= interval


def build_prompt_evolution_context(state: dict, parent_prompt: dict) -> str:
    """Build context for prompt evolution subagent."""
    config = state["config"]
    ext = config.get("extension", "py")

    # Get top-k programs
    correct = [p for p in state["programs"] if p.get("correct", False)]
    correct.sort(key=lambda p: p["combined_score"], reverse=True)
    top_k = correct[:3]

    lines = []
    lines.append("# Prompt Evolution Context")
    lines.append(f"Current generation: {state['generation']}")
    lines.append(f"Best score: {state.get('best_score')}")
    lines.append("")
    lines.append("## Current Prompt (to evolve):")
    lines.append(f"```\n{parent_prompt['prompt_text']}\n```")
    lines.append(f"Fitness: {parent_prompt.get('fitness', 'N/A')}")
    lines.append(f"Programs generated: {parent_prompt.get('program_count', 0)}")
    lines.append("")
    lines.append("## Top Programs (generated with various prompts):")

    for p in top_k:
        code_path = get_program_dir(p["id"]) / f"main.{ext}"
        lines.append(f"\n### {p['id']} (score: {p['combined_score']:.4f})")
        lines.append(f"Type: {p.get('mutation_type', '?')}, Desc: {p.get('description', '')}")
        if code_path.exists():
            code = code_path.read_text(encoding="utf-8")
            block = extract_evolve_block(code)
            lines.append(f"```{ext}\n{block[:2000]}\n```")

    return "\n".join(lines)


PROMPT_EVO_SYSTEM = """You are an expert prompt engineer specializing in crafting optimal task instructions for code generation.

Your goal is to improve the system prompt so that future code generations achieve higher scores. Analyze the successful programs to understand what patterns and techniques led to high scores.

Respond with:
<NAME>Short name for this prompt variant (up to 10 words)</NAME>
<DESCRIPTION>What you changed and why (1-2 sentences)</DESCRIPTION>
<PROMPT>The complete new system prompt text</PROMPT>"""


# ---------------------------------------------------------------------------
# Novelty Check (difflib + embedding-based, ported from ShinkaEvolve's NoveltyJudge)
# ---------------------------------------------------------------------------

NOVELTY_SIMILARITY_THRESHOLD = 0.95  # difflib threshold

# Prompt template matching ShinkaEvolve's prompts_novelty.py
NOVELTY_JUDGE_PROMPT = """You are an expert code reviewer tasked with determining if two code snippets are meaningfully different.

Analyze both programs and determine if the proposed code introduces meaningful changes. Consider:
1. Algorithmic differences: Different approaches, logic, or strategies
2. Structural changes: Different data structures, control flow, or organization
3. Functional improvements: New features, optimizations, or capabilities
4. Implementation variations: Different ways of achieving the same goal
5. Hyperparameter changes that could lead to different performance

Ignore trivial differences like:
- Variable name changes
- Minor formatting or style changes
- Comments or documentation changes
- Insignificant refactoring that doesn't change core logic

Respond with NOVEL or NOT_NOVEL followed by a brief explanation."""


def compute_code_similarity(code_a: str, code_b: str) -> float:
    """
    Compute similarity ratio between two code strings using SequenceMatcher.
    Returns a float between 0.0 (completely different) and 1.0 (identical).
    """
    return difflib.SequenceMatcher(None, code_a, code_b).ratio()


def extract_evolve_block(code: str) -> str:
    """Extract the code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers."""
    lines = code.splitlines()
    in_block = False
    block_lines = []
    for line in lines:
        if "EVOLVE-BLOCK-START" in line:
            in_block = True
            continue
        if "EVOLVE-BLOCK-END" in line:
            break
        if in_block:
            block_lines.append(line)
    return "\n".join(block_lines) if block_lines else code


# ---------------------------------------------------------------------------
# Retry Session
# ---------------------------------------------------------------------------

MAX_RETRIES = 3  # Default max retries per generation attempt


def get_active_session(state: dict) -> dict | None:
    """Get the active retry session, if any."""
    return state.get("retry_session")


def start_retry_session(
    state: dict,
    parent_id: str,
    model_config: dict,
    candidate_id: str,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Start a new retry session."""
    session = {
        "parent_id": parent_id,
        "model_label": model_config["label"],
        "model": model_config["model"],
        "effort": model_config["effort"],
        "candidate_id": candidate_id,
        "attempt": 1,
        "max_retries": max_retries,
        "errors": [],
    }
    state["retry_session"] = session
    return session


def advance_retry_session(state: dict, error_msg: str | None = None) -> dict | None:
    """Advance to next retry attempt. Returns None if max retries exceeded."""
    session = state.get("retry_session")
    if not session:
        return None
    if error_msg:
        session["errors"].append(error_msg)
    session["attempt"] += 1
    if session["attempt"] > session["max_retries"]:
        # Exhausted retries — clear session
        state["retry_session"] = None
        return None
    # Generate new candidate ID for the retry
    session["candidate_id"] = next_program_id(state)
    return session


def clear_retry_session(state: dict) -> None:
    """Clear the active retry session (on success or give-up)."""
    state["retry_session"] = None


# ---------------------------------------------------------------------------
# Islands (multi-population diversity preservation)
# ---------------------------------------------------------------------------

DEFAULT_NUM_ISLANDS = 1  # 1 = disabled (single population)
DEFAULT_MIGRATION_INTERVAL = 10  # Migrate every N generations
DEFAULT_MIGRATION_RATE = 0.2  # Fraction of island to migrate


def assign_island(state: dict, parent_id: str | None) -> int:
    """Assign a program to an island. Children inherit parent's island."""
    config = state["config"]
    num_islands = config.get("num_islands", DEFAULT_NUM_ISLANDS)
    if num_islands <= 1:
        return 0

    # Inherit from parent
    if parent_id:
        parent = get_program_by_id(state, parent_id)
        if parent and parent.get("island_idx") is not None:
            return parent["island_idx"]

    # For new programs without a parent, distribute round-robin
    island_counts = [0] * num_islands
    for p in state["programs"]:
        idx = p.get("island_idx", 0)
        if 0 <= idx < num_islands:
            island_counts[idx] += 1
    return island_counts.index(min(island_counts))


def get_island_archive(state: dict, island_idx: int) -> list[str]:
    """Get archive program IDs for a specific island."""
    config = state["config"]
    archive_size = config.get("archive_size", 10)
    num_islands = config.get("num_islands", DEFAULT_NUM_ISLANDS)

    if num_islands <= 1:
        return state.get("archive_ids", [])

    # Per-island archive: top programs by score within this island
    island_programs = [
        p for p in state["programs"]
        if p.get("island_idx") == island_idx and p.get("correct", False)
    ]
    island_programs.sort(key=lambda p: p["combined_score"], reverse=True)
    per_island_size = max(1, archive_size // num_islands)
    return [p["id"] for p in island_programs[:per_island_size]]


def perform_migration(state: dict) -> list[dict]:
    """Migrate top programs between islands. Returns list of migration events."""
    config = state["config"]
    num_islands = config.get("num_islands", DEFAULT_NUM_ISLANDS)
    migration_rate = config.get("migration_rate", DEFAULT_MIGRATION_RATE)

    if num_islands <= 1:
        return []

    migrations = []
    for src_island in range(num_islands):
        island_progs = [
            p for p in state["programs"]
            if p.get("island_idx") == src_island
            and p.get("correct", False)
            and p.get("generation", 0) > 0  # Don't migrate seed
        ]
        if not island_progs:
            continue

        # Protect the best program on this island (elitism)
        island_progs.sort(key=lambda p: p["combined_score"], reverse=True)
        migratable = island_progs[1:]  # Skip the best

        num_migrants = max(1, int(len(island_progs) * migration_rate))
        migrants = random.sample(migratable, min(num_migrants, len(migratable)))

        for prog in migrants:
            dest_island = random.choice([i for i in range(num_islands) if i != src_island])
            old_island = prog["island_idx"]
            prog["island_idx"] = dest_island
            migrations.append({
                "program_id": prog["id"],
                "from_island": old_island,
                "to_island": dest_island,
                "score": prog["combined_score"],
            })

    return migrations


def should_migrate(state: dict) -> bool:
    """Check if migration should happen this generation."""
    config = state["config"]
    num_islands = config.get("num_islands", DEFAULT_NUM_ISLANDS)
    interval = config.get("migration_interval", DEFAULT_MIGRATION_INTERVAL)
    gen = state["generation"]
    return num_islands > 1 and interval > 0 and gen > 0 and gen % interval == 0


# ---------------------------------------------------------------------------
# Meta-Recommendations (periodic analysis of what's working)
# Ported from ShinkaEvolve's 3-step meta-summarizer
# ---------------------------------------------------------------------------

DEFAULT_META_INTERVAL = 5  # Generate meta-recs every N generations
DEFAULT_MAX_RECS = 5       # Number of recommendations to generate


def should_update_meta(state: dict) -> bool:
    """Check if meta-recommendations should be updated."""
    config = state["config"]
    interval = config.get("meta_interval", DEFAULT_META_INTERVAL)
    gen = state["generation"]
    last_meta_gen = state.get("last_meta_generation", 0)
    return interval > 0 and gen > 0 and (gen - last_meta_gen) >= interval


def build_meta_context(state: dict) -> str:
    """Build context for meta-recommendation generation.

    Produces a summary of recent programs for a subagent to analyze,
    matching ShinkaEvolve's 3-step meta process (condensed into one prompt).
    """
    config = state["config"]
    num_islands = config.get("num_islands", DEFAULT_NUM_ISLANDS)
    last_meta_gen = state.get("last_meta_generation", 0)

    # Recent programs since last meta update
    recent = [
        p for p in state["programs"]
        if p["generation"] > last_meta_gen
    ]
    recent.sort(key=lambda p: p["generation"])

    # Best program overall
    best_id = state.get("best_id")
    best_prog = get_program_by_id(state, best_id) if best_id else None

    lines = []
    lines.append("# Meta-Analysis Context")
    lines.append(f"Generations {last_meta_gen + 1} to {state['generation']}")
    lines.append(f"Best overall: {best_id} (score: {state.get('best_score')})")
    lines.append("")

    # Per-island summary if using islands
    if num_islands > 1:
        for island_idx in range(num_islands):
            island_progs = [p for p in state["programs"] if p.get("island_idx") == island_idx]
            correct = [p for p in island_progs if p.get("correct", False)]
            best_island = max(correct, key=lambda p: p["combined_score"]) if correct else None
            lines.append(f"## Island {island_idx}: {len(correct)} correct programs"
                        + (f", best={best_island['combined_score']:.4f}" if best_island else ""))
        lines.append("")

    lines.append("## Recent Programs")
    for p in recent[-20:]:  # Last 20
        status = "CORRECT" if p.get("correct") else "INCORRECT"
        island_str = f" island={p.get('island_idx')}" if num_islands > 1 else ""
        lines.append(
            f"- {p['id']} gen={p['generation']} score={p['combined_score']:.4f} "
            f"{status} type={p.get('mutation_type', '?')} "
            f"model={p.get('model_config', '?')}{island_str}"
        )
        if p.get("description"):
            lines.append(f"  desc: {p['description']}")

    # Archive summary
    lines.append("")
    lines.append("## Current Archive")
    for aid in state.get("archive_ids", []):
        ap = get_program_by_id(state, aid)
        if ap:
            lines.append(f"- {ap['id']} score={ap['combined_score']:.4f} "
                         f"type={ap.get('mutation_type', '?')} "
                         f"desc: {ap.get('description', '')[:80]}")

    # Previous recommendations
    prev_recs = state.get("meta_recommendations")
    if prev_recs:
        lines.append("")
        lines.append("## Previous Recommendations")
        lines.append(prev_recs)

    return "\n".join(lines)


META_REC_PROMPT = """You are analyzing an evolutionary code optimization run. Based on the context below, generate {n} actionable recommendations for the next batch of mutations.

For each recommendation:
1. Identify a specific algorithmic pattern, strategy, or approach worth trying
2. Explain WHY it might improve the score based on what's worked/failed so far
3. Be concrete — name specific techniques, data structures, or optimizations

Focus on:
- Patterns that led to score improvements (amplify what works)
- Approaches NOT yet tried (explore new directions)
- Common failure modes to avoid
- Cross-pollination between different successful approaches

{context}

Generate exactly {n} numbered recommendations (1. 2. 3. ...), each 2-3 sentences."""


# ---------------------------------------------------------------------------
# Parent Selection
# ---------------------------------------------------------------------------

def select_parent_weighted(
    archive_programs: list[dict],
    lambda_: float = 10.0,
) -> dict:
    """
    Weighted selection ported from ShinkaEvolve's WeightedSamplingStrategy.
    weight = sigmoid(λ * (score - median) / MAD) * 1/(1 + children_count)
    """
    scores = [p["combined_score"] for p in archive_programs]
    med = median(scores)
    mad = median_absolute_deviation(scores)
    scale = max(mad, 1e-6)

    weights = []
    for p in archive_programs:
        normalized = (p["combined_score"] - med) / scale
        s_i = stable_sigmoid(lambda_ * normalized)
        h_i = 1.0 / (1 + p.get("children_count", 0))
        weights.append(s_i * h_i)

    total = sum(weights)
    if total <= 0:
        # Uniform fallback
        return random.choice(archive_programs)

    r = random.random() * total
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return archive_programs[i]
    return archive_programs[-1]


def select_parent_power_law(
    archive_programs: list[dict],
    alpha: float = 1.0,
) -> dict:
    """
    Power-law selection ported from ShinkaEvolve's PowerLawSamplingStrategy.
    P(rank i) ∝ (i+1)^(-α), where programs are sorted by score descending.
    """
    sorted_progs = sorted(archive_programs, key=lambda p: p["combined_score"], reverse=True)
    weights = [(i + 1) ** (-alpha) for i in range(len(sorted_progs))]
    total = sum(weights)
    if total <= 0:
        return random.choice(sorted_progs)

    r = random.random() * total
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return sorted_progs[i]
    return sorted_progs[-1]


def select_inspirations(state: dict, parent_id: str, n: int = 2) -> list[dict]:
    """Select inspiration programs from the archive (excluding the parent)."""
    archive_ids = state.get("archive_ids", [])
    candidates = [
        get_program_by_id(state, aid)
        for aid in archive_ids
        if aid != parent_id
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return []
    return random.sample(candidates, min(n, len(candidates)))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_init(args):
    """Initialize a new evolution run from one or more seed programs."""
    base = get_base_dir()
    num_islands = args.num_islands

    # Resolve all initial program paths
    initial_paths = []
    for p in args.initial:
        path = Path(p).resolve()
        if not path.exists():
            print(f"Error: Initial program not found: {path}", file=sys.stderr)
            sys.exit(1)
        initial_paths.append(path)

    ext = initial_paths[0].suffix.lstrip(".")
    language = args.language or ext or "python"

    # Build eval command
    eval_cmd = args.eval_cmd

    # Create initial state
    state = {
        "config": {
            "language": language,
            "extension": ext,
            "population_size": args.population_size,
            "archive_size": args.archive_size,
            "selection_strategy": args.selection_strategy,
            "selection_lambda": 10.0,
            "power_law_alpha": 1.0,
            "eval_command": eval_cmd,
            "num_islands": num_islands,
            "migration_interval": args.migration_interval,
            "migration_rate": args.migration_rate,
            "meta_interval": args.meta_interval,
            "meta_max_recs": args.meta_max_recs,
            "evolve_prompts": args.evolve_prompts,
            "prompt_archive_size": args.prompt_archive_size,
            "prompt_evolution_interval": args.prompt_evolution_interval,
        },
        "generation": 0,
        "programs": [],
        "archive_ids": [],
        "best_id": None,
        "best_score": None,
        "model_stats": default_model_stats(),
        "meta_recommendations": None,
        "last_meta_generation": 0,
        "prompt_archive": default_prompt_archive(args.task_sys_msg) if args.evolve_prompts else [],
        "last_prompt_evolution_gen": 0,
    }
    save_state(state)

    # Evaluate each seed program
    archive_ids = []
    best_id = None
    best_score = None

    for idx, initial_path in enumerate(initial_paths):
        prog_id = f"prog_{idx:04d}"
        prog_dir = base / PROGRAMS_DIR / prog_id
        prog_dir.mkdir(parents=True, exist_ok=True)
        prog_file = prog_dir / f"main.{ext}"
        shutil.copy2(initial_path, prog_file)
        print(f"\nSeed {idx + 1}/{len(initial_paths)}: {initial_path.name} → {prog_file}")

        # Assign to island (round-robin across islands)
        island_idx = idx % num_islands if num_islands > 1 else 0

        print(f"  Evaluating...")
        metrics, correct = run_evaluation(state, str(prog_file))
        score = metrics.get("combined_score", 0.0)

        program_entry = {
            "id": prog_id,
            "parent_id": None,
            "generation": 0,
            "combined_score": score,
            "correct": correct,
            "public_metrics": metrics.get("public", {}),
            "in_archive": correct,
            "children_count": 0,
            "island_idx": island_idx,
            "mutation_type": "initial",
            "description": f"Seed: {initial_path.name}",
            "timestamp": time.time(),
        }
        state["programs"].append(program_entry)

        if correct:
            archive_ids.append(prog_id)
            if best_score is None or score > best_score:
                best_id = prog_id
                best_score = score

        # Save metrics to program directory
        with open(prog_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(prog_dir / "correct.json", "w") as f:
            json.dump({"correct": correct}, f, indent=2)

        island_str = f", island {island_idx}" if num_islands > 1 else ""
        print(f"  Score: {score}, Correct: {correct}{island_str}")

    state["archive_ids"] = archive_ids
    state["best_id"] = best_id
    state["best_score"] = best_score
    save_state(state)

    print(f"\nInitialized claude-evolve:")
    print(f"  Seed programs: {len(initial_paths)}")
    print(f"  Correct: {len(archive_ids)}/{len(initial_paths)}")
    print(f"  Best: {best_id} (score: {best_score})")
    print(f"  State saved to: {base / STATE_FILE}")


def cmd_select(args):
    """Select a parent and inspirations for the next mutation."""
    state = load_state()
    config = state["config"]
    ext = config.get("extension", "py")

    # Check for active retry session
    session = get_active_session(state)
    parent = None
    if args.retry and session:
        # Reuse same parent and model from the retry session
        parent = get_program_by_id(state, session["parent_id"])
        if not parent:
            print("Error: Retry session parent not found. Starting fresh.", file=sys.stderr)
            clear_retry_session(state)
            session = None

    if args.retry and session and parent:
        # Advance the retry session
        new_session = advance_retry_session(state, error_msg=args.retry_error)
        if new_session is None:
            print("Max retries exhausted. Starting fresh selection.", file=sys.stderr)
            save_state(state)
            session = None
        else:
            session = new_session
            nid = session["candidate_id"]
            candidate_dir = get_program_dir(nid)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_file = candidate_dir / f"main.{ext}"

            parent_code_path = get_program_dir(parent["id"]) / f"main.{ext}"
            print("=" * 60)
            print(f"RETRY ATTEMPT {session['attempt']}/{session['max_retries']}")
            print("=" * 60)
            print(f"  Same parent:  {parent['id']} (score: {parent['combined_score']})")
            print(f"  Same model:   {session['model_label']}")
            print(f"  Parent path:  {parent_code_path}")
            if session["errors"]:
                print(f"  Previous errors:")
                for err in session["errors"]:
                    print(f"    - {err[:120]}")

            # Select new inspirations for variety
            inspirations = select_inspirations(state, parent["id"], n=2)
            for i, insp in enumerate(inspirations):
                insp_path = get_program_dir(insp["id"]) / f"main.{ext}"
                print(f"  Inspiration {i+1}: {insp['id']} (score: {insp['combined_score']}, path: {insp_path})")

            print()
            print("=" * 60)
            print("MODEL CONFIG (retry — same arm)")
            print("=" * 60)
            print(f"  Label:   {session['model_label']}")
            print(f"  Model:   {session['model']}")
            print(f"  Effort:  {session['effort']}")

            print()
            print("=" * 60)
            print("NEXT CANDIDATE")
            print("=" * 60)
            print(f"  ID:        {nid}")
            print(f"  Directory: {candidate_dir}")
            print(f"  Write to:  {candidate_file}")
            print(f"  Generation: {state['generation'] + 1}")
            print()
            print(f"Current best: {state.get('best_id')} (score: {state.get('best_score')})")
            save_state(state)
            return

    # --- Fresh selection (no retry) ---

    # Clear any stale session
    clear_retry_session(state)

    num_islands = config.get("num_islands", DEFAULT_NUM_ISLANDS)

    # Pick an island (round-robin based on generation)
    if num_islands > 1:
        target_island = state["generation"] % num_islands
        archive_programs = [
            p for p in state["programs"]
            if p.get("island_idx") == target_island and p.get("correct", False)
        ]
        # Fallback to global if island is empty
        if not archive_programs:
            archive_programs = [p for p in state["programs"] if p.get("correct", False)]
    else:
        target_island = 0
        # Get archive programs (correct only)
        archive_programs = [
            p for p in state["programs"]
            if p["id"] in state.get("archive_ids", []) and p.get("correct", False)
        ]
        if not archive_programs:
            archive_programs = [p for p in state["programs"] if p.get("correct", False)]

    if not archive_programs:
        print("Error: No correct programs in population. Cannot select parent.", file=sys.stderr)
        sys.exit(1)

    # Select parent
    strategy = args.strategy or config.get("selection_strategy", "weighted")
    if strategy == "weighted":
        parent = select_parent_weighted(
            archive_programs,
            lambda_=config.get("selection_lambda", 10.0),
        )
    elif strategy == "power_law":
        parent = select_parent_power_law(
            archive_programs,
            alpha=config.get("power_law_alpha", 1.0),
        )
    else:
        print(f"Error: Unknown strategy '{strategy}'", file=sys.stderr)
        sys.exit(1)

    # Select inspirations
    inspirations = select_inspirations(state, parent["id"], n=2)

    # Determine next candidate ID and path
    nid = next_program_id(state)
    candidate_dir = get_program_dir(nid)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_file = candidate_dir / f"main.{ext}"

    # Select model config via UCB1 bandit
    model_stats = state.get("model_stats", default_model_stats())
    chosen_model = select_model_ucb1(
        model_stats,
        total_generations=state["generation"],
    )

    # Start a retry session for this generation
    start_retry_session(state, parent["id"], chosen_model, nid)
    save_state(state)

    # Print structured context for Claude Code
    parent_code_path = get_program_dir(parent["id"]) / f"main.{ext}"
    print("=" * 60)
    print("PARENT PROGRAM")
    print("=" * 60)
    print(f"  ID:      {parent['id']}")
    print(f"  Score:   {parent['combined_score']}")
    print(f"  Gen:     {parent['generation']}")
    print(f"  Type:    {parent.get('mutation_type', 'unknown')}")
    print(f"  Desc:    {parent.get('description', '')}")
    print(f"  Path:    {parent_code_path}")
    if parent.get("public_metrics"):
        print(f"  Metrics: {json.dumps(parent['public_metrics'], indent=4)}")

    for i, insp in enumerate(inspirations):
        insp_path = get_program_dir(insp["id"]) / f"main.{ext}"
        print()
        print(f"{'=' * 60}")
        print(f"INSPIRATION {i + 1}")
        print(f"{'=' * 60}")
        print(f"  ID:      {insp['id']}")
        print(f"  Score:   {insp['combined_score']}")
        print(f"  Gen:     {insp['generation']}")
        print(f"  Type:    {insp.get('mutation_type', 'unknown')}")
        print(f"  Desc:    {insp.get('description', '')}")
        print(f"  Path:    {insp_path}")
        if insp.get("public_metrics"):
            print(f"  Metrics: {json.dumps(insp['public_metrics'], indent=4)}")

    print()
    print("=" * 60)
    print("MODEL CONFIG (bandit-selected)")
    print("=" * 60)
    print(f"  Label:   {chosen_model['label']}")
    print(f"  Model:   {chosen_model['model']}")
    print(f"  Effort:  {chosen_model['effort']}")
    print(f"  Pulls:   {chosen_model['total_pulls']}")
    if chosen_model["total_pulls"] > 0:
        avg = chosen_model["total_reward"] / chosen_model["total_pulls"]
        print(f"  Avg reward: {avg:.4f}")
    print(f"  Retries: up to {MAX_RETRIES} (use 'select --retry' on failure)")

    # Show selected prompt if prompt evolution is enabled
    prompt_archive = state.get("prompt_archive", [])
    selected_prompt_id = None
    if prompt_archive and config.get("evolve_prompts", False):
        selected_prompt = select_prompt_ucb(prompt_archive)
        selected_prompt_id = selected_prompt.get("id")
        print()
        print("=" * 60)
        print("SYSTEM PROMPT (UCB-selected)")
        print("=" * 60)
        print(f"  ID:      {selected_prompt_id}")
        print(f"  Fitness: {selected_prompt.get('fitness', 'N/A')}")
        print(f"  Name:    {selected_prompt.get('name', '?')}")
        text = selected_prompt.get("prompt_text", "")
        print(f"  Text:    {text[:300]}{'...' if len(text) > 300 else ''}")
        print(f"  (Pass --prompt-id {selected_prompt_id} to update)")

    # Show meta-recommendations if available
    meta_recs = state.get("meta_recommendations")
    if meta_recs:
        print()
        print("=" * 60)
        print("META-RECOMMENDATIONS (guide your mutation)")
        print("=" * 60)
        print(meta_recs)

    if num_islands > 1:
        print()
        print(f"  Target island: {target_island}")

    print()
    print("=" * 60)
    print("NEXT CANDIDATE")
    print("=" * 60)
    print(f"  ID:        {nid}")
    print(f"  Directory: {candidate_dir}")
    print(f"  Write to:  {candidate_file}")
    print(f"  Generation: {state['generation'] + 1}")
    print()
    print(f"Current best: {state.get('best_id')} (score: {state.get('best_score')})")


def run_evaluation(state: dict, candidate_path: str) -> tuple[dict, bool]:
    """Run the user's evaluator and parse results."""
    config = state["config"]
    eval_cmd = config["eval_command"]

    # Create a temp results dir next to the candidate
    candidate = Path(candidate_path).resolve()
    results_dir = candidate.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Format the command
    cmd = eval_cmd.format(
        program_path=str(candidate),
        results_dir=str(results_dir),
    )

    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=300,
    )

    if result.stdout:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr, file=sys.stderr)

    # Parse metrics.json and correct.json
    metrics_file = results_dir / "metrics.json"
    correct_file = results_dir / "correct.json"

    metrics = {}
    correct = False

    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        print("Warning: metrics.json not found", file=sys.stderr)
        metrics = {"combined_score": 0.0}

    if correct_file.exists():
        with open(correct_file) as f:
            correct_data = json.load(f)
            correct = correct_data.get("correct", False)
            if not correct:
                print(f"  Error: {correct_data.get('error', 'unknown')}")
    else:
        print("Warning: correct.json not found", file=sys.stderr)
        correct = result.returncode == 0

    return metrics, correct


def cmd_check_novelty(args):
    """Check if a candidate is novel compared to archive programs.

    Two-stage check (mirrors ShinkaEvolve's NoveltyJudge):
    1. Fast difflib similarity against parent + top archive programs
    2. If similarity > threshold, outputs context for LLM judge (haiku subagent)
    """
    state = load_state()
    config = state["config"]
    ext = config.get("extension", "py")

    candidate_path = Path(args.candidate).resolve()
    if not candidate_path.exists():
        print(f"Error: Candidate not found: {candidate_path}", file=sys.stderr)
        sys.exit(1)

    candidate_code = candidate_path.read_text(encoding="utf-8")
    candidate_block = extract_evolve_block(candidate_code)

    # Compare against parent
    parent_id = args.parent_id
    comparisons = []

    # Add parent
    parent_path = get_program_dir(parent_id) / f"main.{ext}"
    if parent_path.exists():
        parent_code = parent_path.read_text(encoding="utf-8")
        parent_block = extract_evolve_block(parent_code)
        sim = compute_code_similarity(candidate_block, parent_block)
        comparisons.append(("parent", parent_id, sim, parent_block))

    # Compare against top archive programs
    archive_ids = state.get("archive_ids", [])
    for aid in archive_ids:
        if aid == parent_id:
            continue
        archive_path = get_program_dir(aid) / f"main.{ext}"
        if archive_path.exists():
            archive_code = archive_path.read_text(encoding="utf-8")
            archive_block = extract_evolve_block(archive_code)
            sim = compute_code_similarity(candidate_block, archive_block)
            comparisons.append(("archive", aid, sim, archive_block))

    # Sort by similarity descending
    comparisons.sort(key=lambda x: x[2], reverse=True)

    threshold = args.threshold or NOVELTY_SIMILARITY_THRESHOLD
    max_sim = comparisons[0][2] if comparisons else 0.0
    most_similar_role, most_similar_id, _, most_similar_block = comparisons[0] if comparisons else ("", "", 0, "")

    print("=" * 60)
    print("NOVELTY CHECK")
    print("=" * 60)
    print(f"  Candidate: {candidate_path.name}")
    print(f"  Threshold: {threshold}")
    print()
    print("  Similarity scores:")
    for role, pid, sim, _ in comparisons[:5]:
        marker = " *** HIGH" if sim > threshold else ""
        print(f"    {role:<8} {pid:<12} {sim:.4f}{marker}")

    # Stage 2: Embedding-based similarity (if model available)
    candidate_embedding = compute_embedding(candidate_block)
    embed_max_sim = 0.0
    embed_most_similar_id = ""
    if candidate_embedding:
        embed_sims = find_most_similar_by_embedding(candidate_embedding)
        if embed_sims:
            embed_most_similar_id, embed_max_sim = embed_sims[0]
            print()
            print("  Embedding similarity (cosine):")
            for pid, sim in embed_sims[:5]:
                marker = " *** HIGH" if sim > EMBEDDING_SIMILARITY_THRESHOLD else ""
                print(f"    {pid:<12} {sim:.4f}{marker}")

    # Combined verdict
    is_novel_difflib = max_sim <= threshold
    is_novel_embed = embed_max_sim <= EMBEDDING_SIMILARITY_THRESHOLD

    if is_novel_difflib and is_novel_embed:
        print()
        print(f"  Verdict: NOVEL (difflib={max_sim:.4f}, embed={embed_max_sim:.4f})")
        print(f"  Action:  Proceed to evaluation")
    elif not is_novel_difflib or not is_novel_embed:
        print()
        print(f"  Verdict: POTENTIALLY NOT NOVEL (similarity {max_sim:.4f} > {threshold})")
        print(f"  Action:  Use LLM judge to confirm (spawn haiku subagent)")
        print()
        print("=" * 60)
        print("LLM JUDGE CONTEXT")
        print("=" * 60)
        print(f"  Compare candidate against: {most_similar_id} ({most_similar_role})")
        print(f"  Most similar program path: {get_program_dir(most_similar_id) / f'main.{ext}'}")
        print(f"  Candidate path: {candidate_path}")
        print()
        print("  Judge prompt (use with haiku subagent):")
        print(f"  System: {NOVELTY_JUDGE_PROMPT[:200]}...")
        print()
        print(f"  If judge says NOT_NOVEL: reject and retry (select --retry)")
        print(f"  If judge says NOVEL: proceed to evaluation")


def cmd_evaluate(args):
    """Evaluate a candidate program."""
    state = load_state()
    candidate_path = Path(args.candidate).resolve()

    if not candidate_path.exists():
        print(f"Error: Candidate not found: {candidate_path}", file=sys.stderr)
        sys.exit(1)

    metrics, correct = run_evaluation(state, str(candidate_path))

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Score:   {metrics.get('combined_score', 'N/A')}")
    print(f"  Correct: {correct}")
    if metrics.get("public"):
        print(f"  Metrics: {json.dumps(metrics['public'], indent=4)}")

    # Save to candidate directory
    candidate_dir = candidate_path.parent
    with open(candidate_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(candidate_dir / "correct.json", "w") as f:
        json.dump({"correct": correct}, f, indent=2)


def cmd_update(args):
    """Add evaluated candidate to population and update archive."""
    state = load_state()
    config = state["config"]

    candidate_dir = Path(args.candidate_dir).resolve()
    metrics_file = candidate_dir / "metrics.json"
    correct_file = candidate_dir / "correct.json"

    if not metrics_file.exists() or not correct_file.exists():
        print("Error: Run 'evaluate' first — metrics.json/correct.json not found.", file=sys.stderr)
        sys.exit(1)

    with open(metrics_file) as f:
        metrics = json.load(f)
    with open(correct_file) as f:
        correct = json.load(f).get("correct", False)

    # Determine program ID from directory name
    prog_id = candidate_dir.name
    score = metrics.get("combined_score", 0.0)

    # Increment parent's children count
    parent = get_program_by_id(state, args.parent_id)
    if parent:
        parent["children_count"] = parent.get("children_count", 0) + 1

    # Assign island
    island_idx = assign_island(state, args.parent_id)

    # Add program to population
    program_entry = {
        "id": prog_id,
        "parent_id": args.parent_id,
        "generation": state["generation"] + 1,
        "combined_score": score,
        "correct": correct,
        "public_metrics": metrics.get("public", {}),
        "in_archive": False,
        "children_count": 0,
        "island_idx": island_idx,
        "mutation_type": args.mutation_type,
        "description": args.description,
        "model_config": args.model_config,
        "timestamp": time.time(),
    }
    state["programs"].append(program_entry)
    state["generation"] += 1

    # Update model bandit stats (asymmetric reward like ShinkaEvolve)
    if args.model_config:
        model_stats = state.get("model_stats", default_model_stats())
        parent_score = parent["combined_score"] if parent else 0.0
        delta = score - parent_score

        # Asymmetric reward: only positive improvements count.
        # Incorrect programs get 0 reward (ShinkaEvolve imputes worst).
        # This prevents a single bad result from killing an arm.
        if correct:
            reward = max(0.0, delta / max(abs(parent_score), 1e-6))
        else:
            reward = 0.0

        for m in model_stats:
            if m["label"] == args.model_config:
                m["total_pulls"] += 1
                m["total_reward"] += reward
                m["best_score_delta"] = max(m["best_score_delta"], delta)
                if correct:
                    m["n_correct"] = m.get("n_correct", 0) + 1
                else:
                    m["n_incorrect"] = m.get("n_incorrect", 0) + 1
                break
        state["model_stats"] = model_stats

    # Update archive: top archive_size correct programs by score
    archive_size = config.get("archive_size", 10)
    correct_programs = [p for p in state["programs"] if p.get("correct", False)]
    correct_programs.sort(key=lambda p: p["combined_score"], reverse=True)

    new_archive_ids = [p["id"] for p in correct_programs[:archive_size]]

    # Update in_archive flags
    for p in state["programs"]:
        p["in_archive"] = p["id"] in new_archive_ids

    state["archive_ids"] = new_archive_ids

    # Update best
    if correct and (state.get("best_score") is None or score > state["best_score"]):
        state["best_id"] = prog_id
        state["best_score"] = score

    # Compute and store embedding for this program
    ext = config.get("extension", "py")
    code_path = candidate_dir / f"main.{ext}"
    if code_path.exists():
        code_text = code_path.read_text(encoding="utf-8")
        emb = compute_and_store_embedding(prog_id, code_text)
        if emb:
            print(f"  Embedding computed (dim={len(emb)})")

    # Update prompt fitness if prompt evolution is enabled
    if args.prompt_id:
        update_prompt_fitness(state, args.prompt_id, score, correct)

    # Clear retry session on successful update
    clear_retry_session(state)

    # Check for island migration
    migration_events = []
    if should_migrate(state):
        migration_events = perform_migration(state)
        # Rebuild archive after migration
        correct_programs = [p for p in state["programs"] if p.get("correct", False)]
        correct_programs.sort(key=lambda p: p["combined_score"], reverse=True)
        new_archive_ids = [p["id"] for p in correct_programs[:archive_size]]
        state["archive_ids"] = new_archive_ids

    save_state(state)

    # Print summary
    entered_archive = prog_id in new_archive_ids
    parent_score = parent["combined_score"] if parent else None
    num_islands = config.get("num_islands", DEFAULT_NUM_ISLANDS)

    print()
    print("=" * 60)
    print("POPULATION UPDATED")
    print("=" * 60)
    print(f"  Program:  {prog_id}")
    print(f"  Score:    {score}")
    print(f"  Correct:  {correct}")
    print(f"  Archive:  {'YES' if entered_archive else 'no'}")
    if num_islands > 1:
        print(f"  Island:   {island_idx}")
    if parent_score is not None:
        delta = score - parent_score
        print(f"  vs Parent: {'+' if delta >= 0 else ''}{delta:.6f}")
    print(f"  Generation: {state['generation']}")
    print(f"  Best:     {state['best_id']} (score: {state['best_score']})")
    print(f"  Archive size: {len(new_archive_ids)}/{archive_size}")

    # Report migration
    if migration_events:
        print()
        print(f"  MIGRATION ({len(migration_events)} programs moved):")
        for m in migration_events:
            print(f"    {m['program_id']} (score={m['score']:.4f}): "
                  f"island {m['from_island']} → {m['to_island']}")

    # Flag if meta-recommendations should be updated
    if should_update_meta(state):
        print()
        print("=" * 60)
        print("META-RECOMMENDATIONS DUE")
        print("=" * 60)
        print("  Run: python claude-evolve/evolve.py meta-update")

    # Flag if prompt evolution should trigger
    if should_evolve_prompt(state):
        print()
        print("=" * 60)
        print("PROMPT EVOLUTION DUE")
        print("=" * 60)
        print("  Run: python claude-evolve/evolve.py prompt-evolve")


def cmd_prompt_evolve(args):
    """Evolve the system prompt. Outputs context for a subagent,
    or saves a new prompt directly with --set."""
    state = load_state()

    if args.set_prompt:
        # Save a new evolved prompt
        pid = next_prompt_id(state)
        parent_id = args.parent_prompt_id or (
            state["prompt_archive"][-1]["id"] if state.get("prompt_archive") else None
        )
        new_prompt = {
            "id": pid,
            "prompt_text": args.set_prompt,
            "name": args.name or "evolved",
            "description": args.desc or "Evolved system prompt",
            "parent_id": parent_id,
            "generation": len(state.get("prompt_archive", [])),
            "patch_type": args.patch_type or "full",
            "program_count": 0,
            "correct_program_count": 0,
            "total_percentile": 0.0,
            "fitness": 0.6,  # Optimistic prior for new prompts
            "program_scores": [],
            "in_archive": True,
        }
        archive = state.get("prompt_archive", [])
        max_size = state["config"].get("prompt_archive_size", DEFAULT_PROMPT_ARCHIVE_SIZE)
        if len(archive) >= max_size:
            # Replace worst by fitness
            worst = min(archive, key=lambda p: p.get("fitness", 0))
            archive.remove(worst)
        archive.append(new_prompt)
        state["prompt_archive"] = archive
        state["last_prompt_evolution_gen"] = state["generation"]
        save_state(state)
        print(f"Prompt {pid} added to archive (size: {len(archive)}/{max_size})")
        return

    # Output context for subagent to evolve the prompt
    archive = state.get("prompt_archive", [])
    if not archive:
        print("No prompts in archive. Initialize with --task-sys-msg in init.", file=sys.stderr)
        sys.exit(1)

    parent = select_prompt_ucb(archive)
    context = build_prompt_evolution_context(state, parent)

    print("=" * 60)
    print("PROMPT EVOLUTION")
    print("=" * 60)
    print(f"  Parent prompt: {parent['id']} (fitness: {parent.get('fitness', 'N/A')})")
    print(f"  Archive size: {len(archive)}")
    print()
    print("Spawn a subagent with this context:")
    print("-" * 60)
    print(f"System: {PROMPT_EVO_SYSTEM}")
    print()
    print(context)
    print("-" * 60)
    print()
    print("Then save the result:")
    print(f'  python claude-evolve/evolve.py prompt-evolve '
          f'--set-prompt "..." --name "..." --desc "..." '
          f'--parent-prompt-id {parent["id"]}')


def cmd_prompt_show(_args):
    """Show the prompt archive."""
    state = load_state()
    archive = state.get("prompt_archive", [])

    print("=" * 60)
    print("PROMPT ARCHIVE")
    print("=" * 60)
    if not archive:
        print("  (empty — prompt evolution not enabled)")
        return

    for p in sorted(archive, key=lambda x: x.get("fitness", 0), reverse=True):
        print(f"\n  {p['id']} (fitness: {p.get('fitness', 'N/A'):.4f}, "
              f"programs: {p.get('program_count', 0)}, "
              f"correct: {p.get('correct_program_count', 0)})")
        print(f"  Name: {p.get('name', '?')}")
        print(f"  Desc: {p.get('description', '')[:100]}")
        text = p.get("prompt_text", "")
        print(f"  Text: {text[:150]}{'...' if len(text) > 150 else ''}")


def cmd_meta_update(args):
    """Generate or update meta-recommendations.

    Builds context from recent programs and outputs a prompt for a subagent
    to generate recommendations. If --set is provided, saves recommendations
    directly (used after a subagent produces them).
    """
    state = load_state()
    config = state["config"]

    if args.set:
        # Save recommendations provided by the user/subagent
        state["meta_recommendations"] = args.set
        state["last_meta_generation"] = state["generation"]
        save_state(state)
        print("Meta-recommendations updated.")
        print(f"  Generation: {state['generation']}")
        print(f"  Recommendations:\n{args.set}")
        return

    # Build context and output prompt for subagent
    context = build_meta_context(state)
    n = config.get("meta_max_recs", DEFAULT_MAX_RECS)
    prompt = META_REC_PROMPT.format(n=n, context=context)

    print("=" * 60)
    print("META-RECOMMENDATION UPDATE")
    print("=" * 60)
    print(f"  Generation: {state['generation']}")
    print(f"  Last meta update: gen {state.get('last_meta_generation', 0)}")
    print(f"  Programs since last update: {state['generation'] - state.get('last_meta_generation', 0)}")
    print()
    print("Spawn a subagent (haiku recommended for speed) with this prompt:")
    print()
    print("-" * 60)
    print(prompt)
    print("-" * 60)
    print()
    print("Then save the result:")
    print('  python claude-evolve/evolve.py meta-update --set "1. ... 2. ... 3. ..."')

    # Also output current recs if any
    current = state.get("meta_recommendations")
    if current:
        print()
        print("Current recommendations (will be replaced):")
        print(current)


def cmd_meta_show(_args):
    """Show current meta-recommendations."""
    state = load_state()
    recs = state.get("meta_recommendations")
    last_gen = state.get("last_meta_generation", 0)

    print("=" * 60)
    print("META-RECOMMENDATIONS")
    print("=" * 60)
    print(f"  Last updated: generation {last_gen}")
    print()
    if recs:
        print(recs)
    else:
        print("  (none yet — run 'meta-update' to generate)")


def cmd_status(_args):
    """Print current evolution status."""
    state = load_state()
    config = state["config"]

    total = len(state["programs"])
    correct_count = sum(1 for p in state["programs"] if p.get("correct", False))
    archive_ids = state.get("archive_ids", [])

    print("=" * 60)
    print("CLAUDE-EVOLVE STATUS")
    print("=" * 60)
    print(f"  Generation:    {state['generation']}")
    print(f"  Total programs: {total}")
    print(f"  Correct:       {correct_count}/{total}")
    print(f"  Archive:       {len(archive_ids)}/{config.get('archive_size', 10)}")
    print(f"  Best:          {state.get('best_id')} (score: {state.get('best_score')})")
    print(f"  Language:      {config.get('language', 'unknown')}")
    print(f"  Strategy:      {config.get('selection_strategy', 'weighted')}")
    num_islands = config.get("num_islands", DEFAULT_NUM_ISLANDS)
    if num_islands > 1:
        print(f"  Islands:       {num_islands}")
        print(f"  Migration:     every {config.get('migration_interval', DEFAULT_MIGRATION_INTERVAL)} gens, "
              f"rate={config.get('migration_rate', DEFAULT_MIGRATION_RATE)}")
    meta_recs = state.get("meta_recommendations")
    if meta_recs:
        print(f"  Meta-recs:     updated at gen {state.get('last_meta_generation', 0)}")
    print()

    if archive_ids:
        print("Archive:")
        for aid in archive_ids:
            p = get_program_by_id(state, aid)
            if p:
                marker = " <-- BEST" if aid == state.get("best_id") else ""
                island_str = f"  I{p.get('island_idx', 0)}" if num_islands > 1 else ""
                print(f"  {p['id']}  score={p['combined_score']:.6f}  "
                      f"gen={p['generation']}  type={p.get('mutation_type', '?')}"
                      f"{island_str}{marker}")

    # Model ensemble stats
    model_stats = state.get("model_stats", [])
    used_models = [m for m in model_stats if m["total_pulls"] > 0]
    if used_models:
        print()
        min_pulls = state.get("config", {}).get("bandit_min_pulls", BANDIT_MIN_PULLS)
        warmup_done = all(m["total_pulls"] >= min_pulls for m in model_stats)
        phase = "exploiting" if warmup_done else f"warming up (need {min_pulls} pulls/arm)"
        print(f"Model Ensemble (Asymmetric UCB1 — {phase}):")
        print(f"  {'Label':<18} {'Pulls':<7} {'OK/Fail':<9} {'Avg Reward':<12} {'Best Delta':<12}")
        print(f"  {'-'*58}")
        for m in sorted(used_models, key=lambda x: x["total_reward"] / max(x["total_pulls"], 1), reverse=True):
            avg = m["total_reward"] / m["total_pulls"]
            ok = m.get("n_correct", 0)
            fail = m.get("n_incorrect", 0)
            print(f"  {m['label']:<18} {m['total_pulls']:<7} {ok}/{fail:<6} {avg:<12.4f} {m['best_score_delta']:<12.6f}")


def cmd_leaderboard(args):
    """Print top programs sorted by score."""
    state = load_state()
    top_n = args.top

    correct_programs = [p for p in state["programs"] if p.get("correct", False)]
    correct_programs.sort(key=lambda p: p["combined_score"], reverse=True)

    print("=" * 60)
    print(f"LEADERBOARD (top {top_n})")
    print("=" * 60)
    print(f"{'Rank':<5} {'ID':<12} {'Score':<12} {'Gen':<5} {'Parent':<12} {'Type':<10} {'Description'}")
    print("-" * 90)

    for i, p in enumerate(correct_programs[:top_n]):
        parent = p.get('parent_id') or '-'
        print(f"{i+1:<5} {p['id']:<12} {p['combined_score']:<12.6f} "
              f"{p['generation']:<5} {parent:<12} "
              f"{p.get('mutation_type', '?'):<10} {p.get('description', '')[:40]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="claude-evolve: Evolutionary code optimization for Claude Code"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # init
    p_init = subparsers.add_parser("init", help="Initialize from seed program(s)")
    p_init.add_argument("--initial", required=True, nargs="+",
                        help="Path(s) to seed program(s). Multiple files seed the archive with diverse starting points.")
    p_init.add_argument("--eval-cmd", required=True,
                        help="Evaluation command with {program_path} and {results_dir} placeholders")
    p_init.add_argument("--language", default=None, help="Language (default: inferred from extension)")
    p_init.add_argument("--population-size", type=int, default=20)
    p_init.add_argument("--archive-size", type=int, default=10)
    p_init.add_argument("--selection-strategy", default="weighted",
                        choices=["weighted", "power_law"])
    p_init.add_argument("--num-islands", type=int, default=1,
                        help="Number of islands (1=disabled, 2-4 typical)")
    p_init.add_argument("--migration-interval", type=int, default=10,
                        help="Migrate between islands every N generations")
    p_init.add_argument("--migration-rate", type=float, default=0.2,
                        help="Fraction of island population to migrate")
    p_init.add_argument("--meta-interval", type=int, default=5,
                        help="Generate meta-recommendations every N generations")
    p_init.add_argument("--meta-max-recs", type=int, default=5,
                        help="Number of meta-recommendations to generate")
    p_init.add_argument("--evolve-prompts", action="store_true",
                        help="Enable prompt co-evolution")
    p_init.add_argument("--task-sys-msg", default=None,
                        help="Initial task system message (seed prompt for evolution)")
    p_init.add_argument("--prompt-archive-size", type=int, default=DEFAULT_PROMPT_ARCHIVE_SIZE,
                        help="Max prompts in archive")
    p_init.add_argument("--prompt-evolution-interval", type=int,
                        default=DEFAULT_PROMPT_EVOLUTION_INTERVAL,
                        help="Evolve prompt every N generations")

    # select
    p_select = subparsers.add_parser("select", help="Select parent for next mutation")
    p_select.add_argument("--strategy", default=None, choices=["weighted", "power_law"])
    p_select.add_argument("--retry", action="store_true",
                          help="Retry with same parent/model from active session")
    p_select.add_argument("--retry-error", default=None,
                          help="Error message from previous attempt (for context)")

    # check-novelty
    p_novelty = subparsers.add_parser("check-novelty", help="Check if candidate is novel")
    p_novelty.add_argument("--candidate", required=True, help="Path to candidate program file")
    p_novelty.add_argument("--parent-id", required=True, help="Parent program ID to compare against")
    p_novelty.add_argument("--threshold", type=float, default=None,
                           help=f"Similarity threshold (default: {NOVELTY_SIMILARITY_THRESHOLD})")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a candidate program")
    p_eval.add_argument("--candidate", required=True, help="Path to candidate program file")

    # update
    p_update = subparsers.add_parser("update", help="Add evaluated candidate to population")
    p_update.add_argument("--candidate-dir", required=True, help="Directory containing candidate + metrics")
    p_update.add_argument("--parent-id", required=True, help="Parent program ID")
    p_update.add_argument("--mutation-type", required=True,
                          choices=["targeted", "full_rewrite", "crossover", "fix"],
                          help="Type of mutation applied")
    p_update.add_argument("--description", required=True, help="Brief description of the change")
    p_update.add_argument("--model-config", default=None,
                          help="Model config label used (e.g. 'opus/high'). Updates bandit stats.")
    p_update.add_argument("--prompt-id", default=None,
                          help="Prompt ID used for this mutation (for prompt fitness tracking)")

    # prompt-evolve
    p_pe = subparsers.add_parser("prompt-evolve", help="Evolve the system prompt")
    p_pe.add_argument("--set-prompt", default=None, help="Save a new evolved prompt directly")
    p_pe.add_argument("--name", default=None, help="Short name for the new prompt")
    p_pe.add_argument("--desc", default=None, help="Description of changes")
    p_pe.add_argument("--parent-prompt-id", default=None, help="Parent prompt ID")
    p_pe.add_argument("--patch-type", default="full", choices=["diff", "full"])

    # prompt-show
    subparsers.add_parser("prompt-show", help="Show prompt archive")

    # meta-update
    p_meta = subparsers.add_parser("meta-update", help="Generate/update meta-recommendations")
    p_meta.add_argument("--set", default=None,
                        help="Set recommendations directly (from subagent output)")

    # meta-show
    subparsers.add_parser("meta-show", help="Show current meta-recommendations")

    # status
    subparsers.add_parser("status", help="Show current evolution status")

    # leaderboard
    p_lb = subparsers.add_parser("leaderboard", help="Show top programs")
    p_lb.add_argument("--top", type=int, default=10, help="Number of programs to show")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "init": cmd_init,
        "select": cmd_select,
        "check-novelty": cmd_check_novelty,
        "evaluate": cmd_evaluate,
        "update": cmd_update,
        "prompt-evolve": cmd_prompt_evolve,
        "prompt-show": cmd_prompt_show,
        "meta-update": cmd_meta_update,
        "meta-show": cmd_meta_show,
        "status": cmd_status,
        "leaderboard": cmd_leaderboard,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
