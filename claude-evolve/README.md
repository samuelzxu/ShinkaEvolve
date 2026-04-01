# claude-evolve

Evolutionary code optimization powered by Claude Code. No API keys required.

claude-evolve replicates [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)'s core evolution loop but uses Claude Code itself as the mutation operator. A lightweight Python orchestrator manages population state, selection, and evaluation, while Claude Code (via the `claude` CLI) proposes code mutations.

## How It Works

```
select parent (weighted/power-law)
  → pick model+effort (UCB1 bandit)
  → spawn Claude subagent to mutate the code
  → check novelty (difflib + embeddings + LLM judge)
  → evaluate (run your benchmark/test)
  → update population & archive
  → repeat
```

Each generation, the bandit selects a (model, effort) configuration from the pool — opus, sonnet, and haiku at various thinking effort levels — creating diversity in how mutations are generated. Over time, the bandit learns which configurations produce the best improvements.

## Requirements

- **Claude Code** (`claude` CLI) with Claude Pro Max, Team, or API access
- **Python 3.10+** (stdlib only for core; optional deps below)

```bash
pip install -r requirements.txt  # optional: sentence-transformers for embedding dedup
```

## Quick Start

### 1. Prepare your task

You need two files:

**`initial.py`** (or `.cu`, `.rs`, `.cpp`, etc.) — your seed program with evolve markers:
```python
# EVOLVE-BLOCK-START
def my_algorithm():
    # This code will be evolved
    ...
# EVOLVE-BLOCK-END

def run():
    # This stays fixed — entry point for evaluation
    return my_algorithm()
```

**`evaluate.py`** — evaluator that benchmarks your program:
```python
import argparse, json, os

def main(program_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    # Load and run the program
    # ... your benchmarking logic ...

    # Write metrics (combined_score drives selection)
    metrics = {
        "combined_score": 42.0,        # scalar for ranking
        "public": {                     # shown to mutation agent
            "gflops": 412.3,
            "bandwidth_util": 0.78,
            "latency_ms": 2.4,
        },
        "private": {},                  # tracked but hidden
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Write correctness
    correct = True  # did it produce valid output?
    error = None     # error message if incorrect
    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": error}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
```

### 2. Initialize

```bash
# Single seed program
python claude-evolve/evolve.py init \
  --initial initial.py \
  --eval-cmd "python evaluate.py --program_path {program_path} --results_dir {results_dir}"

# Multiple seeds (different starting approaches)
python claude-evolve/evolve.py init \
  --initial naive.py tiled.py vectorized.py \
  --eval-cmd "python evaluate.py --program_path {program_path} --results_dir {results_dir}" \
  --num-islands 3
```

### 3. Run evolution

**Autonomous loop** (recommended):
```bash
# Run 50 generations, pause every 10 for confirmation
python claude-evolve/run_loop.py --generations 50 --pause-every 10

# Fully autonomous
python claude-evolve/run_loop.py --generations 100 --no-pause

# With longer timeout for complex programs (opus/max can take 10-15 min)
python claude-evolve/run_loop.py -n 50 --no-pause --mutation-timeout 1200
```

**Manual step-by-step** (or via Claude Code skill):
```bash
python claude-evolve/evolve.py select          # pick parent + model
# ... write mutation to the candidate path shown ...
python claude-evolve/evolve.py check-novelty --candidate <path> --parent-id <id>
python claude-evolve/evolve.py evaluate --candidate <path>
python claude-evolve/evolve.py update --candidate-dir <dir> --parent-id <id> \
  --mutation-type full_rewrite --description "what changed" --model-config "sonnet/high"
```

### 4. Inspect results

```bash
python claude-evolve/evolve.py status
python claude-evolve/evolve.py leaderboard --top 10
```

## All Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize from seed program(s) |
| `select` | Select parent, model, and candidate path for next mutation |
| `select --retry` | Retry with same parent/model after failure |
| `check-novelty` | Check if candidate is novel vs archive (difflib + embeddings + LLM judge) |
| `evaluate` | Run the evaluator on a candidate |
| `update` | Add evaluated candidate to population, update archive and bandit |
| `status` | Show generation, archive, model ensemble stats |
| `leaderboard` | Show top programs ranked by score |
| `meta-update` | Generate or save meta-recommendations |
| `meta-show` | Show current meta-recommendations |
| `prompt-evolve` | Evolve the system prompt |
| `prompt-show` | Show prompt archive |

## Features

### Multi-Model Ensemble

The bandit pool includes 8 configurations:

| Label | Model | Effort | Character |
|-------|-------|--------|-----------|
| `opus/max` | opus | max | Deepest reasoning, slowest |
| `opus/high` | opus | high | Deep analysis |
| `opus/medium` | opus | medium | Balanced opus |
| `sonnet/high` | sonnet | high | Strong analysis, faster |
| `sonnet/medium` | sonnet | medium | Workhorse |
| `sonnet/low` | sonnet | low | Quick intuitive changes |
| `haiku/medium` | haiku | medium | Fast exploration |
| `haiku/low` | haiku | low | Fastest, most diverse |

Selection uses **Asymmetric UCB1** (ported from ShinkaEvolve):
- **Warm-up**: each arm gets K=4 pulls before exploitation begins
- **Asymmetric reward**: only positive improvements count (bad results don't kill an arm)
- **Epsilon-exploration**: 20% random selection to prevent premature convergence

### Novelty Detection

Three-stage check (ported from ShinkaEvolve's `NoveltyJudge`):
1. **difflib** — fast text similarity on the EVOLVE-BLOCK (threshold: 0.95)
2. **Embedding cosine similarity** — semantic similarity via `intfloat/multilingual-e5-large` (threshold: 0.97)
3. **LLM judge** — if either is above threshold, a haiku subagent judges "NOVEL" or "NOT_NOVEL"

If not novel, the candidate is rejected and the retry loop tries again with the same model.

### Retry Loop

Up to 3 attempts per generation (ported from ShinkaEvolve's 3-layer retry):
- Same model and parent throughout (fair bandit trial)
- Error messages from previous attempts are passed as context
- `select --retry --retry-error "circles overlapped"` advances to the next attempt

### Islands

Multi-population diversity preservation:
```bash
python claude-evolve/evolve.py init --initial a.py b.py c.py --num-islands 3
```
- Programs inherit their parent's island
- Parent selection is scoped to the target island
- **Migration** every N generations moves top programs between islands (elitist — best per island is protected)

### Meta-Recommendations

Periodic analysis of what's working:
- Every N generations (default 5), the system flags that meta-recs are due
- A haiku subagent analyzes the archive and recent programs
- Generates numbered recommendations (e.g., "try simulated annealing", "focus on memory coalescing")
- Recommendations are injected into the `select` output to guide future mutations

```bash
python claude-evolve/evolve.py meta-update        # generate context for subagent
python claude-evolve/evolve.py meta-update --set "1. ... 2. ..."  # save recommendations
python claude-evolve/evolve.py meta-show           # view current recommendations
```

### Prompt Evolution

Co-evolve the system prompt alongside the code:
```bash
python claude-evolve/evolve.py init --initial prog.py --eval-cmd "..." \
  --evolve-prompts --task-sys-msg "Optimize this GPU kernel for maximum throughput"
```
- Prompt archive with UCB selection
- Percentile-based fitness (prompts that produce high-scoring programs are preferred)
- Periodic evolution via subagent

### Multi-Metric Evaluation

The evaluator can return arbitrarily many metrics:
```json
{
  "combined_score": 412.3,
  "public": {
    "gflops": 412.3,
    "bandwidth_util": 0.78,
    "occupancy": 0.65,
    "latency_ms": 2.4
  }
}
```
- `combined_score` — single scalar driving selection and archive ranking
- `public` — detailed metrics shown to the mutation agent for informed optimization
- `private` — tracked internally but not shown to the mutator

### Cross-Conversation Persistence

All state lives on disk:
- `state.json` — population, archive, bandit stats, meta-recs, prompt archive
- `embeddings.json` — code embedding vectors
- `programs/prog_XXXX/` — each program's code, metrics, and correctness

Resume in a new conversation:
```bash
python claude-evolve/evolve.py status
python claude-evolve/evolve.py leaderboard
# Then continue with select → mutate → evaluate → update
```

## Directory Structure

```
claude-evolve/
  evolve.py           # Orchestrator (stdlib only)
  run_loop.py         # Autonomous loop driver
  SKILL.md            # Claude Code skill definition
  requirements.txt    # Optional: sentence-transformers
  README.md           # This file
  state.json          # Population state (auto-generated)
  embeddings.json     # Code embeddings (auto-generated)
  programs/
    prog_0000/
      main.py         # Program code
      metrics.json    # Evaluation metrics
      correct.json    # Correctness result
      results/        # Raw evaluator output
    prog_0001/
      ...
```

## Configuration Reference

All options are set at `init` time:

| Flag | Default | Description |
|------|---------|-------------|
| `--initial` | (required) | Seed program path(s) |
| `--eval-cmd` | (required) | Evaluation command template |
| `--language` | auto | Language (inferred from extension) |
| `--population-size` | 20 | Max population |
| `--archive-size` | 10 | Max archived programs |
| `--selection-strategy` | weighted | `weighted` or `power_law` |
| `--num-islands` | 1 | Number of islands (1=disabled) |
| `--migration-interval` | 10 | Generations between migrations |
| `--migration-rate` | 0.2 | Fraction of island to migrate |
| `--meta-interval` | 5 | Generations between meta-rec updates |
| `--meta-max-recs` | 5 | Number of recommendations |
| `--evolve-prompts` | off | Enable prompt co-evolution |
| `--task-sys-msg` | none | Initial system prompt for evolution |
| `--prompt-archive-size` | 5 | Max prompts in archive |
| `--prompt-evolution-interval` | 10 | Generations between prompt mutations |

## Comparison with ShinkaEvolve

| Feature | ShinkaEvolve | claude-evolve |
|---------|-------------|---------------|
| LLM mutation | API calls to multiple providers | Claude Code subagents (no API keys) |
| Model selection | UCB/Thompson over 4+ models | Asymmetric UCB1 over model+effort combos |
| Parent selection | Weighted/power-law/beam/sequential | Weighted/power-law |
| Archive | SQLite, fitness/crowding criteria | JSON, fitness-based |
| Novelty | Embedding + LLM judge | difflib + local embedding + LLM judge |
| Retry loop | 3 novelty x 3 resample x 1 patch | 3 attempts with error feedback |
| Islands | Multi-island with elitist migration | Same |
| Meta-recs | 3-step LLM summarizer | Single-step subagent analysis |
| Prompt evolution | UCB-selected prompt archive | Same |
| Async pipeline | Concurrent proposals + evaluations | Sequential (Claude Code limitation) |
| Cost tracking | Per-model API cost tracking | N/A (Pro Max = unlimited) |
| Evaluation | `run_shinka_eval` harness | Any script outputting metrics.json + correct.json |

## License

Same as the parent ShinkaEvolve project.
