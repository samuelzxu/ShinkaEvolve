---
name: claude-evolve
description: Run evolutionary code optimization using Claude Code as the mutation operator. No API keys needed — Claude Code proposes mutations, a Python orchestrator handles population state and evaluation.
---

# Claude Evolve

Evolutionary code optimization where Claude Code is the mutation operator.
No LLM API keys required — works entirely with Claude Pro Max.

## When to Use

- User wants to optimize code (GPU kernels, algorithms, etc.) through iterative mutation and selection
- User has `evaluate.py` and `initial.<ext>` ready (or wants to create them)
- No LLM API keys available — using Claude Pro Max only

## Setup (first time only)

### 1. Verify task files exist

```bash
ls evaluate.py initial.*
```

The task needs:
- `initial.<ext>` — seed program with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers around the code to evolve
- `evaluate.py` — evaluator that accepts `--program_path` and `--results_dir` args, writes `metrics.json` (must include `combined_score`) and `correct.json` (must include `correct: true/false`)

### 2. Initialize evolution

```bash
python claude-evolve/evolve.py init \
  --initial initial.py \
  --eval-cmd "python evaluate.py --program_path {program_path} --results_dir {results_dir}"
```

### 3. Verify initialization

```bash
python claude-evolve/evolve.py status
```

## Evolution Step (repeat for each generation)

### Step 1: Select parent

```bash
python claude-evolve/evolve.py select
```

This prints:
- Parent program path, score, and metrics
- 1-2 inspiration programs from the archive
- **Model config** — the bandit-selected (model, effort) to use for this mutation
- Path for the next candidate file

### Step 2: Read parent and inspirations

Read the parent program file and any inspiration files shown in the output.
Study their code and metrics carefully. Understand what makes higher-scoring
programs better.

### Step 3: Propose a mutation (using the selected model config)

The `select` output includes a **MODEL CONFIG** section with a `model` and
`effort` level chosen by a UCB1 bandit. You MUST spawn a subagent with the
specified model to generate the mutation. This creates diversity — different
models reason differently about code optimization.

Use the Agent tool to spawn the mutation:

```
Agent(
    subagent_type="general-purpose",
    model=<model from select output>,   # "opus", "sonnet", or "haiku"
    prompt="<see mutation prompt template below>"
)
```

**Mutation prompt template** (fill in the values from `select` output):

```
You are a code mutation operator in an evolutionary optimization loop.
Your thinking effort level is: {effort}

PARENT PROGRAM (score: {score}):
<paste parent code or file path>

INSPIRATION PROGRAMS:
<paste inspiration code if any>

TASK: Write an improved version of the parent program. Strategy options:
- Targeted edit: change a specific section
- Full rewrite: fundamentally different algorithmic approach
- Crossover: combine ideas from parent and inspiration

Rules:
- Preserve EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers
- Only modify code within these markers
- Keep the same function signatures and I/O contract
- Write the result to: {candidate_path}
```

The effort level guides how the model approaches the mutation:
- **low**: Quick, intuitive changes — parameter tweaks, simple restructuring
- **medium**: Balanced analysis and modification
- **high**: Deep analysis of performance bottlenecks, algorithmic improvements
- **max** (opus only): Exhaustive reasoning about optimal approaches

IMPORTANT rules for the generated code:
- Preserve `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers
- Only modify code within these markers
- Keep the same function signatures and I/O contract
- The program must remain syntactically valid

### Step 4: Check novelty

Before evaluating, check if the mutation is meaningfully different from
existing programs. This prevents the archive from filling with clones.

```bash
python claude-evolve/evolve.py check-novelty \
  --candidate claude-evolve/programs/prog_XXXX/main.<ext> \
  --parent-id <parent_id>
```

This performs a fast difflib similarity check against the parent and archive:

- **If NOVEL** (similarity <= 0.95): proceed to evaluation
- **If POTENTIALLY NOT NOVEL** (similarity > 0.95): the output includes
  context for an LLM judge. Spawn a **haiku** subagent to compare:

```
Agent(
    model="haiku",
    prompt="<judge prompt from check-novelty output>

    Read these two files and respond NOVEL or NOT_NOVEL:
    - Existing: <most_similar_program_path>
    - Proposed: <candidate_path>

    Are these meaningfully different? Consider algorithmic approach,
    data structures, and optimization strategy. Ignore variable names,
    comments, and formatting."
)
```

- **If judge says NOT_NOVEL**: reject and retry with `select --retry`
- **If judge says NOVEL**: proceed to evaluation

### Step 5: Evaluate

```bash
python claude-evolve/evolve.py evaluate \
  --candidate claude-evolve/programs/prog_XXXX/main.<ext>
```

**If evaluation fails (incorrect):** retry with the same model and parent:

```bash
python claude-evolve/evolve.py select --retry \
  --retry-error "the error message from evaluation"
```

This reuses the same parent program and model config (same bandit arm),
but gives the subagent the error message as context for a fix attempt.
The retry loop allows up to 3 attempts per generation before giving up
and moving to a fresh selection.

### Step 6: Update population

```bash
python claude-evolve/evolve.py update \
  --candidate-dir claude-evolve/programs/prog_XXXX \
  --parent-id <parent_id_from_step_1> \
  --mutation-type <targeted|full_rewrite|crossover|fix> \
  --description "Brief description of what changed" \
  --model-config "<label from select output, e.g. opus/high>"
```

The `--model-config` flag feeds reward back to the UCB1 bandit. Over time,
the bandit learns which model+effort combinations produce the best
improvements and allocates more pulls to them.

### Step 7: Review and decide

```bash
python claude-evolve/evolve.py status
```

Report the result to the user:
- Generation number
- New score vs parent score (improvement or regression)
- Whether it entered the archive
- Current best score

## Full Generation Flow (with retries)

```
select              → picks parent + model + starts retry session
  ↓
mutate (subagent)   → writes candidate code
  ↓
check-novelty       → fast similarity check
  ↓ (not novel?)    → select --retry → mutate again (same model)
  ↓ (novel)
evaluate            → runs evaluator
  ↓ (incorrect?)    → select --retry --retry-error "..." → mutate again
  ↓ (correct)
update              → adds to population, clears retry session
```

Up to 3 retries per generation. Same model throughout (fair bandit trial).

## Batch Mode

When the user requests multiple generations (e.g., "run 10 generations"):

1. Repeat the full generation flow in a loop
2. After each generation, briefly report: generation, score delta, archive status
3. **Ask the user for confirmation every 5 generations** unless they explicitly requested fully autonomous execution
4. Adjust strategy based on results:
   - If improving: keep exploring similar directions
   - If stagnating (no improvement for 3+ generations): try a radically different approach
   - If programs are failing correctness: use `fix` mutation type, focus on making it correct first

## Resuming Across Conversations

All state persists on disk. To resume in a new conversation:

```bash
python claude-evolve/evolve.py status
python claude-evolve/evolve.py leaderboard
```

Then continue the evolution loop from Step 1.

## Inspecting Results

```bash
python claude-evolve/evolve.py leaderboard --top 10
```

To read the best program:
```bash
# Find the best program ID from leaderboard, then read it
cat claude-evolve/programs/prog_XXXX/main.<ext>
```

## Key Principles

1. **Diversity matters** — do not always use the same mutation strategy. The selection algorithm naturally handles exploitation; your job is to provide diverse explorations.
2. **Learn from inspirations** — the archive contains different high-quality solutions. Cross-pollination between approaches often yields breakthroughs.
3. **Correctness first** — an incorrect program scores 0 and is excluded from the archive. If your mutations break correctness, fix that before optimizing further.
4. **Understand the metrics** — read the evaluation output carefully. The `public_metrics` often contain clues about what to optimize next.
5. **Be bold** — incremental tweaks produce incremental gains. Periodically try fundamentally different algorithms or data structures.
