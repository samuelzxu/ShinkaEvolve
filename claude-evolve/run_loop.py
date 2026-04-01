#!/usr/bin/env python3
"""
Autonomous evolution loop driver for claude-evolve.

Runs the full select → mutate → check-novelty → evaluate → update loop
by shelling out to `claude` CLI for mutations and `evolve.py` for everything else.

Usage:
    python claude-evolve/run_loop.py --generations 50
    python claude-evolve/run_loop.py --generations 50 --pause-every 10
    python claude-evolve/run_loop.py --generations 100 --no-pause
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

EVOLVE = Path(__file__).resolve().parent / "evolve.py"


def run_evolve(*args: str, timeout: int = 300) -> str:
    """Run evolve.py with given args, return stdout."""
    cmd = [sys.executable, str(EVOLVE)] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"[ERROR] evolve.py {' '.join(args)} failed:")
        print(result.stderr[-500:] if result.stderr else "(no stderr)")
        return ""
    return result.stdout


def parse_select_output(output: str) -> dict:
    """Parse the structured output from evolve.py select."""
    info = {}

    # Check if this is a retry output
    is_retry = "RETRY ATTEMPT" in output

    if is_retry:
        # Retry format: "Same parent: prog_XXXX (score: ...)"
        m = re.search(r"Same parent:\s+(prog_\d+)", output)
        if m:
            info["parent_id"] = m.group(1)
        m = re.search(r"Parent path:\s+(.+)", output)
        if m:
            info["parent_path"] = m.group(1).strip()
        m = re.search(r"Same model:\s+(\S+)", output)
        if m:
            info["model_label"] = m.group(1)
            parts = m.group(1).split("/")
            if len(parts) == 2:
                info["model"] = parts[0]
                info["effort"] = parts[1]
    else:
        # Normal select format
        # Parent ID
        m = re.search(r"PARENT PROGRAM.*?ID:\s+(prog_\d+)", output, re.DOTALL)
        if m:
            info["parent_id"] = m.group(1)

        # Parent path
        m = re.search(r"Path:\s+(.+\.(?:py|cu|rs|cpp|jl))", output)
        if m:
            info["parent_path"] = m.group(1).strip()

        # Model config
        m = re.search(r"Label:\s+(\S+)", output)
        if m:
            info["model_label"] = m.group(1)

        m = re.search(r"(?:^|\n)\s+Model:\s+(\S+)", output)
        if m:
            info["model"] = m.group(1)

        m = re.search(r"Effort:\s+(\S+)", output)
        if m:
            info["effort"] = m.group(1)

    # Candidate path
    m = re.search(r"Write to:\s+(.+)", output)
    if m:
        info["candidate_path"] = m.group(1).strip()

    # Candidate dir
    m = re.search(r"Directory:\s+(.+)", output)
    if m:
        info["candidate_dir"] = m.group(1).strip()

    # Candidate ID
    m = re.search(r"NEXT CANDIDATE.*?ID:\s+(prog_\d+)", output, re.DOTALL)
    if m:
        info["candidate_id"] = m.group(1)

    # Meta-recommendations
    meta_match = re.search(
        r"META-RECOMMENDATIONS \(guide your mutation\)\n=+\n(.*?)(?:\n=|\Z)",
        output, re.DOTALL
    )
    if meta_match:
        info["meta_recs"] = meta_match.group(1).strip()

    # Prompt text
    prompt_match = re.search(r"Text:\s+(.+?)(?:\n\s+\(Pass|$)", output, re.DOTALL)
    if prompt_match:
        info["prompt_text"] = prompt_match.group(1).strip()

    # Prompt ID
    prompt_id_match = re.search(r"--prompt-id\s+(\S+)", output)
    if prompt_id_match:
        info["prompt_id"] = prompt_id_match.group(1)

    return info


def build_mutation_prompt(info: dict) -> str:
    """Build the prompt for the claude CLI mutation call."""
    parent_path = info.get("parent_path", "")
    if not parent_path and info.get("parent_id"):
        # Reconstruct from parent_id
        from evolve import get_program_dir, load_state
        state = load_state()
        ext = state["config"].get("extension", "py")
        parent_path = str(get_program_dir(info["parent_id"]) / f"main.{ext}")

    parts = [
        f"You are a code mutation operator in an evolutionary optimization loop.",
        f"Your thinking effort level is: {info.get('effort', 'medium')}",
        "",
        f"Read the parent program at: {parent_path}",
        "",
        "Write an improved version. Vary your strategy:",
        "- Targeted edit: change a specific section",
        "- Full rewrite: fundamentally different algorithmic approach",
        "- Crossover: combine ideas from parent and inspiration",
        "",
        "Rules:",
        "- Preserve EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers",
        "- Only modify code within these markers",
        "- Keep the same function signatures and I/O contract",
        "- Multiply final values by 0.999 safety margin for floating point",
        f"- Write the result to: {info['candidate_path']}",
    ]

    if info.get("meta_recs"):
        parts.extend(["", "Recommendations from prior analysis:", info["meta_recs"]])

    if info.get("prompt_text"):
        parts.extend(["", "Task context:", info["prompt_text"]])

    return "\n".join(parts)


def run_claude_mutation(info: dict, timeout: int = 600) -> bool:
    """Call claude CLI to generate a mutation. Returns True if file was written."""
    model = info.get("model", "sonnet")
    prompt = build_mutation_prompt(info)
    candidate_path = Path(info["candidate_path"])

    # Ensure directory exists
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "claude",
        "--model", model,
        "-p", prompt,
        "--output-format", "text",
        "--max-turns", "10",
        "--allowedTools", "Read,Write,Edit,Bash,Glob,Grep",
    ]

    print(f"    Spawning claude --model {model} ...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(Path.cwd()),
        )
        if result.returncode != 0:
            print(f"    [WARN] claude exited with code {result.returncode}")
            if result.stderr:
                print(f"    stderr: {result.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        print(f"    [WARN] claude timed out after {timeout}s")
        return False

    return candidate_path.exists()


def run_generation(gen_num: int, max_retries: int = 3) -> dict:
    """Run one full generation. Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"  GENERATION {gen_num}")
    print(f"{'='*60}")

    last_error = ""
    for attempt in range(1, max_retries + 1):
        # Select
        if attempt == 1:
            select_output = run_evolve("select")
        else:
            select_output = run_evolve("select", "--retry",
                                       "--retry-error", last_error or "unknown")

        if not select_output:
            print("  [ERROR] select failed")
            return {"success": False, "error": "select failed"}

        info = parse_select_output(select_output)
        if "candidate_path" not in info:
            print("  [ERROR] could not parse select output")
            return {"success": False, "error": "parse failed"}

        print(f"  Attempt {attempt}/{max_retries}: "
              f"parent={info.get('parent_id')}, model={info.get('model_label')}")

        # Mutate
        if not run_claude_mutation(info):
            print(f"    Mutation failed to write file")
            last_error = "mutation failed to produce output"
            continue

        # Novelty check
        novelty_output = run_evolve(
            "check-novelty",
            "--candidate", info["candidate_path"],
            "--parent-id", info["parent_id"],
        )
        if "NOT NOVEL" in novelty_output and "POTENTIALLY" in novelty_output:
            # Could spawn a haiku judge here, but for speed just reject
            print(f"    Not novel — retrying")
            last_error = "not novel (high similarity to archive)"
            continue

        # Evaluate
        eval_output = run_evolve("evaluate", "--candidate", info["candidate_path"],
                                 timeout=600)
        if not eval_output:
            print(f"    Evaluation failed")
            last_error = "evaluation script failed"
            continue

        # Check correctness
        correct_file = Path(info["candidate_dir"]) / "correct.json"
        if correct_file.exists():
            with open(correct_file) as f:
                correct_data = json.load(f)
                if not correct_data.get("correct", False):
                    error = correct_data.get("error", "unknown")
                    print(f"    Incorrect: {error[:100]}")
                    last_error = error
                    continue

        # Update
        update_args = [
            "update",
            "--candidate-dir", info["candidate_dir"],
            "--parent-id", info["parent_id"],
            "--mutation-type", "full_rewrite",
            "--description", f"gen{gen_num} auto-evolved ({info.get('model_label', '?')})",
            "--model-config", info.get("model_label", ""),
        ]
        if info.get("prompt_id"):
            update_args.extend(["--prompt-id", info["prompt_id"]])

        update_output = run_evolve(*update_args)

        # Parse score from update output
        score_match = re.search(r"Score:\s+([\d.]+)", update_output)
        score = float(score_match.group(1)) if score_match else 0.0

        archive_match = re.search(r"Archive:\s+(YES|no)", update_output)
        in_archive = archive_match and archive_match.group(1) == "YES"

        print(f"    Score: {score:.4f} | Archive: {'YES' if in_archive else 'no'} | "
              f"Model: {info.get('model_label', '?')}")

        # Check for triggered actions (meta-update, prompt-evolve)
        if "META-RECOMMENDATIONS DUE" in update_output:
            print("    [AUTO] Running meta-update...")
            # Quick meta-update via haiku
            meta_context = run_evolve("meta-update")
            # In fully autonomous mode, we'd spawn claude here
            # For now, just note it
            print("    [NOTE] Meta-update context generated (manual save needed)")

        if "PROMPT EVOLUTION DUE" in update_output:
            print("    [NOTE] Prompt evolution due (manual trigger needed)")

        return {
            "success": True,
            "score": score,
            "in_archive": in_archive,
            "model": info.get("model_label", "?"),
            "candidate_id": info.get("candidate_id", "?"),
            "attempts": attempt,
        }

    # All retries exhausted
    print(f"  All {max_retries} attempts failed")
    return {"success": False, "error": "max retries exhausted"}


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous evolution loop for claude-evolve"
    )
    parser.add_argument("--generations", "-n", type=int, default=10,
                        help="Number of generations to run")
    parser.add_argument("--pause-every", type=int, default=5,
                        help="Pause for confirmation every N generations (0=never)")
    parser.add_argument("--no-pause", action="store_true",
                        help="Never pause (fully autonomous)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts per generation")
    parser.add_argument("--mutation-timeout", type=int, default=1200,
                        help="Timeout for claude mutation calls (seconds). "
                             "opus/max may need 10-15 min for complex programs.")
    args = parser.parse_args()

    if args.no_pause:
        args.pause_every = 0

    # Print initial status
    print(run_evolve("status"))

    results = []
    start_time = time.time()

    for gen in range(1, args.generations + 1):
        result = run_generation(gen, max_retries=args.max_retries)
        results.append(result)

        # Pause for confirmation
        if args.pause_every > 0 and gen % args.pause_every == 0 and gen < args.generations:
            print(f"\n--- Paused after {gen}/{args.generations} generations ---")
            print(run_evolve("status"))
            try:
                response = input("Continue? [Y/n/status/leaderboard] ").strip().lower()
                if response in ("n", "no", "quit", "q"):
                    print("Stopped by user.")
                    break
                elif response == "status":
                    print(run_evolve("status"))
                elif response == "leaderboard":
                    print(run_evolve("leaderboard"))
            except (KeyboardInterrupt, EOFError):
                print("\nStopped.")
                break

    # Final summary
    elapsed = time.time() - start_time
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    print(f"\n{'='*60}")
    print(f"  RUN COMPLETE")
    print(f"{'='*60}")
    print(f"  Generations attempted: {len(results)}")
    print(f"  Successful: {len(successes)}")
    print(f"  Failed: {len(failures)}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/max(len(results),1):.0f}s/gen)")

    if successes:
        scores = [r["score"] for r in successes]
        print(f"  Best score: {max(scores):.4f}")
        print(f"  Avg score:  {sum(scores)/len(scores):.4f}")

    print()
    print(run_evolve("status"))


if __name__ == "__main__":
    main()
