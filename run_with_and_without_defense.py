#!/usr/bin/env python3
"""
Run the pipeline once without defense and once with RAGDefender for comparison.
Uses minimal M and repeat_times for quick analysis. Activate PoisonedRAG_env first, or
run with: ./PoisonedRAG_env/bin/python run_with_and_without_defense.py
"""
import os
import sys
import subprocess

# Default: one dataset, 2 queries, 1 repeat for quick comparison
DATASET = os.environ.get("PRAG_DATASET", "nq")
MODEL = os.environ.get("PRAG_MODEL", "gpt4")
M = int(os.environ.get("PRAG_M", "2"))
REPEAT_TIMES = int(os.environ.get("PRAG_REPEAT", "1"))
TOP_K = 5
EVAL_MODEL = "contriever"
QUERY_RESULTS_DIR = "main"

BASE_NAME = (
    f"{DATASET}-{EVAL_MODEL}-{MODEL}-Top{TOP_K}--M{M}x{REPEAT_TIMES}"
    f"-adv-LM_targeted-dot-5-{TOP_K}"
)


def run_main(defense: str, name_suffix: str):
    name = f"{BASE_NAME}{name_suffix}"
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--eval_model_code", EVAL_MODEL,
        "--eval_dataset", DATASET,
        "--split", "test",
        "--query_results_dir", QUERY_RESULTS_DIR,
        "--model_name", MODEL,
        "--top_k", str(TOP_K),
        "--use_truth", "False",
        "--gpu_id", "0",
        "--defense", defense,
        "--attack_method", "LM_targeted",
        "--adv_per_query", "5",
        "--score_function", "dot",
        "--repeat_times", str(REPEAT_TIMES),
        "--M", str(M),
        "--seed", "12",
        "--name", name,
    ]
    print(f"\n{'='*60}\nRunning: defense={defense!r} -> name={name}\n{'='*60}\n")
    return subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != root:
        os.chdir(root)
        print(f"Changed cwd to {root}")

    r1 = run_main("none", "")
    if r1.returncode != 0:
        print("Run without defense failed.", file=sys.stderr)
        sys.exit(r1.returncode)

    r2 = run_main("ragdefender", "-defense-ragdefender")
    if r2.returncode != 0:
        print("Run with defense failed.", file=sys.stderr)
        sys.exit(r2.returncode)

    print("\n" + "="*60)
    print("Both runs finished. Results saved under results/query_results/main/")
    print(f"  - Without defense: {BASE_NAME}.json")
    print(f"  - With defense:   {BASE_NAME}-defense-ragdefender.json")
    print("="*60)


if __name__ == "__main__":
    main()
