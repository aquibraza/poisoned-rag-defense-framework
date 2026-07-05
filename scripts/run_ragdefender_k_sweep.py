#!/usr/bin/env python3
"""Fair RAGDefender evaluation harness: sweep k (retrieved context size)
across datasets and defenses to test whether RAGDefender recovers once
k > N, i.e. once the retrieved context is no longer 100% poisoned.

Two independent, both-off-by-default cost controls apply:

1. This script's own dry-run vs --execute: by default it only PRINTS the
   main.py commands it would run -- no subprocess is spawned at all. Pass
   --execute to actually run them.
2. main.py's own --dry_run: even when this script --execute's a command,
   the generated main.py invocations pass `--dry_run True` by default, so
   NO LLM API calls are made (zero cost) -- only retrieval, defense, and
   detection-quality diagnostics run and are logged. Pass --live_generation
   (together with --execute) to additionally disable main.py's --dry_run
   and make real, billed LLM calls.

Diagnostics-only quick mode (the fastest way to test the core k=N
saturation hypothesis -- see docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md):
    python scripts/run_ragdefender_k_sweep.py --quick_hotpotqa
    python scripts/run_ragdefender_k_sweep.py --quick_hotpotqa --execute

Full k-sweep (print commands only):
    python scripts/run_ragdefender_k_sweep.py \
        --datasets nq hotpotqa msmarco --k_values 5 10 20 50 --N 5 \
        --max_queries 100 --defenses none ragdefender_original

Full k-sweep, actually run (still --dry_run True on main.py, zero API cost):
    python scripts/run_ragdefender_k_sweep.py --execute --max_queries 10

Only ever add --live_generation if you explicitly want to spend real
GPT-4/PaLM/etc. API calls:
    python scripts/run_ragdefender_k_sweep.py --quick_hotpotqa --execute --live_generation
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_DATASETS = ["nq", "hotpotqa", "msmarco"]
DEFAULT_K_VALUES = [5, 10, 20, 50]
ALL_DEFENSE_CHOICES = [
    "none",
    "ragdefender",
    "ragdefender_original",
    "oracle_remove_all_poison",
    "random_remove_same_count",
]

# "Diagnostics-only quick mode" preset (see docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md).
QUICK_HOTPOTQA_DATASET = "hotpotqa"
QUICK_HOTPOTQA_K_VALUES = [5, 10]
QUICK_HOTPOTQA_MAX_QUERIES = 10
QUICK_HOTPOTQA_N = 5
QUICK_HOTPOTQA_DEFENSES = [
    "none",
    "ragdefender_original",
    "oracle_remove_all_poison",
    "random_remove_same_count",
]


def build_run_name(dataset: str, k: int, N: int, defense: str, max_queries: int) -> str:
    return f"{dataset}-contriever-gpt4-Top{k}--M{max_queries}x1-adv-LM_targeted-dot-{N}-{k}-defense-{defense}"


def build_command(
    *,
    dataset: str,
    k: int,
    N: int,
    defense: str,
    max_queries: int,
    model_name: str,
    diagnostics_dir: str,
    query_results_dir: str,
    live_generation: bool,
    seed: int,
    random_removal_seed: int,
    model_config_path: str = None,
):
    name = build_run_name(dataset, k, N, defense, max_queries)
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--eval_model_code", "contriever",
        "--eval_dataset", dataset,
        "--split", "test",
        "--query_results_dir", query_results_dir,
        "--model_name", model_name,
        "--top_k", str(k),
        "--use_truth", "False",
        "--gpu_id", "0",
        "--defense", defense,
        "--attack_method", "LM_targeted",
        "--adv_per_query", str(N),
        "--score_function", "dot",
        "--repeat_times", "1",
        "--M", str(max_queries),
        "--limit", str(max_queries),
        "--seed", str(seed),
        "--random_removal_seed", str(random_removal_seed),
        "--log_diagnostics", "True",
        "--diagnostics_dir", diagnostics_dir,
        "--dry_run", "False" if live_generation else "True",
        "--name", name,
    ]
    # main.py already supports --model_config_path (defaults to
    # model_configs/{model_name}_config.json when omitted); this just
    # exposes an explicit override at the sweep-script level rather than
    # relying on that model_name-derived default.
    if model_config_path:
        cmd += ["--model_config_path", model_config_path]
    return cmd, name


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--k_values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--N", type=int, default=5, help="Number of injected adversarial passages per query.")
    parser.add_argument(
        "--max_queries", type=int, default=100,
        help="Number of target queries per run (mapped to main.py's --M and --limit).",
    )
    parser.add_argument(
        "--defenses", nargs="+", default=["none", "ragdefender_original"], choices=ALL_DEFENSE_CHOICES,
    )
    parser.add_argument("--model_name", default="gpt4")
    parser.add_argument(
        "--model_config_path", default=None,
        help=(
            "Explicit path to a model config JSON (e.g. model_configs/gpt4_config.json). "
            "Passed straight through to main.py's own --model_config_path. If omitted, "
            "main.py falls back to its default: model_configs/{model_name}_config.json."
        ),
    )
    parser.add_argument(
        "--query_results_dir", default="ragdefender_k_sweep",
        help="Subdirectory of results/query_results/ for this sweep's result JSON files.",
    )
    parser.add_argument("--diagnostics_dir", default="results/diagnostics/ragdefender")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--random_removal_seed", type=int, default=12)
    parser.add_argument(
        "--execute", action="store_true",
        help=(
            "Actually run the generated main.py commands via subprocess. Without this "
            "flag, commands are only printed -- no subprocess is spawned."
        ),
    )
    parser.add_argument(
        "--live_generation", action="store_true",
        help=(
            "DANGER: pass --dry_run False to main.py, making real (billed) LLM API "
            "calls. Only takes effect together with --execute. Off by default."
        ),
    )
    parser.add_argument(
        "--quick_hotpotqa", action="store_true",
        help=(
            "Diagnostics-only preset: HotpotQA only, max_queries=10, N=5, k in [5, 10], "
            "defenses=[none, ragdefender_original, oracle_remove_all_poison, "
            "random_remove_same_count]. The fastest way to test the core k=N "
            "saturation hypothesis. Overrides --datasets/--k_values/--max_queries/--N/--defenses."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick_hotpotqa:
        datasets = [QUICK_HOTPOTQA_DATASET]
        k_values = QUICK_HOTPOTQA_K_VALUES
        max_queries = QUICK_HOTPOTQA_MAX_QUERIES
        N = QUICK_HOTPOTQA_N
        defenses = QUICK_HOTPOTQA_DEFENSES
        print("Using --quick_hotpotqa preset (diagnostics-only quick mode).")
    else:
        datasets = args.datasets
        k_values = args.k_values
        max_queries = args.max_queries
        N = args.N
        defenses = args.defenses

    if args.live_generation and not args.execute:
        print("WARNING: --live_generation has no effect without --execute; ignoring.", file=sys.stderr)

    live_generation = args.execute and args.live_generation

    commands = []
    for dataset in datasets:
        for k in k_values:
            for defense in defenses:
                cmd, name = build_command(
                    dataset=dataset,
                    k=k,
                    N=N,
                    defense=defense,
                    max_queries=max_queries,
                    model_name=args.model_name,
                    diagnostics_dir=args.diagnostics_dir,
                    query_results_dir=args.query_results_dir,
                    live_generation=live_generation,
                    seed=args.seed,
                    random_removal_seed=args.random_removal_seed,
                    model_config_path=args.model_config_path,
                )
                commands.append((cmd, name))

    print(
        f"Prepared {len(commands)} run(s): {len(datasets)} dataset(s) x "
        f"{len(k_values)} k value(s) x {len(defenses)} defense(s), max_queries={max_queries}, N={N}"
    )
    if live_generation:
        print("!!! --live_generation is set: these runs WILL make real, billed LLM API calls. !!!")
    elif args.execute:
        print("Executing with main.py --dry_run True: no LLM API calls will be made (zero cost).")
    else:
        print("Script-level dry-run: printing commands only. Pass --execute to actually run them.")
    print()

    exit_code = 0
    for cmd, name in commands:
        print("-" * 80)
        print(f"[{name}]")
        print(" ".join(cmd))
        if args.execute:
            result = subprocess.run(cmd, cwd=REPO_ROOT)
            if result.returncode != 0:
                print(f"Run failed for {name} with return code {result.returncode}", file=sys.stderr)
                exit_code = result.returncode
    print("-" * 80)

    if not args.execute:
        print(
            f"\n{len(commands)} command(s) printed above. Re-run with --execute to actually "
            "run them (main.py --dry_run True by default: zero LLM API cost)."
        )
    else:
        print(f"\nDiagnostics written under: {args.diagnostics_dir}/")
        print("Next: python scripts/summarize_ragdefender_diagnostics.py")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
