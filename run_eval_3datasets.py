#!/usr/bin/env python3
"""
Run the PoisonedRAG pipeline on 3 BEIR datasets (nq, hotpotqa, msmarco)
with GPT-4 and RAGDefender enabled, 10 target queries per dataset.

Each run writes a single JSON result file that already contains both
no-defense and with-defense outputs per query. These files are later
consumed by the ASR evaluation script.

Usage (from repo root, with your venv activated):
    python run_eval_3datasets.py
"""
import os
import sys
import subprocess


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_for_dataset(dataset: str) -> int:
    eval_model = "contriever"
    model_name = "gpt4"
    top_k = 5
    M = 10
    repeat_times = 1
    query_results_dir = "main"
    seed = 12

    base_name = (
        f"{dataset}-{eval_model}-{model_name}-Top{top_k}--M{M}x{repeat_times}"
        f"-adv-LM_targeted-dot-5-{top_k}-defense-ragdefender"
    )

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--eval_model_code",
        eval_model,
        "--eval_dataset",
        dataset,
        "--split",
        "test",
        "--query_results_dir",
        query_results_dir,
        "--model_name",
        model_name,
        "--top_k",
        str(top_k),
        "--use_truth",
        "False",
        "--gpu_id",
        "0",
        "--defense",
        "ragdefender",
        "--attack_method",
        "LM_targeted",
        "--adv_per_query",
        "5",
        "--score_function",
        "dot",
        "--repeat_times",
        str(repeat_times),
        "--M",
        str(M),
        "--seed",
        str(seed),
        "--random_targets",
        "True",
        "--name",
        base_name,
    ]

    print("\n" + "=" * 80)
    print(f"Running dataset={dataset} with GPT-4 and RAGDefender")
    print("Command:", " ".join(cmd))
    print("=" * 80 + "\n")

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode == 0:
        print(
            f"Finished dataset={dataset}. "
            f"Results: results/query_results/{query_results_dir}/{base_name}.json"
        )
    else:
        print(f"Run failed for dataset={dataset} with return code {result.returncode}")
    return result.returncode


def main() -> None:
    datasets = ["nq", "hotpotqa", "msmarco"]
    for ds in datasets:
        code = run_for_dataset(ds)
        if code != 0:
            sys.exit(code)


if __name__ == "__main__":
    main()

