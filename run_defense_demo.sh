#!/bin/bash
# Run with and without defense using PoisonedRAG_env. Execute in your terminal:
#   cd /path/to/PoisonedRAG && ./run_defense_demo.sh
# Or: source PoisonedRAG_env/bin/activate && python -u main.py ...
set -e
cd "$(dirname "$0")"
VENV_PYTHON="./PoisonedRAG_env/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "PoisonedRAG_env not found. Create it or run from the PoisonedRAG repo root."
  exit 1
fi
NAME="nq-contriever-gpt3.5-Top5--M2x1-adv-LM_targeted-dot-5-5-defense-ragdefender"
echo "Running with defense=ragdefender (outputs both no-defense and with-defense results)..."
"$VENV_PYTHON" -u main.py \
  --eval_model_code contriever \
  --eval_dataset nq \
  --split test \
  --query_results_dir main \
  --model_name gpt3.5 \
  --top_k 5 \
  --use_truth False \
  --gpu_id 0 \
  --defense ragdefender \
  --attack_method LM_targeted \
  --adv_per_query 5 \
  --score_function dot \
  --repeat_times 1 \
  --M 2 \
  --seed 12 \
  --name "$NAME"
echo ""
echo "Results: results/query_results/main/${NAME}.json"
