import os

def run(test_params):
    log_file, log_name = get_log_name(test_params)
    
    cmd = f"python3 -u main.py \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --top_k {test_params['top_k']}\
        --use_truth {test_params['use_truth']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --seed {test_params['seed']}\
        --name {log_name}"
        
    print(f"Running command: {cmd}")
    os.system(cmd)

def get_log_name(test_params):
    os.makedirs(f"logs/{test_params['query_results_dir']}_logs", exist_ok=True)
    
    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"
    
    if test_params['note'] != None:
        log_name = test_params['note']
    
    return f"logs/{test_params['query_results_dir']}_logs/{log_name}.txt", log_name

# Test with Llama2 configuration
test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",  # Start with just one dataset for testing
    'split': "test",
    'query_results_dir': 'llama2_test',

    # LLM setting - Changed to llama7b
    'model_name': 'llama7b',  # This should match the config file name
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 0,  # Will use CPU on M4 Air

    # attack
    'attack_method': 'LM_targeted',
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times': 1,  # Reduced for testing
    'M': 1,  # Reduced for testing
    'seed': 12,

    'note': None
}

print("🚀 Running PoisonedRAG test with Llama2...")
print(f"Model: {test_params['model_name']}")
print(f"Dataset: {test_params['eval_dataset']}")
print(f"Attack method: {test_params['attack_method']}")
print("=" * 50)

run(test_params)
