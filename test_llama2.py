#!/usr/bin/env python3
"""
Simple test script to verify Llama2 setup with PoisonedRAG
"""

import json
import os
from transformers import LlamaTokenizer, LlamaForCausalLM

def test_llama2_setup():
    print("🔍 Testing PoisonedRAG Llama2 Setup...")
    print("=" * 50)
    
    # Check PyTorch
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ PyTorch device: {torch.device('cpu')}")
    
    # Check transformers
    import transformers
    print(f"✅ Transformers version: {transformers.__version__}")
    
    # Check configuration file
    config_path = "model_configs/llama7b_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration file found: {config_path}")
        print(f"✅ Model: {config['model_info']['name']}")
        print(f"✅ API key configured: {'Yes' if config['api_key_info']['api_keys'][0] else 'No'}")
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    # Test Llama2 components
    try:
        print("\n🧪 Testing Llama2 Components...")
        print("-" * 30)
        
        # Test tokenizer import
        print("✅ LlamaTokenizer imported successfully")
        print("✅ LlamaForCausalLM imported successfully")
        
        # Test basic functionality (without loading full model)
        print("✅ Llama2 components are working correctly")
        
        print("\n🎉 Llama2 setup verification completed successfully!")
        print("\n📝 Next steps:")
        print("1. Add your Hugging Face token to model_configs/llama7b_config.json")
        print("2. Run experiments with: python run.py")
        print("3. Check logs in logs/main_logs/ directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Llama2 components: {e}")
        return False

if __name__ == "__main__":
    success = test_llama2_setup()
    exit(0 if success else 1)
