"""
Diagnostic script to check model files and test loading
"""
import os
import json

# Configuration - UPDATE THIS to match your api.py
MODEL_PATH = "/home/dhiren/Desktop/vscode/MonoAI-2.5/finetunethisfirst/chatbot-model"  # Change this to your actual path
BASE_MODEL = "gpt2"

print("="*60)
print("MODEL DIAGNOSTIC TOOL")
print("="*60)

# Step 1: Check if directory exists
model_path = os.path.abspath(MODEL_PATH)
print(f"\n1. Checking model path: {model_path}")

if not os.path.exists(model_path):
    print("   ❌ Directory does NOT exist!")
    print(f"   Please update MODEL_PATH in this script to the correct location")
    exit(1)
else:
    print("   ✓ Directory exists")

# Step 2: List all files
print(f"\n2. Files in model directory:")
files = os.listdir(model_path)
for f in sorted(files):
    file_path = os.path.join(model_path, f)
    if os.path.isdir(file_path):
        print(f"   📁 {f}/")
    else:
        size = os.path.getsize(file_path)
        print(f"   📄 {f} ({size:,} bytes)")

# Step 3: Check required files
print(f"\n3. Checking required files:")
required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",
]

all_present = True
for req_file in required_files:
    file_path = os.path.join(model_path, req_file)
    if os.path.exists(file_path):
        print(f"   ✓ {req_file}")
    else:
        print(f"   ❌ {req_file} - MISSING!")
        all_present = False

if not all_present:
    print("\n⚠️  Some required files are missing!")
    print("   The training may not have completed successfully.")
    exit(1)

# Step 4: Read adapter config
print(f"\n4. Reading adapter_config.json:")
try:
    with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
        config = json.load(f)
    print(f"   ✓ PEFT type: {config.get('peft_type', 'unknown')}")
    print(f"   ✓ Base model: {config.get('base_model_name_or_path', 'unknown')}")
    print(f"   ✓ Task type: {config.get('task_type', 'unknown')}")
except Exception as e:
    print(f"   ❌ Error reading config: {e}")
    exit(1)

# Step 5: Try loading the model
print(f"\n5. Attempting to load model...")
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig
    
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("   ✓ Tokenizer loaded")
    
    print("   Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("   ✓ Base model loaded")
    
    print("   Loading PEFT adapter...")
    # Method 1: Direct load with config
    peft_config = PeftConfig.from_pretrained(model_path)
    model = PeftModel(base_model, peft_config)
    
    # Load adapter weights
    from safetensors.torch import load_file
    adapter_weights = load_file(os.path.join(model_path, "adapter_model.safetensors"))
    model.load_state_dict(adapter_weights, strict=False)
    
    print("   ✓ PEFT adapter loaded successfully!")
    
    # Step 6: Test generation
    print(f"\n6. Testing model generation...")
    model.eval()
    test_text = "Instruction: What is AI?\nResponse:"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Test prompt: {test_text}")
    print(f"   Model response: {response}")
    
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED!")
    print("="*60)
    print(f"\nYour model is working correctly!")
    print(f"Model path to use in api.py: {model_path}")
    
except ImportError as e:
    print(f"   ❌ Missing package: {e}")
    print("   Run: pip install torch transformers peft safetensors")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    print(f"\n   Full error details:")
    import traceback
    traceback.print_exc()
    exit(1)