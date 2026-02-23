"""
Flask API Server for Finetuned Chatbot
Run this after finetuning to serve your model via REST API
"""

# Install: pip install flask flask-cors transformers torch peft

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for static website access

# ========================================
# CONFIGURATION
# ========================================
MODEL_PATH = "/finetunethisfirst/chatbot-model"  # Path to your finetuned model
BASE_MODEL = "gpt2"  # Changed to match the finetuning script

# ========================================
# LOAD MODEL AT STARTUP
# ========================================
print("Loading model...")

# Check if model exists
import os
model_path = os.path.abspath(MODEL_PATH)
adapter_config_path = os.path.join(model_path, "adapter_config.json")

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model directory not found: {model_path}")
    print("Please run the finetuning script first: python model.py")
    exit(1)

if not os.path.exists(adapter_config_path):
    print(f"❌ ERROR: adapter_config.json not found in: {model_path}")
    print("The model may not have been saved correctly during training.")
    print("\nFiles in model directory:")
    if os.path.exists(model_path):
        for f in os.listdir(model_path):
            print(f"  - {f}")
    exit(1)

print(f"✓ Model directory found: {model_path}")

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Load the finetuned LoRA adapter
print(f"Loading adapter from: {model_path}")

# Try loading with local_files_only to avoid HuggingFace Hub validation
try:
    model = PeftModel.from_pretrained(
        base_model, 
        model_path,
        local_files_only=True,
        is_trainable=False
    )
except Exception as e:
    print(f"Error loading with local_files_only: {e}")
    print("Trying alternative method...")
    # Alternative: Load config first, then model
    from peft import PeftConfig
    config = PeftConfig.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, model_path)

model.eval()

print("✅ Model loaded successfully!")
print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ========================================
# API ENDPOINTS
# ========================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": BASE_MODEL,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    Expected JSON: {"message": "user message"}
    Returns JSON: {"response": "bot response"}
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Format prompt for instruction-following
        prompt = f"Instruction: {user_message}\nResponse:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract only the response part
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("Response:")[-1].strip()
        
        return jsonify({
            "response": response,
            "success": True
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming endpoint (for future use)
    """
    return jsonify({"message": "Streaming not implemented yet"}), 501

# ========================================
# RUN SERVER
# ========================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Chatbot API Server")
    print("="*50)
    print(f"Model: {BASE_MODEL}")
    print(f"Endpoint: http://localhost:5000/chat")
    print(f"Health Check: http://localhost:5000/health")
    print("\nPress CTRL+C to stop the server")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
