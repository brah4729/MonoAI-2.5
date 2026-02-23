"""
Chatbot Finetuning Script
This script finetunes a GPT-2 model on instruction data for chatbot purposes
"""

# Install required packages first:
# pip install transformers datasets torch accelerate peft bitsandbytes

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

# ========================================
# CONFIGURATION
# ========================================
MODEL_NAME = "gpt2"  # You can use: "gpt2", "gpt2-medium", "distilgpt2"
OUTPUT_DIR = "./chatbot-model"
DATASET_NAME = "tatsu-lab/alpaca"  # Using Alpaca dataset (52k instructions)

# Training parameters
MAX_LENGTH = 256
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# ========================================
# STEP 1: LOAD AND PREPARE DATASET
# ========================================
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train[:5000]")  # Using 5k samples for speed

# Format the dataset for instruction following
def format_instruction(example):
    """Format data as: Instruction: X\nResponse: Y"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
    else:
        prompt = f"Instruction: {instruction}\nResponse: {output}"
    
    return {"text": prompt}

print("Formatting dataset...")
dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

# Split into train/validation
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

# ========================================
# STEP 2: LOAD MODEL AND TOKENIZER
# ========================================
print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# ========================================
# STEP 3: CONFIGURE LORA (Efficient Finetuning)
# ========================================
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["c_attn"],  # For GPT-2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========================================
# STEP 4: TOKENIZE DATASET
# ========================================
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

print("\nTokenizing dataset...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

# ========================================
# STEP 5: SETUP TRAINING
# ========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,  # Changed to match eval_steps
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to="none",  # Disable wandb/tensorboard
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# ========================================
# STEP 6: TRAIN THE MODEL
# ========================================
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

trainer.train()

# ========================================
# STEP 7: SAVE THE MODEL
# ========================================
print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Model saved to: {OUTPUT_DIR}")
print("\nTo use this model, run the API server script next!")

# ========================================
# STEP 8: TEST THE MODEL
# ========================================
print("\nTesting model with sample prompt...")
model.eval()

test_prompt = "Instruction: What is machine learning?\nResponse:"
inputs = tokenizer(test_prompt, return_tensors="pt")

if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nPrompt: {test_prompt}")
print(f"Response: {response}")