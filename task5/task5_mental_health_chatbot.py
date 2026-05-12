"""
Task 5: Mental Health Support Chatbot (Fine-tuned Deep Learning)

Overview:
This script builds a fine-tuned conversational chatbot designed to provide 
supportive, empathetic responses for stress, anxiety, and emotional wellness.

What It Includes:
- Training on the Amod/mental_health_counseling_conversations dataset
- Fine-tuning DistilGPT2 on empathetic counseling-style exchanges
- Simple demo for generating chatbot responses
- Training visuals and model outputs for review

Technologies:
Python, Hugging Face Transformers, Hugging Face Datasets, PyTorch
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
print('Libraries loaded.')

# ============================================================================
# 1. LOAD MENTAL HEALTH COUNSELING DATASET
# ============================================================================

print('\n--- Loading Dataset ---')
print('Loading mental health counseling conversations dataset...')
raw_dataset = load_dataset('Amod/mental_health_counseling_conversations')
print('Dataset loaded:', raw_dataset)

df = raw_dataset['train'].to_pandas()
print(f'\nDataset Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print('\nFirst 3 samples:')
print(df.head(3))

# Identify column names
context_col = 'Context' if 'Context' in df.columns else df.columns[0]
response_col = 'Response' if 'Response' in df.columns else df.columns[-1]

# Analyze text lengths
df['response_len'] = df[response_col].str.split().str.len()
df['context_len'] = df[context_col].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df['context_len'].clip(upper=150), bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Patient Context Length')
axes[0].set_xlabel('Words')
axes[1].hist(df['response_len'].clip(upper=300), bins=30, color='#e74c3c', edgecolor='white')
axes[1].set_title('Counselor Response Length')
axes[1].set_xlabel('Words')
plt.tight_layout()
plt.savefig('task5_data_distribution.png', dpi=100)
plt.show()
print(f'\nTotal conversation pairs: {len(df)}')

# ============================================================================
# 2. PREPARE DATA FOR FINE-TUNING
# ============================================================================

print('\n--- Preparing Data ---')

SUBSET_SIZE = 300  # Use 300 samples for faster demo
MAX_LEN = 64       # Sequence length for fine-tuning

def format_sample(example):
    """Format example into text for language modeling."""
    context = str(example[context_col]).strip()[:200]   # truncate long contexts
    response = str(example[response_col]).strip()[:200]
    return {'text': f"User: {context}\nCounselor: {response}<|endoftext|>"}

subset = raw_dataset['train'].select(range(SUBSET_SIZE))
formatted = subset.map(format_sample, remove_columns=subset.column_names)

print(f'Subset size: {len(formatted)}')
print('\nSample formatted text:')
print(formatted[0]['text'][:200])

# ============================================================================
# 3. LOAD DISTILGPT2 & TOKENIZER
# ============================================================================

print('\n--- Loading Model ---')

MODEL_NAME = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

print(f'Model: {MODEL_NAME}')
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

# Tokenize dataset
def tokenize(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LEN,
        padding='max_length'
    )

print('\nTokenizing dataset...')
tokenized = formatted.map(tokenize, batched=True, remove_columns=['text'])
tokenized.set_format('torch')
print(f'Tokenized samples: {len(tokenized)}')

# ============================================================================
# 4. FINE-TUNE WITH HUGGING FACE TRAINER
# ============================================================================

print('\n--- Fine-Tuning ---')

split = tokenized.train_test_split(test_size=0.1, seed=42)

training_args = TrainingArguments(
    output_dir='./task5_model',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',
    save_strategy='no',
    logging_steps=10,
    warmup_steps=20,
    weight_decay=0.01,
    fp16=False,
    report_to='none',
    use_cpu=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split['train'],
    eval_dataset=split['test'],
    data_collator=data_collator,
)

print(f'Starting fine-tuning ({SUBSET_SIZE} samples, 1 epoch, CPU)...')
train_result = trainer.train()
print(f'Done! Training loss: {train_result.training_loss:.4f}')

# ============================================================================
# 5. PLOT TRAINING CURVES
# ============================================================================

print('\n--- Training Curves ---')

logs = trainer.state.log_history
train_logs = [x for x in logs if 'loss' in x and 'eval_loss' not in x]
eval_logs = [x for x in logs if 'eval_loss' in x]

plt.figure(figsize=(9, 4))
if train_logs:
    plt.plot([x['step'] for x in train_logs],
             [x['loss'] for x in train_logs], label='Train Loss', color='steelblue')
if eval_logs:
    plt.plot([x['step'] for x in eval_logs],
             [x['eval_loss'] for x in eval_logs],
             label='Eval Loss', color='#e74c3c', marker='o', ms=8)
plt.title('Fine-Tuning Loss — DistilGPT2 on Mental Health Data')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('task5_loss_curve.png', dpi=100)
plt.show()

# ============================================================================
# 6. GENERATE EMPATHETIC RESPONSES
# ============================================================================

print('\n--- Response Generation ---')

def generate_response(user_message: str, max_new_tokens: int = 60) -> str:
    """Generate an empathetic counselor response."""
    prompt = f"User: {user_message}\nCounselor:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=48)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if 'Counselor:' in full_text:
        return full_text.split('Counselor:')[-1].strip()
    return full_text[len(prompt):].strip()

print('Response generator ready.')

# Test the chatbot
test_messages = [
    "I've been feeling really anxious about my exams lately.",
    "I feel so lonely and nobody understands me.",
    "I had a really stressful day at work today.",
    "I'm struggling to sleep because I keep worrying."
]

print('\n' + '='*70)
print('MENTAL HEALTH SUPPORT CHATBOT DEMO')
print('='*70)

for msg in test_messages:
    print(f'\nUser:      {msg}')
    response = generate_response(msg)
    print(f'Counselor: {response}')

# ============================================================================
# 7. SAVE MODEL
# ============================================================================

print('\n--- Saving Model ---')
model.save_pretrained('./task5_model/final')
tokenizer.save_pretrained('./task5_model/final')
print('Model saved to ./task5_model/final')

# ============================================================================
# INTERACTIVE CHAT MODE
# ============================================================================

print('\n' + '='*70)
print('INTERACTIVE CHAT MODE')
print('='*70)
print('Chat with the mental health support bot. Type "exit" to quit.\n')

try:
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("\nCounselorBot: Take care of yourself. Remember, seeking professional help is a sign of strength.")
            break
        if user_input:
            response = generate_response(user_input)
            print(f"CounselorBot: {response}\n")
except KeyboardInterrupt:
    print("\n\nChat ended.")

# ============================================================================
# SUMMARY
# ============================================================================

print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print("""
Model Details:
  - Base Model: DistilGPT2 (82M parameters)
  - Dataset: Amod/mental_health_counseling_conversations
  - Training: 1 epoch on 300 samples (demo run)
  - Fine-tuning approach: Causal language modeling on counselor-patient pairs

Dataset Coverage:
  - Topics: Anxiety, depression, stress, loneliness, relationships
  - Format: Real counselor-patient conversations
  - Data split: 270 train, 30 validation

Training Configuration:
  - Batch size: 16
  - Learning rate: 5e-5 (default)
  - Warmup steps: 20
  - Max length: 64 tokens
  - Device: CPU-friendly

Key Features:
  1. Empathetic response generation
  2. Temperature-based sampling for variety
  3. Repetition penalty to avoid loops
  4. Top-p (nucleus) sampling for quality

Limitations:
  - Small dataset (300 samples) produces proof-of-concept results
  - 1 epoch may lead to overfitting on demo data
  - For production: train on full dataset with 3+ epochs
  - Consider larger base models (GPT-Neo 1.3B) for better quality

Safety Note:
  - This chatbot is for emotional support demonstration ONLY
  - Always encourage users to seek professional mental health support
  - Not a replacement for licensed therapists or counselors
""")
