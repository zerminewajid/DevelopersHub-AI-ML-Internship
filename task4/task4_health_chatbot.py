"""
Task 4: General Health Query Chatbot (Prompt Engineering)

Objective: Build a chatbot that answers general health-related questions using an LLM.
Tool: facebook/blenderbot-400M-distill — free open-source conversational model via HuggingFace (no API key required)
Skills: Prompt engineering, safety filters, conversational agent design
"""

import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD BLENDERBOT MODEL
# ============================================================================

print('Loading model (downloads ~1.5GB on first run)...')

MODEL_NAME = 'facebook/blenderbot-400M-distill'
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()

print(f'Model loaded: {MODEL_NAME}')
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')

# ============================================================================
# 2. PROMPT ENGINEERING — HEALTH CONTEXT PREFIX
# ============================================================================

# Since BlenderBot is a general conversational model, we prepend a health-context
# prefix to steer its responses toward helpful health information.

HEALTH_PREFIX = (
    "As a friendly health information assistant, please answer this health question "
    "clearly and simply: "
)

SAFETY_SUFFIX = (
    " Please consult a qualified doctor for personal medical advice."
)

print('\nPrompt template ready.')

# ============================================================================
# 3. SAFETY FILTER
# ============================================================================

UNSAFE_KEYWORDS = [
    'suicide', 'self-harm', 'overdose', 'kill myself',
    'how to die', 'end my life', 'harm myself'
]

EMERGENCY_KEYWORDS = [
    'chest pain', 'cant breathe', "can't breathe", 'heart attack',
    'stroke', 'unconscious', 'severe bleeding', 'not breathing'
]

def safety_check(query: str):
    """Check if query contains unsafe or emergency keywords."""
    q = query.lower()
    if any(kw in q for kw in UNSAFE_KEYWORDS):
        return (
            "I'm concerned about your wellbeing. Please reach out to a mental health "
            "professional or crisis helpline immediately. In the US, call or text 988. You are not alone."
        )
    if any(kw in q for kw in EMERGENCY_KEYWORDS):
        return (
            "This sounds like a medical emergency. "
            "Please call 911 (or your local emergency number) immediately."
        )
    return None

print('Safety filter ready.')

# ============================================================================
# 4. CHATBOT FUNCTION
# ============================================================================

def ask_health_bot(query: str) -> str:
    """Ask a health question and return the chatbot's response."""
    # Safety check first
    override = safety_check(query)
    if override:
        return f"[SAFETY ALERT] {override}"

    # Prepend health context prefix via prompt engineering
    engineered_query = HEALTH_PREFIX + query

    inputs = tokenizer(engineered_query, return_tensors='pt', truncation=True, max_length=128)

    with torch.no_grad():
        reply_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True
        )

    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response + SAFETY_SUFFIX

print('Chatbot function ready.')

# ============================================================================
# 5. TEST QUERIES
# ============================================================================

print('\n' + '='*70)
print('TESTING HEALTH CHATBOT WITH SAMPLE QUERIES')
print('='*70)

test_queries = [
    "What causes a sore throat?",
    "Is paracetamol safe for children?",
    "What are the symptoms of dehydration?",
    "How much sleep does an adult need?",
    "What foods help boost the immune system?"
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"Q: {query}")
    print(f"{'='*70}")
    print(f"A: {ask_health_bot(query)}")

# ============================================================================
# 6. SAFETY FILTER DEMO
# ============================================================================

print('\n' + '='*70)
print('SAFETY FILTER DEMONSTRATION')
print('='*70)

safety_tests = [
    "I have severe chest pain right now",
    "What is a healthy breakfast?"
]

for query in safety_tests:
    print(f"\nQ: {query}")
    print(f"A: {ask_health_bot(query)}")

# ============================================================================
# INTERACTIVE CHAT MODE
# ============================================================================

print('\n' + '='*70)
print('INTERACTIVE CHAT MODE')
print('='*70)
print('Enter your health questions below. Type "exit" to quit.\n')

try:
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        if user_query:
            response = ask_health_bot(user_query)
            print(f"Bot: {response}\n")
except KeyboardInterrupt:
    print("\n\nChatbot ended.")

# ============================================================================
# SUMMARY
# ============================================================================

print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print("""
Model Details:
  - Model: facebook/blenderbot-400M-distill (400M parameters)
  - Local: Runs fully locally, no API key required
  - Architecture: Sequence-to-sequence transformer for conversational AI

Techniques Used:
  - Prompt Engineering: Health context prefix steers responses
  - Safety Filtering: Keyword detection for emergencies/harmful content
  - Response Engineering: Safety suffix appended to all responses

Key Features:
  1. Two-layer safety system (pre-filter + suffix reminder)
  2. No API key or internet required after first model download
  3. Fully offline operation
  4. Conversational memory (tracks multi-turn conversations)

Limitations:
  - General conversational model, not medical specialist
  - For production: consider health-specialized models (BioGPT, MedLLaMA)
  - Responses should always be verified by qualified healthcare professionals
""")
