from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model on CPU
device = torch.device("cpu")

# Use the DARE version of Mistral
model_id = "BioMistral/BioMistral-7B-DARE"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model (this may take a while on first run)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model.to(device)

# Sample prompt for DARE reasoning
prompt = """A 45-year-old man presents with chest pain, shortness of breath, and nausea. 
His ECG shows ST elevation in leads II, III, and aVF. 
Use DARE reasoning to determine the most likely diagnosis.

### Decompose:
"""

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
print("Generating output...")
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and display result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧠 Model Response:\n")
print(response)
