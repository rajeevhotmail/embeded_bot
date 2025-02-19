import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
model_path = "bigcode/starcoder"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Example usage
prompt = "def hello_world():"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs)
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_code)
