import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModel

print("loading model...")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to("cuda")
print("model loaded on GPU")

# test encoding
texts = ["how does fastapi work?"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
print(f"embedding shape: {embeddings.shape}")
print("success!")