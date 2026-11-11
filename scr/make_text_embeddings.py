import os
import numpy as np
import torch
from torchvision.datasets import CIFAR100
from transformers import CLIPModel, CLIPTokenizer

MODEL_NAME = "openai/clip-vit-base-patch32"
TEMPLATE_PATH = "./scr/label_templates.txt"
DATA_PATH = "./data"
OUT_FILE = "cifar100_text_embeddings.npz"
BATCH_SIZE = 128

def read_templates(path):
    with open(path, "r", encoding="utf-8") as f:
        templates = [line.rstrip("\n") for line in f if line.strip()]
    for t in templates:
        if "{}" not in t:
            raise ValueError(f"template {t} lacks a placeholder")
    return templates

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Used device: {device}")

    class_names = [c.replace("_", " ") for c in CIFAR100(root=DATA_PATH, train=False, download=True).classes]
    templates = read_templates(TEMPLATE_PATH)
    T, C = len(templates), len(class_names)
    print(f"{T} templates, {C} classes, {T*C} text embeddings")
    texts = [t.format(c) for t in templates for c in class_names]

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    all_embeddings = []
    with torch.no_grad():
        for start_id in range(0, len(texts), BATCH_SIZE):
            tokens = tokenizer(texts[start_id:start_id + BATCH_SIZE], padding=True, truncation=True, return_tensors="pt").to(device)
            feats = model.get_text_features(**tokens).pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu())
    embeddings = torch.cat(all_embeddings).numpy().astype(np.float32).reshape(T, C, -1)

    np.savez(os.path.join(DATA_PATH, OUT_FILE), embeddings=embeddings, templates=np.array(templates))
    print(f"Saved embeddings {embeddings.shape} (templates, classes, dim)")
