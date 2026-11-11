import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_PATH = "./data"
OUT_FILE = "cifar100_test_embeddings.npy"
LABEL_FILE = "cifar100_test_labels.npy"
BATCH_SIZE = 128

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Used device: {device}")

    dataset = CIFAR100(root=DATA_PATH, train=False, download=True)
    print(f"Downloaded {len(dataset)} test images")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    def collate(batch):
        images, labels = zip(*batch)
        return processor(images=list(images), return_tensors="pt")["pixel_values"], torch.tensor(labels)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)

    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for i, (pixels, labels) in enumerate(loader):
            feats = model.get_image_features(pixel_values=pixels.to(device)).pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu())
            all_labels.append(labels)
            if (i + 1) % 10 == 0:
                print(f"  batch {i + 1}/{len(loader)}")

    embeddings = torch.cat(all_embeddings).numpy().astype(np.float32)
    labels = torch.cat(all_labels).numpy()

    np.save(os.path.join(DATA_PATH, OUT_FILE), embeddings)
    print(f"Saved embeddings {embeddings.shape}")
    np.save(os.path.join(DATA_PATH, LABEL_FILE), labels)
    print(f"Saved labels {labels.shape}")
