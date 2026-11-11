import csv

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from torchvision.datasets import CIFAR100


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = CIFAR100(root="./data", train=False, download=True)
class_names = dataset.classes

templates = [
    "a photo of the {}.",
    "a low-resolution photo of the {}.",
    "a close-up photo of the {}.",
    "a cropped photo of the {}.",
]
prompts = [
    template.format(class_name.replace("_", " "))
    for class_name in class_names
    for template in templates
]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

text_inputs = processor(
    text=prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
).to(device)

with torch.inference_mode():
    text_features = model.get_text_features(**text_inputs).pooler_output
    text_features = F.normalize(text_features, dim=-1)
    text_features = text_features.reshape(
        len(class_names),
        len(templates),
        -1,
    ).mean(dim=1)
    text_features = F.normalize(text_features, dim=-1)

predictions = []
actual_labels = []
similarity_scores = []

batch_size = 32
with torch.inference_mode():
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[index] for index in range(start, end)]
        images = [image for image, _ in batch]
        labels = [label for _, label in batch]

        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs).pooler_output
        image_features = F.normalize(image_features, dim=-1)
        scores = image_features @ text_features.T

        predictions.extend(scores.argmax(dim=1).cpu().tolist())
        actual_labels.extend(labels)
        similarity_scores.extend(scores.cpu().tolist())

correct = sum(
    actual == predicted
    for actual, predicted in zip(actual_labels, predictions)
)
accuracy = correct / len(actual_labels) * 100
print(f"\nPrompt-ensemble accuracy: {accuracy:.2f}%")

with open("prompt_ensemble_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "image_id",
        "actual_class",
        "predicted_class",
        *[f"score_{name}" for name in class_names],
    ])

    for image_id, (actual, predicted, scores) in enumerate(
        zip(actual_labels, predictions, similarity_scores)
    ):
        writer.writerow([
            image_id,
            class_names[actual],
            class_names[predicted],
            *scores,
        ])

print("Results saved to prompt_ensemble_results.csv")
