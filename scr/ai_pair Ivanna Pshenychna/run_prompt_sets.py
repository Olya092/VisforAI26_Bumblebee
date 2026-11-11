import os
import pickle
import time

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import CLIPModel, CLIPProcessor

OUT_DIR = "outputs"
DATA_DIR = "./data"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64

# {} is replaced by the class name. "a" is kept even before vowels
# ("a apple") so that only the wording we vary changes, as in the baseline.
PROMPT_SETS = {
    "baseline_raw": ("a photo of a {}", "raw"),
    "baseline": ("a photo of a {}", "clean"),
    "rw_depicting": ("an image depicting a {}", "clean"),
    "rw_picture": ("a picture of a {}", "clean"),
    "rw_the": ("a photo of the {}", "clean"),
    "rw_bare": ("{}", "clean"),
    "at_blurry": ("a blurry photo of a {}", "clean"),
    "at_lowres": ("a low resolution photo of a {}", "clean"),
    "at_bw": ("a black and white photo of a {}", "clean"),
    "at_small": ("a photo of a small {}", "clean"),
    "at_big": ("a photo of a big {}", "clean"),
    "at_outdoors": ("a photo of a {} outdoors", "clean"),
    "at_indoors": ("a photo of a {} indoors", "clean"),
}

# image modifications (applied to the 32x32 image before CLIP resizes it)
IMAGE_CONDITIONS = {
    "original": lambda img: img,
    "blur": lambda img: img.filter(ImageFilter.GaussianBlur(radius=1)),
    "downscale": lambda img: img.resize((16, 16)).resize((32, 32)),
}


FINE_NAMES = """apple aquarium_fish baby bear beaver bed bee beetle bicycle bottle bowl boy
bridge bus butterfly camel can castle caterpillar cattle chair chimpanzee clock cloud cockroach
couch crab crocodile cup dinosaur dolphin elephant flatfish forest fox girl hamster house
kangaroo keyboard lamp lawn_mower leopard lion lizard lobster man maple_tree motorcycle mountain
mouse mushroom oak_tree orange orchid otter palm_tree pear pickup_truck pine_tree plain plate
poppy porcupine possum rabbit raccoon ray road rocket rose sea seal shark shrew skunk skyscraper
snail snake spider squirrel streetcar sunflower sweet_pepper table tank telephone television
tiger tractor train trout tulip turtle wardrobe whale willow_tree wolf woman worm""".split()

COARSE_NAMES = """aquatic_mammals fish flowers food_containers fruit_and_vegetables
household_electrical_devices household_furniture insects large_carnivores
large_man-made_outdoor_things large_natural_outdoor_scenes large_omnivores_and_herbivores
medium_mammals non-insect_invertebrates people reptiles small_mammals trees vehicles_1
vehicles_2""".split()


def load_cifar100_test():
    pkl = os.path.join(DATA_DIR, "cifar-100-python", "test")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        arr = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        images = [Image.fromarray(a) for a in arr]
        return images, np.array(d["fine_labels"]), np.array(d["coarse_labels"])

    import glob
    import io
    import pandas as pd
    from huggingface_hub import snapshot_download
    print("Downloading CIFAR-100 test split from Hugging Face...")
    root = snapshot_download("uoft-cs/cifar100", repo_type="dataset",
                             allow_patterns=["*test*"], local_dir=os.path.join(DATA_DIR, "hf"))
    files = sorted(glob.glob(os.path.join(root, "**", "*test*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    images = [Image.open(io.BytesIO(x["bytes"])).convert("RGB") for x in df["img"]]
    return images, df["fine_label"].to_numpy(), df["coarse_label"].to_numpy()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def image_embeddings(model, processor, images, device):
    feats = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE]
        pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        pooled = model.vision_model(pixel_values=pixels).pooler_output
        emb = model.visual_projection(pooled)
        feats.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu())
    return torch.cat(feats)


@torch.inference_mode()
def text_embeddings(model, processor, prompts, device):
    tok = processor(text=prompts, return_tensors="pt", padding=True)
    tok = {k: v.to(device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
    pooled = model.text_model(**tok).pooler_output
    emb = model.text_projection(pooled)
    return torch.nn.functional.normalize(emb, dim=-1).float().cpu()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    images, labels, coarse_labels = load_cifar100_test()
    raw_names = FINE_NAMES
    coarse_names = COARSE_NAMES
    clean_names = [n.replace("_", " ") for n in raw_names]
    # same image order as the shared baseline_results.csv?
    first = [raw_names[l] for l in labels[:8]]
    assert first == ["mountain", "forest", "seal", "mushroom", "sea", "tulip", "camel",
                     "butterfly"], f"image order differs from baseline: {first}"
    print(f"Loaded {len(images)} test images (order matches baseline_results.csv)")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    scale = model.logit_scale.exp().item()

    text = {}
    for key, (template, naming) in PROMPT_SETS.items():
        names = raw_names if naming == "raw" else clean_names
        text[key] = text_embeddings(model, processor, [template.format(n) for n in names], device)

    set_keys = list(PROMPT_SETS)
    cond_keys = list(IMAGE_CONDITIONS)
    logits = np.zeros((len(cond_keys), len(set_keys), len(images), 100), dtype=np.float16)

    for ci, cond in enumerate(cond_keys):
        t0 = time.time()
        imgs = [IMAGE_CONDITIONS[cond](im) for im in images]
        img_emb = image_embeddings(model, processor, imgs, device)
        for si, key in enumerate(set_keys):
            logits[ci, si] = (scale * img_emb @ text[key].T).numpy().astype(np.float16)
        acc = (logits[ci, set_keys.index("baseline")].argmax(1) == labels).mean()
        print(f"{cond:10s} done in {time.time() - t0:.0f}s, baseline accuracy {acc:.3f}")

    np.savez_compressed(
        os.path.join(OUT_DIR, "clip_logits.npz"),
        logits=logits, labels=labels, coarse_labels=coarse_labels,
        class_names=np.array(raw_names), coarse_names=np.array(coarse_names),
        prompt_sets=np.array(set_keys),
        templates=np.array([PROMPT_SETS[k][0] for k in set_keys]),
        conditions=np.array(cond_keys),
    )
    np.save(os.path.join(OUT_DIR, "cifar100_test_images.npy"),
            np.stack([np.asarray(im) for im in images]).astype(np.uint8))

    import csv
    with open(os.path.join("scr", "ai_pair Ivanna Pshenychna", "predictions_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "actual_class", "superclass"]
                   + [f"pred_{k}" for k in set_keys] + [f"ptrue_{k}" for k in set_keys])
        lg = logits[0].astype(np.float32)
        probs = np.exp(lg - lg.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        preds = lg.argmax(-1)
        for i in range(len(images)):
            w.writerow([i, raw_names[labels[i]], coarse_names[coarse_labels[i]]]
                       + [raw_names[preds[s, i]] for s in range(len(set_keys))]
                       + [f"{probs[s, i, labels[i]]:.4f}" for s in range(len(set_keys))])

    print("\nAccuracy on original images:")
    for si, key in enumerate(set_keys):
        print(f"  {key:14s} {(logits[0, si].argmax(1) == labels).mean():.3f}")
    print(f"\nSaved to {OUT_DIR}/ and scr/ai_pair Ivanna Pshenychna/predictions_summary.csv")


if __name__ == "__main__":
    main()
