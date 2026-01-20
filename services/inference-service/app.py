import json
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms, models


app = FastAPI(title="NEU Defect Inference Service", version="1.0")

# ---- Config ----
IMG_SIZE = 224
CONF_THRESHOLD = 0.60  # below this -> "Uncertain"

# Paths (service folder -> project root -> training outputs)
SERVICE_DIR = Path(__file__).resolve().parent


MODEL_PATH = SERVICE_DIR / "models" / "model.pth"
LABELS_PATH = SERVICE_DIR / "models" / "labels.json"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform must match training/evaluate
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = None
idx_to_class = None


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return {int(k): v for k, v in labels["idx_to_class"].items()}


def build_model(num_classes: int):
    weights = models.ResNet18_Weights.DEFAULT
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@app.on_event("startup")
def startup_event():
    global model, idx_to_class

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels not found at: {LABELS_PATH}")

    idx_to_class = load_labels()
    num_classes = len(idx_to_class)

    model = build_model(num_classes)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    print(f"✅ Loaded model from {MODEL_PATH}")
    print(f"✅ Loaded labels from {LABELS_PATH}")
    print(f"Device: {device}")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    img = Image.open(file.file).convert("RGB")  # ensure consistent reading
    x = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idxs = torch.topk(probs, k=min(3, probs.shape[0]))
    top_probs = top_probs.cpu().tolist()
    top_idxs = top_idxs.cpu().tolist()

    top3 = [
        {"label": idx_to_class[i], "prob": float(p)}
        for i, p in zip(top_idxs, top_probs)
    ]

    best_label = top3[0]["label"]
    best_conf = top3[0]["prob"]

    if best_conf < CONF_THRESHOLD:
        return {
            "label": "Uncertain",
            "confidence": float(best_conf),
            "top3": top3,
            "message": "Low confidence. Image may be out-of-distribution. Please upload a close-up steel surface image."
        }

    return {
        "label": best_label,
        "confidence": float(best_conf),
        "top3": top3
    }
