import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "neu_dataset"
    test_dir = data_root / "test"

    out_dir = project_root / "training" / "outputs"
    model_path = out_dir / "model.pth"
    labels_path = out_dir / "labels.json"

    img_size = 224
    batch_size = 32
    num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label mapping
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # idx_to_class keys might load as strings from JSON
    idx_to_class = {int(k): v for k, v in labels["idx_to_class"].items()}

    # Transform must match training/valid (including grayscale -> 3 channels)
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_ds = datasets.ImageFolder(str(test_dir), transform=tfms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(test_ds.classes)

    # Build model and load weights
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    # Accuracy
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    print(f"\nTEST accuracy: {acc:.4f}\n")

    # Report + confusion matrix
    target_names = [idx_to_class[i] for i in range(num_classes)]

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
