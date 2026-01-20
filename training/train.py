import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


def main():
    # Project paths
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "neu_dataset"
    train_dir = data_root / "train"
    valid_dir = data_root / "valid"

    out_dir = project_root / "training" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters (safe defaults)
    img_size = 224
    batch_size = 32
    num_epochs = 8
    lr = 1e-3
    num_workers = 0  # keep 0 for Windows stability

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (ResNet expects 3-channel images + ImageNet normalization)
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Grayscale(num_output_channels=3),  # IMPORTANT for grayscale dataset
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),  # IMPORTANT for grayscale dataset
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    valid_ds = datasets.ImageFolder(str(valid_dir), transform=valid_tfms)

    # Save label mapping (needed for inference-service)
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {"class_to_idx": train_ds.class_to_idx, "idx_to_class": idx_to_class},
            f,
            indent=2
        )
    print(f"Saved label mapping to: {labels_path}")
    print("Classes:", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_ds.classes)

    # Model: ResNet18 transfer learning
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_path = out_dir / "model.pth"

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"Epoch {epoch}/{num_epochs} [valid]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"\nEpoch {epoch}/{num_epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved BEST model to {best_model_path} (val_acc={best_val_acc:.4f})")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
