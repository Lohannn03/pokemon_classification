import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 5
EPOCHS = 20

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Validation transform
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Dataset and dataloader
train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
val_dataset = datasets.ImageFolder("data/val", transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

num_classes = len(train_dataset.classes)

# Save class names
with open("class_names.txt", "w") as f:
    for class_name in train_dataset.classes:
        f.write(class_name + "\n")

print("Number of classes:", num_classes)


# Build model
def build_model(model_name="resnet18", pretrained=True, fine_tune_type="all"):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Set trainable layers
    if pretrained:
        if fine_tune_type == "fc":
            for param in model.parameters():
                param.requires_grad = False

        elif fine_tune_type == "partial":
            for param in model.parameters():
                param.requires_grad = False

            for param in model.layer4.parameters():
                param.requires_grad = True

        elif fine_tune_type == "all":
            for param in model.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unsupported fine_tune_type: {fine_tune_type}")

    else:
        for param in model.parameters():
            param.requires_grad = True

    # Replace final classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model.to(device)


# Create display name
def make_display_name(model_name, pretrained, fine_tune_type):
    if model_name == "resnet18":
        model_display = "ResNet18"
    elif model_name == "resnet50":
        model_display = "ResNet50"
    else:
        model_display = model_name

    if fine_tune_type == "all":
        ft_display = "All"
    elif fine_tune_type == "fc":
        ft_display = "FC Only"
    elif fine_tune_type == "partial":
        ft_display = "Partial"
    else:
        ft_display = fine_tune_type

    return f"{model_display}: Pretrained - {pretrained}, Fine-tuning - {ft_display}"


# Create file name
def make_file_name(model_name, pretrained, fine_tune_type):
    return f"{model_name}_pre{pretrained}_ft{fine_tune_type}"


# Train and validate model
def train_model(model, display_name, file_name):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_acc = 0.0
    best_val_loss = float("inf")
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "precision": [],
        "recall": []
    }

    print(f"\n=== {display_name} ===")

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss_total = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()

                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = np.mean(all_preds == all_labels)

        precision, recall, _, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(acc)
        history["precision"].append(precision)
        history["recall"].append(recall)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"acc={acc:.4f}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}"
        )

        # Best model selection
        improved_acc = acc > best_acc
        stable_loss = avg_val_loss <= best_val_loss * 1.05

        if improved_acc and stable_loss:
            best_acc = acc
            best_val_loss = avg_val_loss

            torch.save(model.state_dict(), f"best_{file_name}.pth")

            print(
                f"Best model saved "
                f"(best_acc={best_acc:.4f}, best_val_loss={best_val_loss:.4f})"
            )

            early_stop_counter = 0

        elif improved_acc and not stable_loss:
            print(
                f"Accuracy improved but validation loss is unstable. "
                f"Model not saved. "
                f"(acc={acc:.4f}, val_loss={avg_val_loss:.4f})"
            )
            early_stop_counter += 1

        else:
            early_stop_counter += 1
            print(f"No valid improvement ({early_stop_counter}/{PATIENCE})")

        # Early stopping
        if early_stop_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(
        f"Finished: {display_name} | "
        f"Best Acc={best_acc:.4f}, Best Val Loss={best_val_loss:.4f}"
    )

    return history, best_acc, best_val_loss


# Experiment settings
if __name__ == "__main__":
    configs = [
        ("resnet18", True, "all"),
    ]

    all_histories = {}
    final_results = []

    for model_name, pretrained, fine_tune_type in configs:
        display_name = make_display_name(model_name, pretrained, fine_tune_type)
        file_name = make_file_name(model_name, pretrained, fine_tune_type)

        model = build_model(
            model_name=model_name,
            pretrained=pretrained,
            fine_tune_type=fine_tune_type
        )

        history, best_acc, best_val_loss = train_model(
            model=model,
            display_name=display_name,
            file_name=file_name
        )

        all_histories[display_name] = history
        final_results.append({
            "Experiment": display_name,
            "Best Acc": best_acc,
            "Best Val Loss": best_val_loss
        })

    print("\n========== Final Results ==========")

    for result in final_results:
        print(
            f"{result['Experiment']} | "
            f"Best Acc={result['Best Acc']:.4f}, "
            f"Best Val Loss={result['Best Val Loss']:.4f}"
        )

    print("\nAll experiments finished.")