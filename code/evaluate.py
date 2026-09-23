from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "chest_xray"
MODEL_PATH = BASE_DIR / "models" / "densenet121_pneumonia.pth"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Loading model...")


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Test dataset
# -----------------------------
test_dir = DATA_DIR / "test"

test_dataset = ImageFolder(
    test_dir,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)


# -----------------------------
# Load DenseNet-121
# -----------------------------
model = models.densenet121(weights=None)

num_features = model.classifier.in_features

model.classifier = nn.Linear(
    num_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Our training script saved a state_dict
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# -----------------------------
# Evaluation
# -----------------------------
all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of PNEUMONIA
        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )


# Convert to NumPy arrays
y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)


# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


# -----------------------------
# Print results
# -----------------------------
print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")
print(f"ROC-AUC  :  {roc_auc:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes,
        zero_division=0
    )
)


# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred
)

print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("DenseNet-121 Confusion Matrix")

plt.tight_layout()

cm_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

print("\nConfusion matrix saved at:")
print(cm_path)


print("\nEvaluation complete.")