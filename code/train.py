from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data_preprocessing import get_dataloaders


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 0.0001

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "densenet121_pneumonia.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

train_loader, validation_loader, test_loader, classes = get_dataloaders(
    batch_size=BATCH_SIZE
)

print("Classes:", classes)


# --------------------------------------------------
# Create DenseNet-121
# --------------------------------------------------

print("Loading pretrained DenseNet-121...")

weights = DenseNet121_Weights.DEFAULT

model = densenet121(weights=weights)


# Freeze pretrained feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False


# Replace final classification layer
input_features = model.classifier.in_features

model.classifier = nn.Linear(
    input_features,
    2
)

model = model.to(DEVICE)


# --------------------------------------------------
# Class-weighted loss
# --------------------------------------------------

# NORMAL = 0
# PNEUMONIA = 1
#
# The training data contains more pneumonia images,
# so class-weighted loss helps account for imbalance.

normal_count = 1341
pneumonia_count = 3875

total = normal_count + pneumonia_count

normal_weight = total / (2 * normal_count)
pneumonia_weight = total / (2 * pneumonia_count)

class_weights = torch.tensor(
    [normal_weight, pneumonia_weight],
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_one_epoch():

    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Validation function
# --------------------------------------------------

def validate():

    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    validation_loss = (
        running_loss / len(validation_loader)
    )

    validation_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return validation_loss, validation_accuracy


# --------------------------------------------------
# Training loop
# --------------------------------------------------

best_validation_accuracy = 0.0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    train_loss, train_accuracy = train_one_epoch()

    validation_loss, validation_accuracy = validate()

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )

    # Save best model
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "validation_accuracy": validation_accuracy
            },
            MODEL_PATH
        )

        print(
            f"Best model saved to: {MODEL_PATH}"
        )


print("\nTraining complete.")

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.4f}"
)

print(
    f"Model saved at: {MODEL_PATH}"
)