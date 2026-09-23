from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "densenet121_pneumonia.pth"


# -----------------------------
# Check image path
# -----------------------------
if len(sys.argv) < 2:
    print("Usage:")
    print("python code\\predict.py path\\to\\xray.jpeg")
    raise SystemExit

image_path = Path(sys.argv[1])

if not image_path.exists():
    print("Image not found:")
    print(image_path)
    raise SystemExit


# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -----------------------------
# Image transformation
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
# Load model
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

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()


# -----------------------------
# Load and predict
# -----------------------------
image = Image.open(
    image_path
).convert("RGB")

input_tensor = transform(
    image
).unsqueeze(0).to(device)

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

class_names = [
    "NORMAL",
    "PNEUMONIA"
]

predicted_label = class_names[
    predicted_class
]

confidence = probabilities[
    0,
    predicted_class
].item()


# -----------------------------
# Display result
# -----------------------------
print("\n==============================")
print("PNEUMONIA DETECTION RESULT")
print("==============================")

print("Image:", image_path)
print("Prediction:", predicted_label)
print(
    f"Confidence: {confidence * 100:.2f}%"
)

print("\nNote: This is a research prototype prediction,")
print("not a medical diagnosis.")