from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -----------------------------
# Transform
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
# Load DenseNet-121
# -----------------------------
print("Loading model...")

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

print("Model loaded successfully.")


# -----------------------------
# Find correctly classified
# pneumonia image
# -----------------------------
pneumonia_dir = DATA_DIR / "test" / "PNEUMONIA"

pneumonia_images = list(
    pneumonia_dir.glob("*.jpeg")
)

print(
    f"Searching {len(pneumonia_images)} "
    "pneumonia test images..."
)

selected_image = None
selected_confidence = 0.0

for image_path in pneumonia_images:

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

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0, prediction
        ].item()

    # Class 1 = PNEUMONIA
    if prediction == 1:

        selected_image = image_path
        selected_confidence = confidence

        break


# -----------------------------
# Check whether image was found
# -----------------------------
if selected_image is None:

    print(
        "No correctly classified "
        "pneumonia image was found."
    )

    raise SystemExit


print("\nSelected image:")
print(selected_image)

print(
    f"Prediction confidence: "
    f"{selected_confidence * 100:.2f}%"
)


# -----------------------------
# Load selected image
# -----------------------------
original_image = Image.open(
    selected_image
).convert("RGB")

input_tensor = transform(
    original_image
).unsqueeze(0).to(device)

input_tensor.requires_grad = True


# -----------------------------
# Grad-CAM hooks
# -----------------------------
activations = []
gradients = []


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


target_layer = model.features.denseblock4

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# -----------------------------
# Forward pass
# -----------------------------
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
    0, predicted_class
].item()

print("\nPrediction:")
print("Class:", predicted_label)
print(
    f"Confidence: {confidence * 100:.2f}%"
)


# -----------------------------
# Backward pass
# -----------------------------
model.zero_grad()

target_score = output[
    0,
    predicted_class
]

target_score.backward()


# -----------------------------
# Generate Grad-CAM
# -----------------------------
activation = activations[0]
gradient = gradients[0]

weights = gradient.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (
    weights * activation
).sum(dim=1).squeeze()

cam = torch.relu(cam)

cam = cam.detach().cpu().numpy()

cam = cam - cam.min()

if cam.max() != 0:
    cam = cam / cam.max()


# -----------------------------
# Create heatmap
# -----------------------------
original_array = np.array(
    original_image
)

height, width = original_array.shape[:2]

heatmap = cv2.resize(
    cam,
    (width, height)
)

heatmap = np.uint8(
    255 * heatmap
)

heatmap_color = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)


# -----------------------------
# Create overlay
# -----------------------------
overlay = (
    0.6 * original_array
    + 0.4 * heatmap_color
)

overlay = np.uint8(
    np.clip(overlay, 0, 255)
)


# -----------------------------
# Save Grad-CAM result
# -----------------------------
output_path = (
    RESULTS_DIR /
    "gradcam_correct_pneumonia.png"
)

plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)

plt.imshow(original_array)

plt.title(
    "Original X-ray\n"
    "Actual: PNEUMONIA"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(heatmap)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(overlay)

plt.title(
    f"Prediction: {predicted_label}\n"
    f"Confidence: {confidence * 100:.2f}%"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nGrad-CAM result saved at:")
print(output_path)

print("\nGrad-CAM generation complete.")