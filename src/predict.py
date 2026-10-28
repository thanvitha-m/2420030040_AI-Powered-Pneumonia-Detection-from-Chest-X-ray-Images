from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError


# -----------------------------
# Paths and configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "densenet121_pneumonia.pth"

DEFAULT_CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path\\to\\xray.jpeg")
        return 2

    image_path = Path(sys.argv[1]).expanduser()

    if not image_path.is_file():
        print(f"Error: Image not found: {image_path}")
        return 2

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        print(f"Error: Unsupported image extension: {image_path.suffix}")
        return 2

    if not MODEL_PATH.is_file():
        print(f"Error: Model checkpoint not found: {MODEL_PATH}")
        return 2

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------
    # Image transformation
    # Keep preprocessing consistent with training.
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # -----------------------------
    # Load trusted checkpoint
    # -----------------------------
    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        print(f"Error loading model checkpoint: {exc}")
        return 1

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("classes", DEFAULT_CLASSES)
    else:
        state_dict = checkpoint
        class_names = DEFAULT_CLASSES

    if not isinstance(class_names, (list, tuple)) or len(class_names) != 2:
        print("Error: Checkpoint must define exactly two class labels.")
        return 1

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    try:
        model.load_state_dict(state_dict)
    except (RuntimeError, TypeError) as exc:
        print(f"Error: Checkpoint does not match DenseNet-121: {exc}")
        return 1

    model = model.to(device)
    model.eval()

    # -----------------------------
    # Load image and predict
    # -----------------------------
    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        print(f"Error: Could not read image: {exc}")
        return 2

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())

    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    # -----------------------------
    # Display result
    # -----------------------------
    print("\n" + "=" * 38)
    print("PNEUMONIA DETECTION — RESEARCH PROTOTYPE")
    print("=" * 38)
    print(f"Image:       {image_path.resolve()}")
    print(f"Prediction:  {predicted_label}")
    print(f"Confidence:  {confidence * 100:.2f}%")
    print("\nClass probabilities:")

    for label, probability in zip(class_names, probabilities.tolist()):
        print(f"  {label}: {probability * 100:.2f}%")

    print("\nImportant: Experimental output only.")
    print("Not a medical diagnosis. Consult a qualified clinician.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
