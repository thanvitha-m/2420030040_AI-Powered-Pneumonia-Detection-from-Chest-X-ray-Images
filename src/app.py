from pathlib import Path
import io
import base64

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "densenet121_pneumonia.pth"


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(
    title="Pneumonia Detection API",
    description="DenseNet-121 chest X-ray classification with Grad-CAM",
    version="1.0"
)


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# Image transformation
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Load DenseNet-121
# ==========================================

print("Loading DenseNet-121...")

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


# ==========================================
# Grad-CAM storage
# ==========================================

activations = []
gradients = []


def forward_hook(module, input, output):
    activations.clear()
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.clear()
    gradients.append(grad_output[0])


target_layer = model.features.denseblock4

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ==========================================
# Create Grad-CAM
# ==========================================

def generate_gradcam(image):

    activations.clear()
    gradients.clear()

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    input_tensor.requires_grad = True

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    model.zero_grad()

    target_score = output[
        0,
        predicted_class
    ]

    target_score.backward()

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

    # Original image
    original_array = np.array(image)

    height, width = original_array.shape[:2]

    # Resize CAM
    heatmap = cv2.resize(
        cam,
        (width, height)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # Heatmap
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = (
        0.6 * original_array
        + 0.4 * heatmap_color
    )

    overlay = np.uint8(
        np.clip(overlay, 0, 255)
    )

    # Create combined figure
    figure = plt.figure(
        figsize=(12, 4)
    )

    plt.subplot(1, 3, 1)
    plt.imshow(original_array)
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(
        f"{['NORMAL', 'PNEUMONIA'][predicted_class]}\n"
        f"{confidence * 100:.2f}%"
    )
    plt.axis("off")

    plt.tight_layout()

    # Save figure into memory
    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(figure)

    buffer.seek(0)

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        predicted_class,
        confidence,
        image_base64
    )


# ==========================================
# Home page
# ==========================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>Pneumonia Detection</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 40px auto;
                padding: 20px;
                text-align: center;
            }

            h1 {
                margin-bottom: 5px;
            }

            .subtitle {
                margin-bottom: 30px;
            }

            .box {
                border: 1px solid #ccc;
                border-radius: 12px;
                padding: 30px;
            }

            button {
                padding: 10px 25px;
                margin-top: 20px;
                cursor: pointer;
                font-size: 16px;
            }

            #result {
                margin-top: 30px;
            }

            .prediction {
                font-size: 25px;
                font-weight: bold;
                margin: 15px;
            }

            .confidence {
                font-size: 20px;
                margin-bottom: 25px;
            }

            #gradcam {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 8px;
            }

            .note {
                margin-top: 30px;
                font-size: 14px;
                color: #666;
            }

        </style>

    </head>

    <body>

        <h1>Automated Pneumonia Detection</h1>

        <div class="subtitle">
            DenseNet-121 Chest X-ray Classification
            with Grad-CAM Explainability
        </div>

        <div class="box">

            <form id="uploadForm">

                <input
                    type="file"
                    id="file"
                    accept="image/*"
                    required
                >

                <br>

                <button type="submit">
                    Analyze X-ray
                </button>

            </form>

            <div id="result"></div>

        </div>

        <p class="note">
            Research prototype only.
            This system is not a medical diagnosis.
        </p>


        <script>

            document
                .getElementById("uploadForm")
                .addEventListener(
                    "submit",
                    async function(event) {

                        event.preventDefault();

                        const fileInput =
                            document.getElementById("file");

                        const result =
                            document.getElementById("result");

                        const formData =
                            new FormData();

                        formData.append(
                            "file",
                            fileInput.files[0]
                        );

                        result.innerHTML =
                            "Analyzing X-ray...";

                        try {

                            const response =
                                await fetch(
                                    "/predict",
                                    {
                                        method: "POST",
                                        body: formData
                                    }
                                );

                            const data =
                                await response.json();

                            if (!response.ok) {

                                result.innerHTML =
                                    "Error: " +
                                    data.detail;

                                return;
                            }

                            result.innerHTML =

                                '<div class="prediction">' +
                                'Prediction: ' +
                                data.prediction +
                                '</div>' +

                                '<div class="confidence">' +
                                'Confidence: ' +
                                data.confidence +
                                '%' +
                                '</div>' +

                                '<h2>Grad-CAM Explanation</h2>' +

                                '<img id="gradcam" src="data:image/png;base64,' +
                                data.gradcam +
                                '">';

                        } catch (error) {

                            result.innerHTML =
                                "Could not connect to the server.";

                        }

                    }
                );

        </script>

    </body>

    </html>
    """


# ==========================================
# Prediction endpoint
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        (
            predicted_class,
            confidence,
            gradcam_image
        ) = generate_gradcam(image)

        class_names = [
            "NORMAL",
            "PNEUMONIA"
        ]

        prediction = class_names[
            predicted_class
        ]

        return {
            "prediction": prediction,
            "confidence": round(
                confidence * 100,
                2
            ),
            "gradcam": gradcam_image
        }

    except Exception as error:

        return {
            "error": str(error)
        }