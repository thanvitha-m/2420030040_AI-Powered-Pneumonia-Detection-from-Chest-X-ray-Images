from fastapi import HTTPException
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
    normal_probability = probabilities[0, 0].item()
    pneumonia_probability = probabilities[0, 1].item()

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
    image_base64,
    normal_probability,
    pneumonia_probability
    )


# ==========================================
# Home page
# ==========================================


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#0b1220">
    <title>PulmoAI | Chest X-ray Research</title>

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --bg: #0b1220;
            --panel: #111c2e;
            --panel2: #162338;
            --border: #26374e;
            --text: #f1f5fb;
            --muted: #94a6bd;
            --teal: #43dec0;
            --blue: #7ba9ff;
            --danger: #ff8c9a;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            background:
                radial-gradient(ellipse at 50% -20%, #183b50 0%, transparent 48%),
                var(--bg);
            color: var(--text);
            font-family: "DM Sans", sans-serif;
            min-height: 100vh;
        }

        button, input { font: inherit; }

        .topbar {
            height: 76px;
            border-bottom: 1px solid rgba(148,166,189,.14);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 6%;
            gap: 20px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: Manrope, sans-serif;
            font-weight: 800;
            font-size: 21px;
            letter-spacing: -.7px;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 13px;
            color: var(--teal);
            background: #163c43;
            border: 1px solid #28645e;
            font-size: 23px;
        }

        .brand span { color: var(--teal); }

        .top-status {
            display: flex;
            align-items: center;
            gap: 9px;
            color: #c1d2e2;
            font-size: 12px;
            border: 1px solid var(--border);
            padding: 9px 13px;
            border-radius: 30px;
            background: rgba(17,28,46,.75);
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--teal);
            box-shadow: 0 0 12px #43dec077;
        }

        main {
            width: min(1120px, 90%);
            margin: 0 auto;
            padding: 58px 0 38px;
        }

        .eyebrow {
            color: var(--teal);
            text-transform: uppercase;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 2.5px;
            margin-bottom: 17px;
        }

        h1 {
            font-family: Manrope, sans-serif;
            font-size: clamp(34px, 5vw, 54px);
            line-height: 1.12;
            letter-spacing: -2px;
            margin: 0;
            max-width: 750px;
        }

        h1 span { color: var(--teal); }

        .intro {
            color: var(--muted);
            max-width: 670px;
            font-size: 15px;
            line-height: 1.8;
            margin: 20px 0 34px;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-bottom: 25px;
        }

        .metric {
            background: rgba(17,28,46,.78);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 18px 20px;
        }

        .metric-label {
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 9px;
        }

        .metric-value {
            font-family: Manrope, sans-serif;
            font-size: 16px;
            font-weight: 700;
        }

        .workspace {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 290px;
            gap: 20px;
            align-items: start;
        }

        .card {
            background: linear-gradient(145deg, #142136, #101a2a);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 27px;
            box-shadow: 0 20px 60px #00000018;
        }

        .section-heading {
            font-family: Manrope, sans-serif;
            font-size: 18px;
            font-weight: 800;
            margin: 0 0 7px;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.7;
            margin: 0 0 23px;
        }

        .dropzone {
            position: relative;
            border: 1px dashed #46647d;
            border-radius: 16px;
            background: #0c1727;
            padding: 30px 20px;
            text-align: center;
            transition: .2s ease;
        }

        .dropzone.dragover {
            border-color: var(--teal);
            background: #123039;
        }

        .upload-icon {
            width: 55px;
            height: 55px;
            margin: 0 auto 15px;
            border-radius: 17px;
            display: grid;
            place-items: center;
            color: var(--teal);
            background: #16383d;
            font-size: 26px;
        }

        .drop-title {
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 8px;
        }

        .drop-help {
            color: var(--muted);
            font-size: 12px;
            line-height: 1.7;
        }

        .browse {
            display: inline-block;
            color: var(--teal);
            font-weight: 700;
            cursor: pointer;
            margin-top: 12px;
        }

        #file {
            position: absolute;
            width: 1px;
            height: 1px;
            opacity: 0;
            pointer-events: none;
        }

        .file-info {
            display: none;
            margin-top: 16px;
            background: #16283a;
            padding: 12px 14px;
            border-radius: 10px;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            font-size: 12px;
            color: #d9e7f5;
            overflow-wrap: anywhere;
        }

        .file-info button {
            background: none;
            border: 0;
            color: var(--danger);
            cursor: pointer;
            font-weight: 700;
        }

        .primary-btn {
            width: 100%;
            margin-top: 19px;
            padding: 15px 18px;
            border: 0;
            border-radius: 11px;
            background: var(--teal);
            color: #06211f;
            font-weight: 800;
            cursor: pointer;
            transition: transform .2s, filter .2s;
        }

        .primary-btn:hover { filter: brightness(1.08); transform: translateY(-1px); }
        .primary-btn:disabled { opacity: .55; cursor: wait; transform: none; }

        .side-card { margin-bottom: 16px; }

        .side-number {
            color: var(--teal);
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1.5px;
            margin-bottom: 10px;
        }

        .side-title {
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 7px;
        }

        .side-text {
            color: var(--muted);
            font-size: 12px;
            line-height: 1.8;
        }

        .tag {
            display: inline-block;
            font-size: 11px;
            color: #a9c4ff;
            border: 1px solid #344a70;
            background: #1a2a43;
            border-radius: 7px;
            padding: 6px 9px;
            margin: 5px 4px 0 0;
        }

        #result { margin-top: 23px; }

        .loading {
            padding: 25px;
            border: 1px solid var(--border);
            border-radius: 14px;
            color: var(--muted);
            text-align: center;
            font-size: 13px;
        }

        .spinner {
            width: 30px;
            height: 30px;
            margin: 0 auto 13px;
            border: 3px solid #34465a;
            border-top-color: var(--teal);
            border-radius: 50%;
            animation: spin .8s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .result-card {
            background: #0c1727;
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 23px;
        }

        .result-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 22px;
        }

        .result-label {
            color: var(--muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 9px;
        }

        .result-name {
            font-family: Manrope, sans-serif;
            font-size: 26px;
            font-weight: 800;
            overflow-wrap: anywhere;
        }

        .result-name.normal { color: var(--teal); }
        .result-name.pneumonia { color: #ffad91; }

        .confidence-pill {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 13px 16px;
            text-align: right;
        }

        .confidence-value { font-size: 23px; font-weight: 800; }
        .confidence-caption { color: var(--muted); font-size: 11px; margin-top: 4px; }

        .probability-section {
        margin: 22px 0;
        padding: 18px;
        background: #111e30;
        border: 1px solid var(--border);
        border-radius: 12px;
        }
        
        .probability-title {
        font-family: Manrope, sans-serif;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 18px;
        }

        .probability-row {
        margin-bottom: 16px;
        }

        .probability-row:last-child {
        margin-bottom: 0;
        }

        .probability-label {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 12px;
        margin-bottom: 8px;
        }

        .probability-track {
        height: 9px;
        overflow: hidden;
        background: #26374e;
        border-radius: 20px;
        }

        .probability-fill {
        height: 100%;
        width: 0%;
        border-radius: 20px;
        transition: width .5s ease;
        }

        .normal-fill {
        background: var(--teal);
        }

        .pneumonia-fill {
        background: var(--blue);
        }
        .disclaimer-inline {
            color: #c0cad7;
            background: #202b3b;
            border-radius: 9px;
            padding: 12px 14px;
            font-size: 12px;
            line-height: 1.7;
            margin-bottom: 22px;
        }

        .cam-image {
            width: 100%;
            display: block;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: #09111d;
        }

        .error {
            color: #ffc2c9;
            background: #381f2b;
            border: 1px solid #70404e;
            padding: 15px;
            border-radius: 11px;
            font-size: 13px;
            line-height: 1.7;
        }

        .footer {
            color: #71849c;
            text-align: center;
            font-size: 11px;
            line-height: 1.8;
            margin-top: 35px;
            padding: 20px 0 5px;
            border-top: 1px solid #1d2a3d;
        }

        @media (max-width: 760px) {
            main { padding-top: 38px; }
            .workspace { grid-template-columns: 1fr; }
            .metrics { grid-template-columns: 1fr; }
            .topbar { padding: 0 5%; }
            .top-status { font-size: 10px; }
            .card { padding: 20px; }
            h1 { letter-spacing: -1px; }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: .01ms !important;
                transition-duration: .01ms !important;
            }
        }
    </style>
</head>

<body>
<header class="topbar">
    <div class="brand">
        <div class="brand-icon" aria-hidden="true">✳</div>
        Pulmo<span>AI</span>
    </div>
    <div class="top-status">
        <span class="dot"></span>
        Research prototype
    </div>
</header>

<main>
    <div class="eyebrow">AI-assisted image analysis</div>
    <h1>Chest X-ray <span>Intelligence.</span></h1>
    <p class="intro">
        Explore a DenseNet-121 research model for chest X-ray image
        classification, with Grad-CAM visualizations to inspect image
        regions influencing its output.
    </p>

    <section class="metrics" aria-label="Model information">
        <div class="metric">
            <div class="metric-label">Classification backbone</div>
            <div class="metric-value">DenseNet-121</div>
        </div>
        <div class="metric">
            <div class="metric-label">Output categories</div>
            <div class="metric-value">Normal / Pneumonia</div>
        </div>
        <div class="metric">
            <div class="metric-label">Explainability</div>
            <div class="metric-value">Grad-CAM heatmap</div>
        </div>
    </section>

    <div class="workspace">
        <section class="card">
            <h2 class="section-heading">Analyze an X-ray</h2>
            <p class="section-subtitle">
                Upload a chest X-ray image to generate an experimental
                classification and visualization.
            </p>

            <form id="uploadForm">
                <div class="dropzone" id="dropzone">
                    <div class="upload-icon" aria-hidden="true">↑</div>
                    <div class="drop-title">Drop your image here</div>
                    <div class="drop-help">
                        Choose a JPEG, PNG, or other supported image file
                        from your device.
                    </div>
                    <label for="file" class="browse">Browse files</label>
                    <input type="file" id="file"
                           accept=".jpg,.jpeg,.png,.bmp,.webp,image/jpeg,image/png,image/webp"
                           required>
                </div>

                <div class="file-info" id="fileInfo">
                    <span id="fileName"></span>
                    <button type="button" id="removeFile">Remove</button>
                </div>

                <button class="primary-btn" type="submit" id="analyzeBtn">
                    Analyze image &nbsp; →
                </button>
            </form>

            <div id="result" aria-live="polite"></div>
        </section>

        <aside>
            <section class="card side-card">
                <div class="side-number">01 / MODEL</div>
                <div class="side-title">Deep learning classification</div>
                <p class="side-text">
                    The uploaded image is processed by the saved DenseNet-121
                    model to produce one of two class predictions.
                </p>
                <span class="tag">PyTorch</span>
                <span class="tag">DenseNet-121</span>
            </section>

            <section class="card side-card">
                <div class="side-number">02 / EXPLAINABILITY</div>
                <div class="side-title">Visual model exploration</div>
                <p class="side-text">
                    Grad-CAM creates a heatmap highlighting image regions
                    associated with the selected model output. It is not
                    proof of disease or a clinical explanation.
                </p>
                <span class="tag">Grad-CAM</span>
                <span class="tag">Heatmap</span>
            </section>

            <section class="card">
                <div class="side-number">IMPORTANT</div>
                <div class="side-title">Research use only</div>
                <p class="side-text">
                    This experimental project is not a medical device.
                    Predictions and confidence scores must not be used
                    to diagnose, rule out, or treat pneumonia. Consult
                    a qualified healthcare professional for medical concerns.
                </p>
            </section>
        </aside>
    </div>

    <footer class="footer">
        PULMOAI · CHEST X-RAY RESEARCH PROTOTYPE<br>
        DenseNet-121 classification with Grad-CAM visualization
    </footer>
</main>

<script>
const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("file");
const dropzone = document.getElementById("dropzone");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const result = document.getElementById("result");
const analyzeBtn = document.getElementById("analyzeBtn");

function showFile(file) {
    if (!file) return;

    const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
    const extensionAllowed = /\.(jpe?g|png|webp|bmp)$/i.test(file.name);

    if (!allowed.includes(file.type) && !extensionAllowed) {
        fileInput.value = "";
        fileInfo.style.display = "none";
        result.textContent = "";
        const error = document.createElement("div");
        error.className = "error";
        error.textContent = "Please choose a supported image file (JPEG, PNG, WEBP, or BMP).";
        result.appendChild(error);
        return;
    }

    fileName.textContent = file.name;
    fileInfo.style.display = "flex";
    result.textContent = "";
}

fileInput.addEventListener("change", () => showFile(fileInput.files[0]));

document.getElementById("removeFile").addEventListener("click", () => {
    fileInput.value = "";
    fileInfo.style.display = "none";
    result.textContent = "";
});

["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", event => {
    const files = event.dataTransfer.files;
    if (!files.length) return;

    try {
        const transfer = new DataTransfer();
        transfer.items.add(files[0]);
        fileInput.files = transfer.files;
        showFile(files[0]);
    } catch {
        result.textContent = "Please use Browse files to select your image.";
    }
});

form.addEventListener("submit", async event => {
    event.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        result.textContent = "Please select an image first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing image...";
    result.innerHTML =
        '<div class="loading"><div class="spinner"></div>' +
        'Processing image and generating Grad-CAM visualization...<br>' +
        'This may take a little while on CPU.</div>';

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error || data.detail) {
            throw new Error(data.detail || data.error || "Analysis failed.");
        }

        if (!["NORMAL", "PNEUMONIA"].includes(data.prediction) ||
            typeof data.confidence !== "number" ||
            !data.gradcam) {
            throw new Error("The server returned an incomplete result. Check the terminal for details.");
        }

        const resultCard = document.createElement("div");
        resultCard.className = "result-card";

        const top = document.createElement("div");
        top.className = "result-top";

        const left = document.createElement("div");
        const label = document.createElement("div");
        label.className = "result-label";
        label.textContent = "Model prediction";

        const name = document.createElement("div");
        name.className = "result-name " +
            (data.prediction === "NORMAL" ? "normal" : "pneumonia");
        name.textContent = data.prediction;

        left.append(label, name);

        const confidence = document.createElement("div");
        confidence.className = "confidence-pill";

        const value = document.createElement("div");
        value.className = "confidence-value";
        value.textContent = data.confidence.toFixed(2) + "%";

        const caption = document.createElement("div");
        caption.className = "confidence-caption";
        caption.textContent = "Model confidence score";

        confidence.append(value, caption);
        top.append(left, confidence);

        const note = document.createElement("div");
        note.className = "disclaimer-inline";
        note.textContent =
            "Experimental model output only. Confidence is not a measure " +
            "of clinical certainty. This result cannot diagnose or exclude pneumonia.";
            // Probability bars
            const probabilitySection = document.createElement("div");
            probabilitySection.className = "probability-section";
            
            const probabilityTitle = document.createElement("div");
            probabilityTitle.className = "probability-title";
            probabilityTitle.textContent = "Class probability scores";
            
            probabilitySection.appendChild(probabilityTitle);
            
            function addProbabilityBar(labelText, probability, fillClass) {
            const percentage = Number(probability);
            
            if (!Number.isFinite(percentage)) {
            throw new Error("The server returned invalid probability scores.");
            }
            
            const safePercentage = Math.max(0, Math.min(100, percentage));
            const row = document.createElement("div");
            row.className = "probability-row";
            
            const label = document.createElement("div");
            label.className = "probability-label";
            
            const name = document.createElement("span");
            name.textContent = labelText;
            
            const score = document.createElement("span");
            score.textContent = safePercentage.toFixed(2) + "%";
            
            label.append(name, score);
            
            const track = document.createElement("div");
            track.className = "probability-track";
            
            const fill = document.createElement("div");
            fill.className = "probability-fill " + fillClass;
            fill.style.width = safePercentage + "%";
            
            track.appendChild(fill);
            row.append(label, track);
            probabilitySection.appendChild(row);
            }
            
            addProbabilityBar(
            "NORMAL",
            data.normal_probability,
            "normal-fill"
            );
            
            addProbabilityBar(
            "PNEUMONIA",
            data.pneumonia_probability,
            "pneumonia-fill"
            );
        const heading = document.createElement("h3");
        heading.className = "section-heading";
        heading.textContent = "Grad-CAM visualization";

        const subtitle = document.createElement("p");
        subtitle.className = "section-subtitle";
        subtitle.textContent =
            "The heatmap shows regions associated with the model's selected output.";

        const image = document.createElement("img");
        image.className = "cam-image";
        image.alt = "Original X-ray, Grad-CAM heatmap, and overlay";
        image.src = "data:image/png;base64," + data.gradcam;

        resultCard.append(
        top,
        note,
        probabilitySection,
        heading,
        subtitle,
        image
        );
        result.replaceChildren(resultCard);
        resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (error) {
        const errorBox = document.createElement("div");
        errorBox.className = "error";
        errorBox.textContent =
            "Unable to analyze this image. " + error.message +
            " Please try another supported image.";
        result.replaceChildren(errorBox);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze image →";
    }
});
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

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        (
            predicted_class,
            confidence,
            gradcam_image,
            normal_probability,
            pneumonia_probability
        ) = generate_gradcam(image)

        class_names = ["NORMAL", "PNEUMONIA"]
        prediction = class_names[predicted_class]

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "normal_probability": round(normal_probability * 100, 2),
            "pneumonia_probability": round(pneumonia_probability * 100, 2),
            "gradcam": gradcam_image
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Prediction error: {error}")
        raise HTTPException(
            status_code=400,
            detail="Could not process this image. Please try a valid image file."
        )
