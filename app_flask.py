from flask import Flask, render_template_string, request, jsonify, send_from_directory
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import json
import os
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import base64
import io

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_NAME = "mobilenetv2_100"
IMG_SIZE = 224
N_FOLDS = 5

CLASS_NAMES = ["dry", "normal", "oily"]

PH_MAPPING = {
    "oily": "4.5 – 5.0",
    "normal": "5.0 – 5.5",
    "dry": "5.5 – 6.5"
}

SKINCARE_ADVICE = {
    "oily": {
        "title": "Oily Skin Care",
        "tips": [
            "Use gel-based or foaming cleansers",
            "Look for non-comedogenic products",
            "Use lightweight, oil-free moisturizers",
            "Consider salicylic acid for pore minimization",
            "Use mattifying primers to control shine"
        ]
    },
    "normal": {
        "title": "Normal Skin Care",
        "tips": [
            "Maintain balance with gentle cleansers",
            "Use lightweight moisturizers",
            "Continue sun protection daily",
            "Stay hydrated",
            "Use mild exfoliants occasionally"
        ]
    },
    "dry": {
        "title": "Dry Skin Care",
        "tips": [
            "Use creamy, hydrating cleansers",
            "Apply moisturizer while skin is damp",
            "Use products with hyaluronic acid",
            "Avoid hot water when washing face",
            "Consider barrier repair creams"
        ]
    }
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])


class SkinClassifier(torch.nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.features_dim = self.backbone.num_features
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.features_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


models = []


def load_models():
    global models
    model = SkinClassifier(MODEL_NAME, num_classes=len(CLASS_NAMES), dropout=0.5)
    model.eval()

    for fold in range(1, N_FOLDS + 1):
        model_path = os.path.join(MODEL_DIR, f"skink_model_fold{fold}.pth")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            models.append(model)


def detect_face(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        margin = int(max(w, h) * 0.2)
        h_img, w_img = img_array.shape[:2]

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_img, x + w + margin)
        y2 = min(h_img, y + h + margin)

        cropped = img_array[y1:y2, x1:x2]
        return Image.fromarray(cropped), True
    else:
        return image, False


def preprocess_image(image):
    cropped, face_detected = detect_face(image)
    processed = TRANSFORM(cropped)
    return processed.unsqueeze(0), cropped


def predict(model_list, image_tensor):
    all_probs = []

    with torch.no_grad():
        for model in model_list:
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)

    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    pred_idx = torch.argmax(avg_probs, dim=1).item()
    confidence = avg_probs[0, pred_idx].item()

    return CLASS_NAMES[pred_idx], confidence, avg_probs[0].tolist()


def generate_gradcam(model, image_tensor):
    target_layer = model.backbone.blocks[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=image_tensor)

    original_image = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    visualization = show_cam_on_image(original_image, grayscale_cam[0], use_rgb=True)

    return visualization


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Skin pH Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f8f9fa; }
        .container { max-width: 900px; margin-top: 30px; }
        .card { border: none; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .card-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px 15px 0 0 !important; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; }
        .btn-primary:hover { background: linear-gradient(135deg, #5568d3 0%, #674a87 100%); }
        .result-card { background: white; border-radius: 10px; padding: 20px; margin: 15px 0; }
        .skin-type { font-size: 24px; font-weight: bold; }
        .confidence { font-size: 18px; color: #666; }
        .progress { height: 25px; border-radius: 15px; }
        .progress-bar { font-weight: bold; }
        .tips-list { list-style: none; padding: 0; }
        .tips-list li { padding: 8px 0; border-bottom: 1px solid #eee; }
        .tips-list li:last-child { border-bottom: none; }
        .image-preview { max-width: 100%; border-radius: 10px; }
        .gradcam-img { max-width: 100%; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="card-header py-3">
                <h2 class="mb-0">🧴 AI-Based Skin pH Estimator</h2>
                <p class="mb-0">Upload a face image to analyze your skin type and estimate pH level</p>
            </div>
            <div class="card-body">
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label class="form-label">Choose Image</label>
                        <input type="file" class="form-control" id="imageInput" accept="image/*" required>
                    </div>
                    <button type="submit" class="btn btn-primary btn-lg w-100">Analyze Skin</button>
                </form>

                <div id="loading" class="text-center my-5" style="display: none;">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-3">Analyzing your skin...</p>
                </div>

                <div id="results" style="display: none;">
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <div class="result-card">
                                <h5>📷 Your Image</h5>
                                <img id="previewImage" class="image-preview img-fluid" alt="Uploaded Image">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="result-card">
                                <h5>🔍 Analysis Results</h5>
                                <p class="skin-type">Skin Type: <span id="skinType"></span></p>
                                <p>Estimated pH: <strong id="phRange"></strong></p>
                                <p class="confidence">Confidence: <span id="confidence"></span></p>
                            </div>
                            <div class="result-card">
                                <h5>📊 Probability Distribution</h5>
                                <div id="probDist"></div>
                            </div>
                        </div>
                    </div>

                    <div class="result-card">
                        <h5>🔬 Grad-CAM Attention Map</h5>
                        <p class="text-muted">Regions the model focused on for prediction</p>
                        <img id="gradcamImage" class="gradcam-img img-fluid" alt="Grad-CAM Visualization">
                    </div>

                    <div class="result-card">
                        <h5>💡 Skincare Recommendations</h5>
                        <h6 id="adviceTitle"></h6>
                        <ul id="adviceTips" class="tips-list"></ul>
                    </div>

                    <div class="alert alert-info mt-4">
                        <strong>Note:</strong> This is an AI-based estimation for educational purposes only. Consult a dermatologist for professional skin analysis.
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const fileInput = document.getElementById('imageInput');
            if (!fileInput.files[0]) return;

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                document.getElementById('previewImage').src = 'data:image/jpeg;base64,' + data.preview_image;
                document.getElementById('skinType').textContent = data.skin_type.charAt(0).toUpperCase() + data.skin_type.slice(1);
                document.getElementById('phRange').textContent = data.ph_range;
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';

                let probHtml = '';
                data.probabilities.forEach((prob, idx) => {
                    const labels = ['Dry', 'Normal', 'Oily'];
                    probHtml += '<p class="mb-2">' + labels[idx] + ': ' + (prob * 100).toFixed(1) + '%</p>';
                    probHtml += '<div class="progress mb-3">';
                    probHtml += '<div class="progress-bar bg-' + (idx === 0 ? 'warning' : idx === 1 ? 'success' : 'info') + '" role="progressbar" style="width: ' + (prob * 100) + '%"></div>';
                    probHtml += '</div>';
                });
                document.getElementById('probDist').innerHTML = probHtml;

                document.getElementById('gradcamImage').src = 'data:image/jpeg;base64,' + data.gradcam_image;

                document.getElementById('adviceTitle').textContent = data.advice.title;
                let tipsHtml = '';
                data.advice.tips.forEach(tip => {
                    tipsHtml += '<li>' + tip + '</li>';
                });
                document.getElementById('adviceTips').innerHTML = tipsHtml;

                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').style.display = 'block';

            } catch (error) {
                alert('Error uploading image: ' + error.message);
                document.getElementById('loading').style.display = 'none';
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image = Image.open(file.stream)

        image_tensor, processed_img = preprocess_image(image)
        skin_type, confidence, probs = predict(models, image_tensor)

        gradcam_img = generate_gradcam(models[0], image_tensor)
        gradcam_pil = Image.fromarray(gradcam_img)

        buf = io.BytesIO()
        gradcam_pil.save(buf, format='JPEG')
        gradcam_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        preview_buf = io.BytesIO()
        processed_img.save(preview_buf, format='JPEG')
        preview_base64 = base64.b64encode(preview_buf.getvalue()).decode('utf-8')

        return jsonify({
            'skin_type': skin_type,
            'ph_range': PH_MAPPING[skin_type],
            'confidence': confidence,
            'probabilities': probs,
            'gradcam_image': gradcam_base64,
            'preview_image': preview_base64,
            'advice': SKINCARE_ADVICE[skin_type]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print(f"Loaded {len(models)} models")
    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)