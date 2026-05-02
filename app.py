import streamlit as st
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
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Skin pH Analyzer",
    page_icon="🧴",
    layout="centered"
)

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


def load_model():
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

    model = SkinClassifier(MODEL_NAME, num_classes=len(CLASS_NAMES), dropout=0.5)
    model.eval()

    loaded_models = []
    for fold in range(1, N_FOLDS + 1):
        model_path = os.path.join(MODEL_DIR, f"skink_model_fold{fold}.pth")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            loaded_models.append(model)

    return loaded_models


def detect_face(image):
    """Detect face and return cropped image"""
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
    """Preprocess image: detect face, resize, normalize"""
    cropped, face_detected = detect_face(image)

    display_img = cropped.copy()

    processed = TRANSFORM(cropped)
    return processed.unsqueeze(0), display_img


def predict(models, image_tensor):
    """Ensemble prediction using average probabilities"""
    all_probs = []

    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)

    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    pred_idx = torch.argmax(avg_probs, dim=1).item()
    confidence = avg_probs[0, pred_idx].item()

    return CLASS_NAMES[pred_idx], confidence, avg_probs[0].tolist()


def generate_gradcam(model, image_tensor):
    """Generate Grad-CAM visualization"""
    target_layer = model.backbone.blocks[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=image_tensor)

    original_image = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    visualization = show_cam_on_image(original_image, grayscale_cam[0], use_rgb=True)

    return visualization


def main():
    st.title("🧴 AI-Based Skin pH Estimator")
    st.markdown("""
    **Upload a face image to analyze your skin type and estimate pH level**
    
    *pH is estimated based on dermatological studies correlating skin type with pH ranges.*
    """)

    @st.cache_resource
    def get_models():
        return load_model()

    models = get_models()

    if not models:
        st.error("No models loaded! Please check the model directory.")
        return

    st.divider()

    st.subheader("📸 Upload Image")
    option = st.radio("Choose input method:", ["Upload from device", "Use camera"], horizontal=True)

    image = None
    if option == "Upload from device":
        uploaded_file = st.file_uploader(
            "Choose a face image...",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
    else:
        img_file = st.camera_input("Take a photo")
        if img_file:
            image = Image.open(img_file)

    if image:
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Your Image")
            st.image(image, use_container_width=True)

        with col2:
            with st.spinner("Analyzing skin..."):
                image_tensor, processed_img = preprocess_image(image)
                skin_type, confidence, probs = predict(models, image_tensor)
                ph_range = PH_MAPPING[skin_type]

            st.subheader("🔍 Analysis Results")

            st.success(f"**Skin Type:** {skin_type.capitalize()}")
            st.info(f"**Estimated pH:** {ph_range}")
            st.metric("Confidence", f"{confidence*100:.1f}%")

            st.subheader("📊 Probability Distribution")
            for cls, prob in zip(CLASS_NAMES, probs):
                st.progress(prob, text=f"{cls.capitalize()}: {prob*100:.1f}%")

        st.divider()

        st.subheader("🔬 Grad-CAM Attention Map")
        with st.spinner("Generating attention map..."):
            gradcam_img = generate_gradcam(models[0], image_tensor)
            st.image(gradcam_img, caption="Regions model focused on", use_container_width=True)

        st.divider()

        advice = SKINCARE_ADVICE[skin_type]
        st.subheader("💡 Skincare Recommendations")
        st.markdown(f"**{advice['title']}**")

        for tip in advice['tips']:
            st.markdown(f"- {tip}")

        st.divider()
        st.caption("*Note: This is an AI-based estimation for educational purposes only. Consult a dermatologist for professional skin analysis.*")


if __name__ == "__main__":
    main()