# Skin classifier

Web-based skin type classifier that analyzes face images to estimate skin type and pH level.

## Features

- **Skin Type Detection**: Classifies skin as Dry, Normal, or Oily
- **pH Estimation**: Provides pH range based on skin type
- **Grad-CAM Visualization**: Shows which regions the model focuses on
- **Skincare Recommendations**: Personalized tips for each skin type

## Tech Stack

- Flask (web framework)
- PyTorch + timm (model inference)
- MobileNetV2 (backbone)
- OpenCV (face detection)
- Grad-CAM (attention visualization)

## Setup

1. Install dependencies:
```bash
pip install flask torch timm pillow opencv-python grad-cam
```

2. Run the app:
```bash
python app_flask.py
```

3. Open http://127.0.0.1:5000 in your browser

## Model

- 5-fold ensemble of MobileNetV2 models trained on skin images
- Classes: Dry, Normal, Oily
- Face detection automatically crops to face region

## Note

This is for educational purposes only. Consult a dermatologist for professional skin analysis.
