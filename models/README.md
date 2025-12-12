# Models Directory

This directory should contain the YOLO model files required for violence detection.

## ⚠️ Important: Model Files Not Included in Git

Model files are **NOT** included in this repository due to their large size. You must download them separately from Google Drive.

## 📥 Download Instructions

**Please see [MODELS_README.md](../MODELS_README.md) in the project root for detailed download and installation instructions.**

Quick link: 🔗 [Download Models from Google Drive](YOUR_GOOGLE_DRIVE_LINK)

## 📂 Required Directory Structure

After downloading and placing the model files, this directory should have the following structure:

```
models/
├── yolo11n.pt                          # YOLO11 Nano model (5.4 MB)
├── yolo11n-pose.pt                     # YOLO11 Pose model (6.0 MB) [Optional]
├── yolo11s.pt                          # YOLO11 Small model (18 MB)
├── README.md                           # This file
└── dataset/
    ├── violence_dataset.yaml           # Dataset configuration
    └── violence_detection_run/
        ├── args.yaml                   # Training arguments
        └── weights/
            ├── best.pt                 # ⭐ Main violence detection model (5.2 MB)
            └── last.pt                 # Last training checkpoint (5.2 MB) [Optional]
```

## ✅ Verify Installation

After downloading and placing the model files, run the verification script:

```bash
python verify_models.py
```

This will check if all required models are in the correct locations.

## 🔍 Model Information

### Violence Detection Model (`best.pt`)

The main violence detection model is located at:
```
models/dataset/violence_detection_run/weights/best.pt
```

**Model Details:**
- Framework: YOLOv11
- Task: Object Detection
- Classes: Violence, Weapon, Knife, NonViolence
- Input Size: 640x640
- Format: PyTorch (.pt)

### Base YOLO Models

These are pre-trained YOLO models used as base models:

- **yolo11n.pt**: Nano model - Fastest inference, good for CPU
- **yolo11s.pt**: Small model - Better accuracy, requires GPU for real-time
- **yolo11n-pose.pt**: Pose estimation model (Optional)

## 🚀 Model Usage in Code

The violence detection model is automatically loaded in the code:

```python
from detection_engine.yolo_detection import YOLODetector

# Initialize detector (automatically loads best.pt)
detector = YOLODetector()

# Or specify a custom model path
detector = YOLODetector(model_path="models/yolo11s.pt")
```

## 🔧 Training Your Own Model

If you want to train your own violence detection model:

1. Prepare your dataset in YOLO format
2. Update `violence_dataset.yaml` with your dataset paths
3. Use the Ultralytics YOLO training script:

```python
from ultralytics import YOLO

# Load a base model
model = YOLO('yolo11n.pt')

# Train on your dataset
model.train(
    data='models/dataset/violence_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='violence_detection_run'
)
```

The trained model will be saved in `runs/detect/violence_detection_run/weights/best.pt`

## 📊 Model Performance

Expected performance metrics for the violence detection model:

- mAP50: ~0.85-0.90
- Precision: ~0.80-0.85
- Recall: ~0.75-0.80
- Inference Time: ~50-100ms per frame (on GPU)

## ❓ Troubleshooting

### Model File Not Found Error

```
FileNotFoundError: Model file not found: models/dataset/violence_detection_run/weights/best.pt
```

**Solution**: Download the model files from Google Drive (see [MODELS_README.md](../MODELS_README.md))

### Model Loading Errors

```
RuntimeError: Error loading model
```

**Solutions**:
1. Verify the model file is not corrupted (re-download if needed)
2. Check PyTorch and Ultralytics versions match requirements
3. Ensure sufficient memory is available

### CUDA/GPU Issues

If you get CUDA-related errors:
```bash
# Reinstall PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Notes

- Model files are tracked in `.gitignore` to prevent accidental commits
- Always keep backups of your trained models
- Consider using model versioning for production deployments

## 🆘 Need Help?

- Check [MODELS_README.md](../MODELS_README.md) for download instructions
- See main [README.md](../README.md) for general setup
- Open an issue on GitHub for support

