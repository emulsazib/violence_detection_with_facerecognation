# Model Files Setup Guide

This document explains how to download and setup the required model files for the Violence Detection system.

## 📥 Download Models from Google Drive

### Why are models not in the repository?

The model files are too large to be stored in Git (ranging from 5MB to 18MB each). They have been uploaded to Google Drive for easy access.

## Required Model Files

### 1. YOLO Models

| File Name | Size | Description | Required |
|-----------|------|-------------|----------|
| `yolo11n.pt` | 5.4 MB | YOLO11 Nano - Base object detection model | ✅ Yes |
| `yolo11n-pose.pt` | 6.0 MB | YOLO11 Nano Pose - Human pose estimation | Optional |
| `yolo11s.pt` | 18 MB | YOLO11 Small - Higher accuracy model | Optional |
| `best.pt` | 5.2 MB | **Custom Violence Detection Model** | ✅ **Required** |

### 2. Download Link

**📦 Download All Models (ZIP)**

🔗 **Google Drive Link**: [Download Models Here](https://drive.google.com/file/d/1EV1OQVgnA35doD6CBJgUXV3zQELPKsGc/view?usp=share_link)

> **Note**: Replace `YOUR_GOOGLE_DRIVE_SHARE_LINK` with your actual Google Drive shared link

### 3. Installation Steps

#### Step 1: Download the ZIP file
Click the Google Drive link above and download the `violence_detection_models.zip` file.

#### Step 2: Extract the ZIP file
```bash
# Extract to a temporary location
unzip violence_detection_models.zip -d /tmp/models
```

#### Step 3: Move files to the correct locations

```bash
# Navigate to your project directory
cd violence_detection_with_facerecognation

# Create necessary directories
mkdir -p models/dataset/violence_detection_run/weights

# Copy YOLO base models
cp /tmp/models/yolo11n.pt models/
cp /tmp/models/yolo11n-pose.pt models/
cp /tmp/models/yolo11s.pt models/

# Copy the custom violence detection model (MOST IMPORTANT)
cp /tmp/models/best.pt models/dataset/violence_detection_run/weights/

# Optional: Copy the last checkpoint if available
cp /tmp/models/last.pt models/dataset/violence_detection_run/weights/
```

#### Step 4: Verify Installation

```bash
# Check if all files are in place
ls -lh models/*.pt
ls -lh models/dataset/violence_detection_run/weights/*.pt
```

You should see:
```
models/yolo11n.pt              (5.4 MB)
models/yolo11n-pose.pt         (6.0 MB)
models/yolo11s.pt              (18 MB)
models/dataset/violence_detection_run/weights/best.pt  (5.2 MB)
```

## 📂 Expected Directory Structure

After setup, your `models/` directory should look like this:

```
models/
├── yolo11n.pt                    # YOLO11 Nano model
├── yolo11n-pose.pt               # YOLO11 Pose model
├── yolo11s.pt                    # YOLO11 Small model
└── dataset/
    ├── violence_dataset.yaml     # Dataset configuration (in repo)
    └── violence_detection_run/
        ├── args.yaml             # Training arguments (in repo)
        └── weights/
            ├── best.pt           # ⭐ Main violence detection model
            └── last.pt           # Last training checkpoint (optional)
```

## 🔄 Alternative Download Methods

### Method 1: Using gdown (Recommended for automation)

```bash
pip install gdown

# Download individual files
gdown "YOUR_GOOGLE_DRIVE_FILE_ID" -O models/yolo11n.pt
gdown "YOUR_GOOGLE_DRIVE_FILE_ID" -O models/yolo11s.pt
gdown "YOUR_GOOGLE_DRIVE_FILE_ID" -O models/yolo11n-pose.pt
gdown "YOUR_GOOGLE_DRIVE_FILE_ID" -O models/dataset/violence_detection_run/weights/best.pt
```

### Method 2: Using wget (if files are publicly accessible)

```bash
wget "YOUR_DIRECT_DOWNLOAD_LINK" -O models/yolo11n.pt
# Repeat for other files...
```

### Method 3: Manual Download from Browser

1. Go to the Google Drive link
2. Click on each file
3. Click "Download" button
4. Move downloaded files to the appropriate folders as shown above

## ✅ Verification

Run this Python script to verify all models are correctly installed:

```python
import os
from pathlib import Path

def verify_models():
    """Verify all required model files are present"""
    
    project_root = Path(__file__).resolve().parent
    
    required_models = {
        'yolo11n.pt': project_root / 'models' / 'yolo11n.pt',
        'yolo11s.pt': project_root / 'models' / 'yolo11s.pt',
        'best.pt': project_root / 'models' / 'dataset' / 'violence_detection_run' / 'weights' / 'best.pt'
    }
    
    optional_models = {
        'yolo11n-pose.pt': project_root / 'models' / 'yolo11n-pose.pt',
        'last.pt': project_root / 'models' / 'dataset' / 'violence_detection_run' / 'weights' / 'last.pt'
    }
    
    print("🔍 Checking required model files...\n")
    
    all_present = True
    for name, path in required_models.items():
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"✅ {name}: Found ({size:.1f} MB)")
        else:
            print(f"❌ {name}: NOT FOUND")
            print(f"   Expected at: {path}")
            all_present = False
    
    print("\n🔍 Checking optional model files...\n")
    for name, path in optional_models.items():
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print(f"✅ {name}: Found ({size:.1f} MB)")
        else:
            print(f"⚠️  {name}: Not found (optional)")
    
    print("\n" + "="*50)
    if all_present:
        print("✅ All required models are installed!")
        print("🚀 You're ready to run the application!")
    else:
        print("❌ Some required models are missing!")
        print("📥 Please download them from Google Drive")
        print("🔗 Link: [YOUR_GOOGLE_DRIVE_LINK]")
    print("="*50)
    
    return all_present

if __name__ == "__main__":
    verify_models()
```

Save this as `verify_models.py` in the project root and run:

```bash
python verify_models.py
```

## 🤔 Troubleshooting

### Issue: "Model file not found" error

**Solution**: 
1. Check if the model files are in the correct directories
2. Ensure file names match exactly (case-sensitive)
3. Run `verify_models.py` to check installation

### Issue: "Permission denied" when copying files

**Solution**:
```bash
sudo chmod -R 755 models/
```

### Issue: Google Drive download limit reached

**Solution**:
- Wait 24 hours for quota reset
- Or download from alternative sources (contact repository owner)

### Issue: Models are corrupted

**Solution**:
1. Re-download the files from Google Drive
2. Check file sizes match the expected sizes
3. Verify MD5 checksums if provided

## 📊 Model Information

### Violence Detection Model (`best.pt`)

- **Framework**: YOLOv11
- **Classes**: 
  - Violence (class 0)
  - Weapon (class 1)
  - Knife (class 2)
  - NonViolence (class 3)
- **Input Size**: 640x640
- **Confidence Threshold**: 0.5 (configurable)
- **Training Dataset**: Custom violence detection dataset

### Base YOLO Models

- **yolo11n.pt**: Fastest, lowest accuracy (good for real-time on CPU)
- **yolo11s.pt**: Balanced speed and accuracy
- **yolo11n-pose.pt**: Specialized for human pose estimation

## 🔐 Model Checksums (Optional)

You can verify file integrity using MD5 checksums:

```bash
md5sum models/yolo11n.pt
md5sum models/yolo11s.pt
md5sum models/dataset/violence_detection_run/weights/best.pt
```

Expected checksums:
```
# Add actual checksums here after uploading to Google Drive
# Example:
# a1b2c3d4e5f6... models/yolo11n.pt
# f6e5d4c3b2a1... models/yolo11s.pt
```

## 📧 Support

If you encounter any issues with model downloads or setup:

1. Check this document first
2. Open an issue on GitHub with details
3. Contact the repository maintainer

---

**Last Updated**: December 2024

