# Quick Start Guide

Get up and running with the Violence Detection System in 5 minutes!

## 🚀 Quick Setup (For Experienced Users)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/violence_detection_with_facerecognation.git
cd violence_detection_with_facerecognation

# 2. Run the automated setup
./setup.sh

# 3. Download models from Google Drive (see MODELS_README.md)
# Place them in models/ directory

# 4. Run verification
python verify_models.py

# 5. Start the server
python manage.py runserver
```

Open `http://localhost:8000` in your browser!

## 📖 Detailed Setup (Step by Step)

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum
- Git installed

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/violence_detection_with_facerecognation.git
cd violence_detection_with_facerecognation
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

**On Windows:**
```cmd
python -m venv env
env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Model Files

**IMPORTANT:** Model files must be downloaded separately!

1. Go to: [Google Drive Models Link](YOUR_GOOGLE_DRIVE_LINK)
2. Download `violence_detection_models.zip`
3. Extract and place files in the `models/` directory

Required files:
- `models/yolo11n.pt`
- `models/yolo11s.pt`
- `models/dataset/violence_detection_run/weights/best.pt`

See [MODELS_README.md](MODELS_README.md) for detailed instructions.

### Step 5: Create Directories

```bash
mkdir -p logs media/captured_db media/face_db media/results
```

### Step 6: Setup Database

```bash
python manage.py migrate
```

### Step 7: Verify Installation

```bash
python verify_models.py
```

You should see all green checkmarks ✅

### Step 8: Configure Camera (Optional)

Edit `detection_engine/yolo_detection.py`:

```python
self.camera_stream_url = 'http://YOUR_CAMERA_IP/cam-hi.jpg'
```

### Step 9: Start the Server

```bash
python manage.py runserver
```

### Step 10: Access Dashboard

Open your browser and go to:
```
http://localhost:8000
```

## 🎯 Testing Without Camera

If you don't have a camera, the system will use test frames. You can still test face recognition and violence detection features.

## 📱 ESP32-CAM Setup (Optional)

If you want to use ESP32-CAM:

1. Flash ESP32-CAM with camera streaming firmware
2. Connect ESP32-CAM to your WiFi
3. Note the IP address
4. Update camera URL in the code:
   ```python
   self.camera_stream_url = 'http://ESP32_IP/cam-hi.jpg'
   ```

## 🔧 Common Issues

### Issue: "Model not found"

```bash
# Download models from Google Drive
# See MODELS_README.md
```

### Issue: "Module not found"

```bash
# Activate virtual environment
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "Camera connection failed"

```bash
# Check camera URL
# Test in browser: http://YOUR_CAMERA_IP/cam-hi.jpg
```

### Issue: "Port already in use"

```bash
# Use a different port
python manage.py runserver 8080
```

## 🎨 Using the Dashboard

Once running, the dashboard shows:

- **Live Feed**: Real-time camera feed with detections
- **Violence Alerts**: Notifications when violence is detected
- **Captured Images**: Automatically saved frames
- **Detection Stats**: Real-time statistics

## 🧪 Testing Face Recognition

1. Add reference images to `media/face_db/`:
   ```bash
   mkdir media/face_db/john
   # Add john's photo to media/face_db/john/
   ```

2. Trigger face recognition from the dashboard

3. View results in the logs

## 📚 Next Steps

- Read the full [README.md](README.md)
- Configure advanced settings
- Add more people to face recognition database
- Customize detection thresholds
- Set up production deployment

## 🆘 Need Help?

- Check [README.md](README.md) for detailed documentation
- See [MODELS_README.md](MODELS_README.md) for model setup
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Open an issue on GitHub

## 🎉 You're All Set!

Enjoy using the Violence Detection System!

---

**Quick Links:**
- [Full Documentation](README.md)
- [Model Setup](MODELS_README.md)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)

