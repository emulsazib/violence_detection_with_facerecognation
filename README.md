# Violence Detection with Face Recognition 🚨

A Django-based real-time violence detection system using YOLO11 and face recognition capabilities. This system can detect violent incidents, weapons, and knives in video streams and identify individuals involved using facial recognition.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Django](https://img.shields.io/badge/django-5.1+-green.svg)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Real-time Violence Detection**: Detects violence, weapons, and knives using custom-trained YOLO11 model
- **Face Recognition**: Identifies individuals in violent incidents using DeepFace
- **Live Camera Integration**: Works with ESP32-CAM or any IP camera stream
- **Alert System**: Immediate alerts when violence is detected
- **Image Capture**: Automatically captures and stores frames when violence is detected
- **Web Dashboard**: Real-time monitoring dashboard built with Django
- **Logging System**: Comprehensive logging for debugging and monitoring
- **Multiple Model Support**: Support for various face recognition models (VGG-Face, FaceNet, OpenFace, etc.)

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Model Files Setup](#model-files-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 💻 System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for better performance)
- CUDA-capable GPU (optional, but recommended for real-time processing)
- Webcam or IP camera for live detection

### Supported Operating Systems
- Linux (Ubuntu 20.04+)
- macOS (10.15+)
- Windows 10/11

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/violence_detection_with_facerecognation.git
cd violence_detection_with_facerecognation
```

### 2. Create Virtual Environment

```bash
# Using venv
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Or using conda
conda create -n violence_detection python=3.10
conda activate violence_detection
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Model Files

**IMPORTANT**: Model files are too large for Git and must be downloaded separately from Google Drive.

#### Required Model Files:

1. **YOLO Models** (Download from Google Drive):
   - 📥 [Download All Models (ZIP)](YOUR_GOOGLE_DRIVE_LINK_HERE)
   
   Or download individually:
   - `yolo11n.pt` (5.4 MB) - YOLO11 Nano model
   - `yolo11n-pose.pt` (6.0 MB) - YOLO11 Nano Pose model
   - `yolo11s.pt` (18 MB) - YOLO11 Small model
   - `best.pt` (5.2 MB) - **Custom trained violence detection model** (Required)

2. **Place Model Files**:
   ```bash
   # Create models directory if it doesn't exist
   mkdir -p models/dataset/violence_detection_run/weights
   
   # Move downloaded models to the correct locations:
   # - Place yolo11n.pt, yolo11n-pose.pt, yolo11s.pt in: models/
   # - Place best.pt in: models/dataset/violence_detection_run/weights/
   ```

#### Directory Structure After Model Setup:
```
models/
├── yolo11n.pt
├── yolo11n-pose.pt
├── yolo11s.pt
└── dataset/
    ├── violence_dataset.yaml
    └── violence_detection_run/
        └── weights/
            └── best.pt  # Main violence detection model
```

### 5. Create Required Directories

```bash
# Create necessary directories
mkdir -p logs
mkdir -p media/captured_db
mkdir -p media/face_db/{sazib,shawon,rahat}
mkdir -p media/results
```

### 6. Database Setup

```bash
# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional, for admin access)
python manage.py createsuperuser
```

### 7. Setup Face Recognition Database

Place reference images for face recognition in the `media/face_db/` directory:

```
media/face_db/
├── person1/
│   └── person1.png
├── person2/
│   └── person2.png
└── person3/
    └── person3.png
```

See [FACE_RECOGNITION_SETUP.md](FACE_RECOGNITION_SETUP.md) for detailed instructions.

## ⚙️ Configuration

### Camera Configuration

Edit `detection_engine/yolo_detection.py` to configure your camera:

```python
self.camera_stream_url = 'http://YOUR_CAMERA_IP/cam-hi.jpg'
```

### Model Configuration

To use a different YOLO model, update the model path in `detection_engine/yolo_detection.py`:

```python
self.model_path = os.path.join(project_root, "models", "your_model.pt")
```

### Django Settings

Update `violence_detection/settings.py` for production:

```python
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'your-ip-address']
SECRET_KEY = 'your-secret-key-here'  # Generate a new secret key
```

## 🎯 Usage

### Start the Development Server

```bash
python manage.py runserver
```

The application will be available at `http://localhost:8000/`

### Start Detection

1. Navigate to the dashboard: `http://localhost:8000/`
2. The system will automatically start monitoring the camera feed
3. Violence alerts will appear in real-time when detected

### API Usage

#### Get Latest Frame
```bash
curl http://localhost:8000/api/get_frame/
```

#### Get Violence Status
```bash
curl http://localhost:8000/api/violence_status/
```

#### Trigger Face Recognition
```bash
curl -X POST http://localhost:8000/api/recognize_face/
```

## 📁 Project Structure

```
violence_detection_with_facerecognation/
├── detection_engine/          # Core detection logic
│   ├── yolo_detection.py     # YOLO violence detection
│   └── face_recognation.py   # Face recognition system
├── web/                      # Web app views and URLs
├── templates/                # HTML templates
│   └── web/
│       └── dashboard.html    # Main dashboard
├── static/                   # Static files (CSS, JS)
│   ├── js/
│   └── stylesheet/
├── media/                    # Media files (gitignored)
│   ├── captured_db/         # Captured violence frames
│   ├── face_db/             # Face recognition database
│   └── results/             # Recognition results
├── models/                   # ML models (gitignored - download separately)
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   └── dataset/
│       └── violence_detection_run/
│           └── weights/
│               └── best.pt
├── logs/                     # Application logs (gitignored)
├── violence_detection/       # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔌 API Endpoints

### Detection Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/get_frame/` | GET | Get current camera frame with detections |
| `/api/violence_status/` | GET | Get current violence detection status |
| `/api/violence_logs/` | GET | Get violence detection logs |
| `/api/clear_violence_logs/` | POST | Clear all violence logs |

### Face Recognition Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recognize_face/` | POST | Trigger face recognition on latest frame |
| `/api/face_recognition_stats/` | GET | Get face recognition statistics |
| `/api/latest_recognition/` | GET | Get latest recognition result |

## 🔧 Troubleshooting

### Common Issues

#### 1. Model File Not Found
```
Error: Model file not found: models/dataset/violence_detection_run/weights/best.pt
```
**Solution**: Download the model files from Google Drive and place them in the correct directories (see [Model Files Setup](#model-files-setup))

#### 2. Camera Connection Failed
```
Error: Failed to get frame from camera
```
**Solution**: 
- Check camera IP address in `detection_engine/yolo_detection.py`
- Ensure camera is accessible on your network
- Test camera URL in browser: `http://YOUR_CAMERA_IP/cam-hi.jpg`

#### 3. DeepFace Installation Issues
```
ImportError: DeepFace not available
```
**Solution**:
```bash
pip install deepface
pip install tf-keras
```

#### 4. CUDA/GPU Issues
```
Warning: CUDA not available, using CPU
```
**Solution**: Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 5. Permission Denied for Logs Directory
```
PermissionError: [Errno 13] Permission denied: 'logs/violence_detection.log'
```
**Solution**:
```bash
mkdir -p logs
chmod 755 logs
```

### Performance Optimization

1. **Use GPU**: Install CUDA and PyTorch with GPU support
2. **Reduce Frame Rate**: Adjust camera frame rate for slower systems
3. **Lower Confidence Threshold**: Adjust in `yolo_detection.py` for fewer false positives
4. **Disable Face Recognition**: Set `face_recognition_enabled = False` if not needed

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work - [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLOv11 implementation
- [DeepFace](https://github.com/serengil/deepface) - Face recognition library
- [Django](https://www.djangoproject.com/) - Web framework

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com)

---

**Note**: This system is designed for security and safety applications. Please ensure you comply with local laws and regulations regarding surveillance and privacy when deploying this system.

