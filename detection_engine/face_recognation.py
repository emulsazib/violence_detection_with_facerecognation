import os
import shutil
import json
import cv2
import numpy as np
import time
import logging
from datetime import datetime
import pickle
from typing import Dict, Any, List, Optional
from django.conf import settings
from django.utils import timezone
from .yolo_detection import YOLODetector

# Set up logging
face_logger = logging.getLogger('face_recognition')
debug_logger = logging.getLogger('detection_engine')

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    face_logger.warning(f"DeepFace not available due to: {e}")
    DEEPFACE_AVAILABLE = False

# Try OpenCV face recognition as fallback
try:
    import cv2
    import numpy as np
    # Load OpenCV face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    OPENCV_FACE_AVAILABLE = True
    face_logger.info("OpenCV face detection available as fallback")
except Exception as e:
    face_logger.warning(f"OpenCV face detection not available: {e}")
    OPENCV_FACE_AVAILABLE = False


class FaceRecognation:
    """
    Django-integrated Face Recognition System
    Integrates with YOLO violence detection to identify people in violent incidents
    """

    def __init__(self):
        """Initialize the Django Face Recognition System"""
        if not DEEPFACE_AVAILABLE:
            print("DeepFace is not available. Face recognition will be limited.")
            # Don't raise error, just set a flag
            self.deepface_available = False
        else:
            self.deepface_available = True

        # Django project paths
        self.project_root = settings.BASE_DIR
        self.db_dir = os.path.join(self.project_root, 'media', 'face_db')
        self.captured_images_dir = os.path.join(self.project_root, 'media', 'captured_db')
        self.results_dir = os.path.join(self.project_root, 'media', 'results')
        # Configuration for face recognition - Optimized for speed with FaceNet
        self.config = {
            "primary_model": "Facenet512",  # Primary fast model for instant recognition
            "fallback_models": ["Facenet512", "OpenFace"],  # Backup models if needed
            "all_models": ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace"],  # Full model list for training
            "distance_metrics": ["cosine", "euclidean"],  # Reduced for speed
            "detector_backends": ["opencv"],  # Use fastest detector
            "confidence_threshold": 0.75,  # Balanced threshold for speed vs accuracy
            "strict_confidence_threshold": 0.85,  # Higher threshold for very confident matches
            "max_distance_threshold": 0.5,  # Tighter distance for faster processing
            "consensus_threshold": 1,  # Single model for speed (FaceNet is reliable)
            "max_matches": 5,  # Reduced for faster processing
            "instant_mode": True,  # Enable instant recognition mode
            "quality_thresholds": {
                "excellent": 0.95,
                "good": 0.85,
                "fair": 0.75
            },
            "face_preprocessing": {
                "enabled": True,
                "resize_dimension": 160,  # Smaller for faster processing
                "normalization": True,
                "histogram_equalization": False,  # Disabled for speed
                "gaussian_blur": False
            }
        }

        # People database configuration
        self.people_config = {
            "sazib": ["sazib.png"],
            "shawon": ["shawon.png"],
            "rahat": ["rahat.png"]
        }

        # Milvus configuration for vector database
        self.milvus_config = {
            "host": "localhost",
            "port": "19530",
            "collection_name": "face_embeddings",
            "dimension": 512,  # Face embedding dimension (will be determined by model)
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "nlist": 128,
            "nprobe": 10
        }

        # Model cache tracking
        self.model_cache_files = []
        self.last_cache_check = 0

        # Recognition statistics
        self.recognition_stats = {
            "total_recognitions": 0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
            "last_recognition_time": None
        }

        # Initialize directories and database
        self._initialize_directories()
        self._setup_database()

        print("Django Face Recognition System initialized successfully")

    def _preprocess_face_image(self, image_path):
        """
        Preprocess face image to improve recognition accuracy
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Path to preprocessed image
        """
        try:
            if not self.config["face_preprocessing"]["enabled"]:
                return image_path
                
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return image_path
                
            # Resize to standard dimension
            target_size = self.config["face_preprocessing"]["resize_dimension"]
            img = cv2.resize(img, (target_size, target_size))
            
            # Convert to grayscale for better face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Histogram equalization for better contrast
            if self.config["face_preprocessing"]["histogram_equalization"]:
                gray = cv2.equalizeHist(gray)
            
            # Convert back to BGR
            processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Normalize pixel values
            if self.config["face_preprocessing"]["normalization"]:
                processed_img = processed_img.astype(np.float32) / 255.0
                processed_img = (processed_img * 255).astype(np.uint8)
            
            # Save preprocessed image
            preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg').replace('.png', '_preprocessed.png')
            cv2.imwrite(preprocessed_path, processed_img)
            
            print(f"Preprocessed image saved: {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return image_path

    def _preprocess_face_image_fast(self, image_path):
        """
        Fast preprocessing for immediate recognition - minimal processing for speed
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Path to preprocessed image (or original if preprocessing disabled)
        """
        try:
            if not self.config["face_preprocessing"]["enabled"]:
                return image_path
                
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return image_path
                
            # Minimal processing for speed - just resize
            target_size = self.config["face_preprocessing"]["resize_dimension"]
            img = cv2.resize(img, (target_size, target_size))
            
            # Simple normalization only
            if self.config["face_preprocessing"]["normalization"]:
                img = img.astype(np.float32) / 255.0
                img = (img * 255).astype(np.uint8)
            
            # Save preprocessed image
            preprocessed_path = image_path.replace('.jpg', '_fast_preprocessed.jpg').replace('.png', '_fast_preprocessed.png')
            cv2.imwrite(preprocessed_path, img)
            
            print(f"Fast preprocessed image saved: {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            print(f"Error in fast preprocessing {image_path}: {e}")
            return image_path

    def _create_multiple_reference_images(self, person_name, base_image_path):
        """
        Create multiple reference images from a base image to improve recognition accuracy
        
        Args:
            person_name: Name of the person
            base_image_path: Path to the base reference image
            
        Returns:
            list: List of created reference image paths
        """
        try:
            if not os.path.exists(base_image_path):
                print(f"Base image not found: {base_image_path}")
                return []
                
            # Read base image
            img = cv2.imread(base_image_path)
            if img is None:
                print(f"Could not read base image: {base_image_path}")
                return []
                
            person_dir = os.path.join(self.db_dir, person_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                
            reference_images = []
            
            # Original image
            original_path = os.path.join(person_dir, f"{person_name}_original.png")
            cv2.imwrite(original_path, img)
            reference_images.append(original_path)
            
            # Create variations for better recognition
            variations = [
                ("resized_224", lambda x: cv2.resize(x, (224, 224))),
                ("resized_512", lambda x: cv2.resize(x, (512, 512))),
                ("grayscale", lambda x: cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
                ("histogram_eq", lambda x: cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)),
                ("brightness_plus", lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=30)),
                ("brightness_minus", lambda x: cv2.convertScaleAbs(x, alpha=0.8, beta=-30)),
                ("contrast_plus", lambda x: cv2.convertScaleAbs(x, alpha=1.3, beta=0)),
                ("contrast_minus", lambda x: cv2.convertScaleAbs(x, alpha=0.7, beta=0))
            ]
            
            for variation_name, transform_func in variations:
                try:
                    transformed_img = transform_func(img.copy())
                    variation_path = os.path.join(person_dir, f"{person_name}_{variation_name}.png")
                    cv2.imwrite(variation_path, transformed_img)
                    reference_images.append(variation_path)
                    print(f"Created variation: {variation_path}")
                except Exception as e:
                    print(f"Failed to create variation {variation_name}: {e}")
                    
            print(f"Created {len(reference_images)} reference images for {person_name}")
            return reference_images
            
        except Exception as e:
            print(f"Error creating reference images for {person_name}: {e}")
            return []

    def rebuild_face_database(self):
        
        try:
            print("Starting face database rebuild...")
            
            # Clear existing cache
            self.clear_model_cache()
            
            # Get base images for each person
            base_images = {
                "sazib": os.path.join(self.db_dir, "sazib", "sazib.png"),
                "shawon": os.path.join(self.db_dir, "shawon", "shawon.png"),
                "rahat": os.path.join(self.db_dir, "rahat", "rahat.png")
            }
            
            rebuild_results = {}
            
            for person_name, base_image_path in base_images.items():
                if os.path.exists(base_image_path):
                    print(f"Processing {person_name}...")
                    
                    # Create multiple reference images
                    reference_images = self._create_multiple_reference_images(person_name, base_image_path)
                    
                    if reference_images:
                        rebuild_results[person_name] = {
                            "status": "success",
                            "reference_images_created": len(reference_images),
                            "image_paths": reference_images
                        }
                        print(f"Successfully created {len(reference_images)} reference images for {person_name}")
                    else:
                        rebuild_results[person_name] = {
                            "status": "failed",
                            "error": "No reference images created"
                        }
                        print(f"Failed to create reference images for {person_name}")
                else:
                    rebuild_results[person_name] = {
                        "status": "skipped",
                        "error": "Base image not found"
                    }
                    print(f"Base image not found for {person_name}: {base_image_path}")
            
            # Update metadata
            self._update_database_metadata(rebuild_results)
            
            print("Face database rebuild completed")
            return {
                "success": True,
                "message": "Face database rebuilt successfully",
                "results": rebuild_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error rebuilding face database: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _update_database_metadata(self, rebuild_results):
        """Update database metadata after rebuild"""
        try:
            metadata_path = os.path.join(self.db_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    "database_info": {
                        "version": "1.0",
                        "created_date": "2025-08-21",
                        "description": "Face recognition database for violence detection system",
                        "total_people": 3,
                        "embedding_model": "deepface",
                        "vector_dimension": 512
                    },
                    "people": {}
                }
            
            # Update metadata with rebuild results
            for person_name, result in rebuild_results.items():
                if person_name not in metadata["people"]:
                    metadata["people"][person_name] = {
                        "id": len(metadata["people"]) + 1,
                        "name": person_name.capitalize(),
                        "full_name": f"{person_name.capitalize()} User",
                        "role": f"Person {len(metadata['people']) + 1}",
                        "images": [],
                        "created_date": "2025-08-21",
                        "status": "active",
                        "notes": "Reference person for violence detection system"
                    }
                
                if result["status"] == "success":
                    metadata["people"][person_name]["images"] = [os.path.basename(path) for path in result["image_paths"]]
                    metadata["people"][person_name]["last_updated"] = datetime.now().isoformat()
                    metadata["people"][person_name]["reference_image_count"] = result["reference_images_created"]
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print("Database metadata updated successfully")
            
        except Exception as e:
            print(f"Error updating database metadata: {e}")

    def _initialize_directories(self):
        """Create necessary directories for the face recognition system"""
        directories = [
            self.db_dir,
            self.captured_images_dir,
            self.results_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")

    def _setup_database(self):
        """Setup face database directory structure"""
        print("Setting up face database...")

        # Create directory structure for each person
        for person, images in self.people_config.items():
            person_dir = os.path.join(self.db_dir, person)
            os.makedirs(person_dir, exist_ok=True)

            # Check if reference images exist (they should be manually placed)
            for img in images:
                img_path = os.path.join(person_dir, img)
                if not os.path.exists(img_path):
                    print(f"Reference image not found: {img_path}")
                    print(f"Please place {img} in {person_dir} for {person}")
                else:
                    print(f"Found reference image: {img_path}")

    def verify_database_structure(self):
        """Verify and return database structure information"""
        database_info = {
            "total_people": 0,
            "total_images": 0,
            "people": {},
            "missing_images": []
        }

        for person, expected_images in self.people_config.items():
            person_dir = os.path.join(self.db_dir, person)
            person_info = {
                "expected_images": expected_images,
                "found_images": [],
                "missing_images": []
            }

            if os.path.exists(person_dir):
                # Find all image files in person directory
                for file in os.listdir(person_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        person_info["found_images"].append(file)

                # Check for missing expected images
                for expected_img in expected_images:
                    img_path = os.path.join(person_dir, expected_img)
                    if not os.path.exists(img_path):
                        person_info["missing_images"].append(expected_img)
                        database_info["missing_images"].append(f"{person}/{expected_img}")
            else:
                person_info["missing_images"] = expected_images
                database_info["missing_images"].extend([f"{person}/{img}" for img in expected_images])

            database_info["people"][person] = person_info
            database_info["total_images"] += len(person_info["found_images"])

        database_info["total_people"] = len([p for p in database_info["people"].values() if p["found_images"]])

        return database_info

    def check_model_cache(self):
        """Check existing model cache files"""
        cache_files = []

        if os.path.exists(self.db_dir):
            for f in os.listdir(self.db_dir):
                if f.startswith("ds_model_") and f.endswith(".pkl"):
                    cache_file = os.path.join(self.db_dir, f)
                    cache_files.append(cache_file)

        self.model_cache_files = cache_files
        self.last_cache_check = time.time()

        print(f"Found {len(cache_files)} model cache files")
        return len(cache_files) > 0

    def validate_model_cache(self):
        """Validate if existing model cache is compatible with current database"""
        if not self.model_cache_files:
            return False

        try:
            # Get the newest cache file modification time
            newest_cache_time = max(os.path.getmtime(f) for f in self.model_cache_files)

            # Check if any database image is newer than cache
            for root, dirs, files in os.walk(self.db_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(root, file)
                        if os.path.getmtime(img_path) > newest_cache_time:
                            print(f"Database image {file} is newer than cache - cache invalid")
                            return False

            print("Model cache is valid and up-to-date")
            return True

        except Exception as e:
            print(f"Error validating cache: {e}")
            return False

    def clear_model_cache(self):
        """Clear existing model cache to force rebuild"""
        removed_count = 0

        if os.path.exists(self.db_dir):
            for f in os.listdir(self.db_dir):
                if f.startswith("ds_model_") and f.endswith(".pkl"):
                    cache_file = os.path.join(self.db_dir, f)
                    try:
                        os.remove(cache_file)
                        removed_count += 1
                        print(f"Removed cache file: {f}")
                    except Exception as e:
                        print(f"Failed to remove {f}: {e}")

        self.model_cache_files = []
        print(f"Cleared {removed_count} model cache files")
        return removed_count

    def get_latest_test_image(self):
        """
        Get the path to the most recently captured test image

        Returns:
            str: Path to latest test image or None if no images found
        """
        try:
            if not os.path.exists(self.captured_images_dir):
                return None

            # Get all image files in test_images directory
            image_files = []
            for file in os.listdir(self.captured_images_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    file_path = os.path.join(self.captured_images_dir, file)
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    image_files.append((file_path, mtime))

            if not image_files:
                return None

            # Sort by modification time (newest first)
            image_files.sort(key=lambda x: x[1], reverse=True)
            latest_image = image_files[0][0]

            print(f"Latest test image: {os.path.basename(latest_image)}")
            return latest_image

        except Exception as e:
            print(f"Error getting latest test image: {e}")
            return None

    def recognize_latest_captured_image(self):
        """
        Recognize faces in the most recently captured test image

        Returns:
            dict: Recognition results in JSON format
        """
        try:
            # Get the latest captured image
            latest_image_path = self.get_latest_test_image()

            if not latest_image_path:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "error": "No captured images found",
                    "recognition_results": {"success": False}
                }

            # Perform face recognition on the latest image
            best_results, all_results = self._recognize_face_enhanced(latest_image_path)

            # Format results
            recognition_result = self._format_results_to_json(
                latest_image_path, best_results, all_results, is_violence_frame=True
            )

            # Update statistics
            self.recognition_stats["total_recognitions"] += 1
            if recognition_result["recognition_results"]["success"]:
                self.recognition_stats["successful_recognitions"] += 1
            else:
                self.recognition_stats["failed_recognitions"] += 1
            self.recognition_stats["last_recognition_time"] = timezone.now()

            # Save results
            self._save_recognition_result(recognition_result)

            print(f"Face recognition completed on latest image: {os.path.basename(latest_image_path)}")
            return recognition_result

        except Exception as e:
            print(f"Error in latest image face recognition: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "recognition_results": {"success": False}
            }

    def recognize_face_in_frame(self, frame, save_frame=True):
        """
        Recognize faces in a given frame (from violence detection)

        Args:
            frame: OpenCV frame/image array
            save_frame: Whether to save the frame to captured_images directory

        Returns:
            dict: Recognition results in JSON format
        """
        try:
            # Create temporary frame for recognition without saving if not requested
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # Proper date/time format
            frame_filename = f"face_recognition_frame_{timestamp}.jpg"
            frame_path = os.path.join(self.captured_images_dir, frame_filename)

            # Always save frame temporarily for processing
            cv2.imwrite(frame_path, frame)
            if save_frame:
                print(f"Saved face recognition frame: {frame_path}")
            else:
                print(f"Created temporary frame for recognition: {frame_filename}")

            # Perform face recognition
            if DEEPFACE_AVAILABLE:
                best_results, all_results = self._recognize_face_enhanced(frame_path)
            elif OPENCV_FACE_AVAILABLE:
                # Use OpenCV face detection as fallback
                best_results, all_results = self._recognize_face_opencv(frame)
            else:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "error": "No face recognition libraries available",
                    "recognition_results": {"success": False, "no_libraries": True}
                }

            # Format results
            recognition_result = self._format_results_to_json(
                frame_path, 
                best_results, 
                all_results, 
                is_violence_frame=True,
            )

            # Update statistics
            self.recognition_stats["total_recognitions"] += 1
            if recognition_result["recognition_results"]["success"]:
                self.recognition_stats["successful_recognitions"] += 1
            else:
                self.recognition_stats["failed_recognitions"] += 1
            self.recognition_stats["last_recognition_time"] = timezone.now()

            # Save results
            self._save_recognition_result(recognition_result)

            # Delete temporary frame if not requested to save
            if not save_frame:
                try:
                    os.remove(frame_path)
                    print(f"Deleted temporary recognition frame: {frame_filename}")
                except Exception as e:
                    print(f"Failed to delete temporary frame: {e}")

            return recognition_result

        except Exception as e:
            print(f"Error in face recognition: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "recognition_results": {"success": False}
            }

    def recognize_live_camera_frame(self, camera_url=None):
        """
        Capture a frame from live ESP32 camera and perform face recognition
        
        Args:
            camera_url: Camera stream URL (uses YOLO detector's camera if not provided)
            
        Returns:
            dict: Recognition results in JSON format
        """
        # Check if any face recognition is available
        if not DEEPFACE_AVAILABLE and not OPENCV_FACE_AVAILABLE:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": "Face recognition not available - No face detection libraries working",
                "recognition_results": {"success": False, "no_face_detection": True}
            }
        
        try:
            
            if camera_url is None:
                # Try to get camera URL from existing YOLO detector instance or create one
                try:
                    # Create a temporary YOLO detector to get camera frame
                    temp_detector = YOLODetector()
                    camera_url = temp_detector.camera_stream_url
                except Exception as e:
                    print(f"Failed to get camera URL from YOLO detector: {e}")
                    camera_url = YOLODetector.camera_stream_url  # Default fallback
            
            # Get current frame from ESP32 camera
            frame = self._get_frame_from_camera(camera_url)
            if frame is None:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "error": "Failed to capture frame from ESP32 camera",
                    "recognition_results": {"success": False, "no_camera_access": True}
                }
            
            # Check if this is a test/fallback frame (has "NO CAMERA DETECTED" text)
            # Convert to grayscale and check for text patterns that indicate a test frame
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Test frame is mostly dark with white text
                mean_brightness = np.mean(gray)
                std_brightness = np.std(gray)
                
                # Test frames have low mean brightness but high standard deviation (due to white text)
                if mean_brightness < 100 and std_brightness > 50:
                    print("Detected test/fallback frame - no camera available")
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "error": "Camera not available - using test frame",
                        "recognition_results": {"success": False, "no_camera_access": True}
                    }
            except Exception as e:
                print(f"Failed to analyze frame type: {e}")
            
            print("Captured live frame from ESP32 camera for face recognition")
            
            # Perform face recognition on live frame (don't save it permanently)
            return self.recognize_face_in_frame(frame, save_frame=False)
            
        except Exception as e:
            print(f"Error in live camera face recognition: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "recognition_results": {"success": False}
            }

    def _get_frame_from_camera(self, camera_url, timeout=10, max_retries=3):
        """
        Get a frame from the ESP32 camera stream (similar to YOLO detector)
        
        Args:
            camera_url: Camera stream URL
            timeout: Connection timeout in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            OpenCV frame or None if failed
        """
        import urllib.request
        
        for attempt in range(max_retries):
            try:
                # Create request with timeout
                request = urllib.request.Request(camera_url)
                request.add_header('User-Agent', 'FaceRecognitionSystem/1.0')
                request.add_header('Accept', 'image/jpeg,image/*;q=0.9,*/*;q=0.8')
                request.add_header('Cache-Control', 'no-cache')

                # Open with timeout
                img_resp = urllib.request.urlopen(request, timeout=timeout)

                # Read and decode image
                image_data = img_resp.read()
                if len(image_data) == 0:
                    raise ValueError("Empty image data received")

                # Decode image
                imgnp = np.array(bytearray(image_data), dtype=np.uint8)
                frame = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)

                if frame is None:
                    raise ValueError("Failed to decode image")

                print(f"Successfully captured frame from {camera_url}")
                return frame

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Camera connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)  # Wait 1 second before retry
                else:
                    print(f"Failed to get frame from camera after {max_retries} attempts: {e}")

        # Don't return any fallback frame - return None to indicate camera failure
        print("Camera connection failed - no fallback frame provided")
        return None

    def _detect_faces_in_image(self, img_path):
        """
        Fast face detection optimized for instant recognition
        
        Args:
            img_path: Path to the image file
            
        Returns:
            bool: True if faces are detected, False otherwise
        """
        try:
            # For instant mode, use only OpenCV for speed
            if self.config.get("instant_mode", False):
                return self._detect_faces_fast(img_path)
            
            # Original comprehensive detection for non-instant mode
            return self._detect_faces_comprehensive(img_path)
                
        except Exception as e:
            print(f"Face detection failed: {e}")
            return False
    
    def _detect_faces_fast(self, img_path):
        """Fast face detection using only OpenCV for instant recognition"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                print("Could not read image for face detection")
                return False
                
            # Resize image for faster processing
            height, width = img.shape[:2]
            if width > 640:  # Resize if image is too large
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use faster parameters for instant detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2,  # Slightly less accurate but faster
                minNeighbors=3,   # Reduced for speed
                minSize=(20, 20)  # Smaller minimum size
            )
            
            face_detected = len(faces) > 0
            if face_detected:
                print(f"Fast detection: {len(faces)} faces found")
            else:
                print("Fast detection: No faces found")
                
            return face_detected
            
        except Exception as e:
            print(f"Fast face detection failed: {e}")
            return False
    
    def _detect_faces_comprehensive(self, img_path):
        """Comprehensive face detection with multiple methods"""
        try:
            face_detected = False
            
            # Method 1: OpenCV Haar Cascade
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        print(f"OpenCV detected {len(faces)} faces")
                        face_detected = True
                    else:
                        print("OpenCV: No faces detected")
            except Exception as e:
                print(f"OpenCV face detection failed: {e}")
            
            # Method 2: DeepFace verification (if OpenCV fails)
            if not face_detected:
                try:
                    faces = DeepFace.extract_faces(
                        img_path=img_path,
                        detector_backend='opencv',
                        enforce_detection=False
                    )
                    
                    if faces and len(faces) > 0:
                        valid_faces = []
                        for face in faces:
                            if face.shape[0] > 50 and face.shape[1] > 50:
                                face_std = np.std(face)
                                if face_std > 0.01:
                                    valid_faces.append(face)
                        
                        if len(valid_faces) > 0:
                            print(f"DeepFace detected {len(valid_faces)} valid faces")
                            face_detected = True
                        else:
                            print("DeepFace: Faces detected but all were invalid")
                    else:
                        print("DeepFace: No faces detected")
                except Exception as e:
                    print(f"DeepFace face detection failed: {e}")
            
            if not face_detected:
                print("No faces detected using comprehensive methods")
            
            return face_detected
                
        except Exception as e:
            print(f"Comprehensive face detection failed: {e}")
            return False

    def _recognize_face_fast(self, img_path):
        """Fast face recognition using FaceNet model for instant results"""
        # First, check if there are any faces in the image
        if not self._detect_faces_in_image(img_path):
            print("No faces detected in the image")
            return None, []

        # Use minimal preprocessing for speed
        preprocessed_path = self._preprocess_face_image_fast(img_path)
        
        try:
            # Use only FaceNet model with cosine distance for fastest results
            print(f"Fast recognition using {self.config['primary_model']} with cosine distance...")
            
            start_time = time.time()
            dfs = DeepFace.find(
                img_path=preprocessed_path,
                db_path=self.db_dir,
                model_name=self.config["primary_model"],
                distance_metric="cosine",
                enforce_detection=False,
                silent=True
            )
            
            recognition_time = time.time() - start_time
            print(f"Fast recognition completed in {recognition_time:.3f} seconds")

            if dfs and not dfs[0].empty:
                matches = dfs[0].copy()
                matches['confidence_score'] = 1 - matches['distance']
                matches = matches.sort_values('distance')

                # Apply filtering for fast mode
                valid_matches = matches[
                    (matches['confidence_score'] >= self.config["confidence_threshold"]) &
                    (matches['distance'] <= self.config["max_distance_threshold"])
                ]
                
                if not valid_matches.empty:
                    max_confidence = valid_matches['confidence_score'].max()
                    best_match = valid_matches.iloc[0]
                    person_name = os.path.basename(os.path.dirname(best_match['identity']))
                    
                    result_entry = {
                        'model': self.config["primary_model"],
                        'metric': 'cosine',
                        'matches': valid_matches,
                        'confidence': max_confidence,
                        'match_count': len(valid_matches),
                        'top_person': person_name,
                        'processing_time': recognition_time
                    }
                    
                    print(f"Fast recognition: {person_name} ({max_confidence:.1%} confidence)")
                    return result_entry, [result_entry]
                else:
                    print("No valid matches found in fast mode")
            else:
                print("No matches found in fast mode")

        except Exception as e:
            print(f"Error in fast recognition: {e}")

        return None, []

    def _recognize_face_enhanced(self, img_path):
        """Enhanced face recognition with multiple models - used when instant mode is disabled"""
        # If instant mode is enabled, use fast recognition
        if self.config.get("instant_mode", False):
            return self._recognize_face_fast(img_path)
        
        # Otherwise, use the original comprehensive method
        best_results = None
        best_confidence = 0
        all_results = []
        person_votes = {}

        # First, check if there are any faces in the image
        if not self._detect_faces_in_image(img_path):
            print("No faces detected in the image")
            return None, []

        # Preprocess the input image for better recognition
        preprocessed_path = self._preprocess_face_image(img_path)
        
        # Use all available models for comprehensive recognition
        available_models = self.config["all_models"]

        for model in available_models:
            for metric in self.config["distance_metrics"]:
                try:
                    print(f"Trying {model} with {metric} distance...")

                    dfs = DeepFace.find(
                        img_path=preprocessed_path,
                        db_path=self.db_dir,
                        model_name=model,
                        distance_metric=metric,
                        enforce_detection=False,
                        silent=True
                    )

                    if dfs and not dfs[0].empty:
                        matches = dfs[0].copy()
                        matches['confidence_score'] = 1 - matches['distance']
                        matches = matches.sort_values('distance')

                        # Apply stricter filtering
                        valid_matches = matches[
                            (matches['confidence_score'] >= self.config["confidence_threshold"]) &
                            (matches['distance'] <= self.config["max_distance_threshold"])
                        ]
                        
                        if not valid_matches.empty:
                            max_confidence = valid_matches['confidence_score'].max()
                            best_match = valid_matches.iloc[0]
                            person_name = os.path.basename(os.path.dirname(best_match['identity']))
                            
                            # Track votes for consensus
                            if person_name not in person_votes:
                                person_votes[person_name] = {'votes': 0, 'total_confidence': 0, 'best_confidence': 0}
                            person_votes[person_name]['votes'] += 1
                            person_votes[person_name]['total_confidence'] += max_confidence
                            person_votes[person_name]['best_confidence'] = max(
                                person_votes[person_name]['best_confidence'], max_confidence
                            )
                            
                            result_entry = {
                                'model': model,
                                'metric': metric,
                                'matches': valid_matches,
                                'confidence': max_confidence,
                                'match_count': len(valid_matches),
                                'top_person': person_name
                            }
                            all_results.append(result_entry)

                            if max_confidence > best_confidence:
                                best_confidence = max_confidence
                                best_results = result_entry

                            print(f"Found {len(valid_matches)} valid matches (confidence: {max_confidence:.2%}, person: {person_name})")

                except Exception as e:
                    print(f"Error with {model}/{metric}: {str(e)}")

        # Apply consensus validation for enhanced mode
        if all_results and person_votes and self.config["consensus_threshold"] > 1:
            print(f"Person votes: {person_votes}")
            
            best_consensus_person = None
            best_consensus_confidence = 0
            
            for person, vote_data in person_votes.items():
                if (vote_data['votes'] >= self.config["consensus_threshold"] and 
                    vote_data['best_confidence'] >= self.config["strict_confidence_threshold"]):
                    avg_confidence = vote_data['total_confidence'] / vote_data['votes']
                    if avg_confidence > best_consensus_confidence:
                        best_consensus_confidence = avg_confidence
                        best_consensus_person = person
            
            if not best_consensus_person:
                print("No person met consensus requirements - rejecting all matches")
                return None, []
            else:
                print(f"Consensus reached for {best_consensus_person}")
                consensus_results = [r for r in all_results if r['top_person'] == best_consensus_person]
                if consensus_results:
                    best_results = max(consensus_results, key=lambda x: x['confidence'])

        return best_results, all_results

    def _format_results_to_json(self, test_image_path, best_results, all_results, is_violence_frame=False):
        """Format recognition results to JSON"""
        timestamp = datetime.now().isoformat()

        # Determine recognition mode and model count
        if self.config.get("instant_mode", False):
            models_tested = 1  # Only FaceNet in instant mode
            recognition_mode = "instant"
        else:
            models_tested = len([m for m in self.config["all_models"] if m != "DeepFace"]) * len(self.config["distance_metrics"])
            recognition_mode = "comprehensive"

        json_output = {
            "timestamp": timestamp,
            "test_image": os.path.basename(test_image_path),
            "test_image_path": test_image_path,
            "database_path": self.db_dir,
            "is_violence_frame": is_violence_frame,
            "recognition_results": {
                "success": best_results is not None,
                "best_match": None,
                "all_matches": [],
                "summary": {
                    "total_models_tested": models_tested,
                    "successful_recognitions": len(all_results),
                    "best_confidence": 0.0,
                    "recognition_mode": recognition_mode
                },
                "no_faces_detected": best_results is None and len(all_results) == 0
            }
        }

        # Add processing time if available
        if best_results and 'processing_time' in best_results:
            json_output["recognition_results"]["processing_time"] = best_results['processing_time']

        if best_results:
            matches = best_results['matches']
            print(f"DEBUG: best_results exists, matches type: {type(matches)}, length: {len(matches) if hasattr(matches, '__len__') else 'N/A'}")
            json_output["recognition_results"]["best_confidence"] = float(best_results['confidence'])

            # Best match details - handle both pandas DataFrame and list
            if hasattr(matches, 'empty') and not matches.empty:
                best_match = matches.iloc[0]
                json_output["recognition_results"]["best_match"] = {
                    "person": os.path.basename(os.path.dirname(best_match['identity'])),
                    "image_file": os.path.basename(best_match['identity']),
                    "full_path": best_match['identity'],
                    "confidence": float(best_match['confidence_score']),
                    "distance": float(best_match['distance']),
                    "model_used": best_results['model'],
                    "distance_metric": best_results['metric'],
                    "quality_rating": self._get_quality_rating(best_match['confidence_score'])
                }

                # Top matches
                for idx, (_, row) in enumerate(matches.head(self.config["max_matches"]).iterrows()):
                    match_data = {
                        "rank": idx + 1,
                        "person": os.path.basename(os.path.dirname(row['identity'])),
                        "image_file": os.path.basename(row['identity']),
                        "full_path": row['identity'],
                        "confidence": float(row['confidence_score']),
                        "distance": float(row['distance']),
                        "model_used": best_results['model'],
                        "distance_metric": best_results['metric'],
                        "quality_rating": self._get_quality_rating(row['confidence_score'])
                    }
                    json_output["recognition_results"]["all_matches"].append(match_data)
            elif isinstance(matches, list) and len(matches) > 0:
                # OpenCV results (list) - add basic face detection info
                print(f"DEBUG: Processing OpenCV list with {len(matches)} faces")
                best_face = max(matches, key=lambda x: x.get('confidence', 0))
                json_output["recognition_results"]["best_match"] = {
                    "person": best_face.get('identity', 'Unknown'),
                    "image_file": "OpenCV_Detection",
                    "full_path": "OpenCV_Face_Detection",
                    "confidence": float(best_face.get('confidence', 0.8)),
                    "distance": float(best_face.get('distance', 0.5)),
                    "model_used": best_results['model'],
                    "distance_metric": "opencv_detection",
                    "quality_rating": "Good"
                }
                
                # Add all detected faces to matches
                for idx, face_info in enumerate(matches):
                    print(f"DEBUG: Processing face {idx}: identity='{face_info.get('identity', 'NOT_FOUND')}', confidence={face_info.get('confidence', 0)}")
                    match_data = {
                        "rank": idx + 1,
                        "person": face_info.get('identity', f'Face_{idx + 1}'),
                        "image_file": "OpenCV_Detection",
                        "full_path": "OpenCV_Face_Detection",
                        "confidence": float(face_info.get('confidence', 0.8)),
                        "distance": float(face_info.get('distance', 0.5)),
                        "model_used": best_results['model'],
                        "distance_metric": "opencv_detection",
                        "quality_rating": "Good"
                    }
                    json_output["recognition_results"]["all_matches"].append(match_data)
            elif all_results and len(all_results) > 0:
                # Use all_results for OpenCV when matches is empty
                print(f"DEBUG: Using all_results with {len(all_results)} faces")
                # Find the best match from all_results
                best_face = max(all_results, key=lambda x: x.get('confidence', 0))
                
                json_output["recognition_results"]["best_match"] = {
                    "person": best_face.get('identity', 'Unknown'),
                    "image_file": "OpenCV_Detection",
                    "full_path": "OpenCV_Face_Detection",
                    "confidence": float(best_face.get('confidence', 0.8)),
                    "distance": float(best_face.get('distance', 0.5)),
                    "model_used": best_results['model'],
                    "distance_metric": "opencv_detection",
                    "quality_rating": "Good"
                }
                
                # Add all detected faces to matches
                for idx, face_info in enumerate(all_results):
                    print(f"DEBUG: Processing face {idx}: identity='{face_info.get('identity', 'NOT_FOUND')}', confidence={face_info.get('confidence', 0)}")
                    match_data = {
                        "rank": idx + 1,
                        "person": face_info.get('identity', f'Face_{idx + 1}'),
                        "image_file": "OpenCV_Detection",
                        "full_path": "OpenCV_Face_Detection",
                        "confidence": float(face_info.get('confidence', 0.8)),
                        "distance": float(face_info.get('distance', 0.5)),
                        "model_used": best_results['model'],
                        "distance_metric": "opencv_detection",
                        "quality_rating": "Good"
                    }
                    json_output["recognition_results"]["all_matches"].append(match_data)

        return json_output

    def _get_quality_rating(self, confidence):
        """Get quality rating based on confidence score"""
        thresholds = self.config["quality_thresholds"]
        if confidence > thresholds["excellent"]:
            return "EXCELLENT"
        elif confidence > thresholds["good"]:
            return "GOOD"
        elif confidence > thresholds["fair"]:
            return "FAIR"
        else:
            return "POOR"

    def _save_recognition_result(self, result):
        """Save recognition result to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recognition_result_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"Saved recognition result: {filepath}")

        except Exception as e:
            print(f"Failed to save recognition result: {e}")

    def get_recent_results(self, limit=10):
        """Get recent recognition results"""
        try:
            results = []
            if os.path.exists(self.results_dir):
                files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
                files.sort(reverse=True)  # Most recent first

                for filename in files[:limit]:
                    filepath = os.path.join(self.results_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            result = json.load(f)
                            results.append(result)
                    except Exception as e:
                        print(f"Error reading result file {filename}: {e}")

            return results

        except Exception as e:
            print(f"Error getting recent results: {e}")
            return []

    def get_statistics(self):
        """Get face recognition statistics"""
        return {
            "recognition_stats": self.recognition_stats.copy(),
            "database_info": self.verify_database_structure(),
            "cache_info": {
                "cache_files_count": len(self.model_cache_files),
                "last_cache_check": self.last_cache_check,
                "cache_valid": self.validate_model_cache() if self.model_cache_files else False
            },
            "directories": {
                "database_dir": self.db_dir,
                "captured_images_dir": self.captured_images_dir,
                "results_dir": self.results_dir
            }
        }

    def test_recognition_with_file(self, image_path):
        """Test face recognition with a specific image file"""
        try:
            if not os.path.exists(image_path):
                return {
                    "error": f"Image file not found: {image_path}",
                    "recognition_results": {"success": False}
                }

            # Perform recognition
            best_results, all_results = self._recognize_face_enhanced(image_path)

            # Format results
            result = self._format_results_to_json(image_path, best_results, all_results)

            # Save results
            self._save_recognition_result(result)

            return result

        except Exception as e:
            print(f"Error in test recognition: {e}")
            return {
                "error": str(e),
                "recognition_results": {"success": False}
            }

    def cleanup_old_files(self, days_old=7):
        """Clean up old captured images and results"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            cleaned_count = 0

            # Clean captured images
            if os.path.exists(self.captured_images_dir):
                for filename in os.listdir(self.captured_images_dir):
                    filepath = os.path.join(self.captured_images_dir, filename)
                    if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        cleaned_count += 1

            # Clean old results
            if os.path.exists(self.results_dir):
                for filename in os.listdir(self.results_dir):
                    filepath = os.path.join(self.results_dir, filename)
                    if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        cleaned_count += 1

            print(f"Cleaned up {cleaned_count} old files")
            return cleaned_count

        except Exception as e:
            print(f"Error during cleanup: {e}")
            return 0

    def recognize_with_milvus(self, image_path: str) -> Dict[str, Any]:
        """
        Recognize faces using Milvus vector database

        Args:
            image_path: Path to the image to recognize

        Returns:
            Recognition results with Milvus data
        """
        try:
            if not self.milvus_manager:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "error": "Milvus manager not available",
                    "recognition_results": {"success": False}
                }

            # Search for similar faces in Milvus
            matches = self.milvus_manager.search_similar_faces(
                image_path,
                limit=5,
                threshold=0.8
            )

            if not matches:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "captured_image": os.path.basename(image_path),
                    "recognition_results": {
                        "success": False,
                        "method": "milvus",
                        "message": "No similar faces found in vector database"
                    }
                }

            # Format results
            best_match = matches[0]  # Highest similarity

            result = {
                "timestamp": datetime.now().isoformat(),
                "captured_image": os.path.basename(image_path),
                "captured_image_path": image_path,
                "recognition_results": {
                    "success": True,
                    "method": "milvus",
                    "best_match": {
                        "person": best_match["person_name"],
                        "confidence": best_match["confidence"],
                        "similarity": best_match["similarity"],
                        "distance": best_match["distance"],
                        "image_file": best_match["image_filename"],
                        "person_id": best_match["person_id"],
                        "quality_rating": self._get_quality_rating(best_match["confidence"])
                    },
                    "all_matches": [
                        {
                            "rank": idx + 1,
                            "person": match["person_name"],
                            "confidence": match["confidence"],
                            "similarity": match["similarity"],
                            "distance": match["distance"],
                            "image_file": match["image_filename"],
                            "person_id": match["person_id"],
                            "quality_rating": self._get_quality_rating(match["confidence"])
                        }
                        for idx, match in enumerate(matches[:5])
                    ],
                    "total_matches": len(matches)
                }
            }

            # Update statistics
            self.recognition_stats["total_recognitions"] += 1
            self.recognition_stats["successful_recognitions"] += 1
            self.recognition_stats["last_recognition_time"] = timezone.now()

            # Save results
            self._save_recognition_result(result)

            print(f"Milvus recognition successful: {best_match['person_name']} ({best_match['confidence']:.1%})")
            return result

        except Exception as e:
            print(f"Error in Milvus recognition: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "recognition_results": {"success": False, "method": "milvus"}
            }


    def get_milvus_stats(self) -> Dict[str, Any]:
        """Get Milvus database statistics"""
        try:
            if not self.milvus_manager:
                return {"available": False, "message": "Milvus manager not available"}

            stats = self.milvus_manager.get_collection_stats()
            stats["available"] = True
            return stats

        except Exception as e:
            print(f"Error getting Milvus stats: {e}")
            return {"available": False, "error": str(e)}

    def _recognize_face_opencv(self, frame):
        """
        OpenCV-based face detection and recognition using reference images
        
        Args:
            frame: OpenCV frame/image array
            
        Returns:
            tuple: (best_results, all_results) in format compatible with existing code
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
            
            if len(faces) == 0:
                return [], []
            
            # Load reference faces for comparison
            reference_faces = self._load_reference_faces()
            
            # Format results to match the expected structure for _format_results_to_json
            all_results = []
            best_match = None
            best_confidence = 0.0
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Try to match with reference faces
                matched_person, confidence = self._match_face_with_references(face_roi, reference_faces)
                
                # Create result structure
                result = {
                    "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "confidence": confidence,
                    "face_confidence": confidence,
                    "identity": matched_person,
                    "distance": 1.0 - confidence,  # Convert confidence to distance
                    "threshold": 0.6,
                    "model": "opencv_haarcascade",
                    "detector_backend": "opencv",
                    "similarity_metric": "opencv_template_matching"
                }
                all_results.append(result)
                
                # Track best match
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = result
            
            # Create best_results in the format expected by _format_results_to_json
            if all_results:
                best_results = {
                    "confidence": best_confidence,
                    "model": "opencv_haarcascade",
                    "matches": all_results,  # Include all matches for OpenCV detection
                    "processing_time": 0.1
                }
            else:
                best_results = None
            
            print(f"OpenCV detected {len(faces)} face(s) - Best match: {best_match['identity'] if best_match else 'None'} (confidence: {best_confidence:.2f})")
            return best_results, all_results
            
        except Exception as e:
            print(f"Error in OpenCV face detection: {e}")
            return [], []

    def _load_reference_faces(self):
        """
        Load reference face images from face_db directory
        
        Returns:
            dict: Dictionary mapping person names to their face images
        """
        reference_faces = {}
        
        try:
            if not os.path.exists(self.db_dir):
                print(f"Face database directory not found: {self.db_dir}")
                return reference_faces
            
            # Scan each person's directory
            for person_dir in os.listdir(self.db_dir):
                person_path = os.path.join(self.db_dir, person_dir)
                if not os.path.isdir(person_path):
                    continue
                
                # Look for image files in the person's directory
                for filename in os.listdir(person_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_path, filename)
                        try:
                            # Load and preprocess the reference image
                            ref_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                            if ref_image is not None:
                                # Resize to standard size for comparison
                                ref_image = cv2.resize(ref_image, (100, 100))
                                reference_faces[person_dir] = ref_image
                                print(f"Loaded reference face for {person_dir}")
                                break  # Use first valid image found
                        except Exception as e:
                            print(f"Error loading reference image {image_path}: {e}")
            
            print(f"Loaded {len(reference_faces)} reference faces: {list(reference_faces.keys())}")
            return reference_faces
            
        except Exception as e:
            print(f"Error loading reference faces: {e}")
            return reference_faces

    def _match_face_with_references(self, face_roi, reference_faces):
        """
        Match a detected face with reference faces using template matching
        
        Args:
            face_roi: Grayscale face region of interest
            reference_faces: Dictionary of reference face images
            
        Returns:
            tuple: (person_name, confidence_score)
        """
        try:
            if not reference_faces:
                return "Unknown", 0.5
            
            # Resize face ROI to standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            best_match = "Unknown"
            best_confidence = 0.0
            
            # Compare with each reference face
            for person_name, ref_face in reference_faces.items():
                try:
                    # Method 1: Template matching
                    result = cv2.matchTemplate(face_roi, ref_face, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Method 2: Histogram comparison
                    hist1 = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([ref_face], [0], None, [256], [0, 256])
                    hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    
                    # Method 3: Structural similarity
                    try:
                        from skimage.metrics import structural_similarity as ssim
                        ssim_score = ssim(face_roi, ref_face)
                    except:
                        ssim_score = 0.5
                    
                    # Method 4: Edge detection comparison
                    edges1 = cv2.Canny(face_roi, 50, 150)
                    edges2 = cv2.Canny(ref_face, 50, 150)
                    edge_corr = cv2.compareHist(cv2.calcHist([edges1], [0], None, [256], [0, 256]), 
                                              cv2.calcHist([edges2], [0], None, [256], [0, 256]), 
                                              cv2.HISTCMP_CORREL)
                    
                    # Method 5: Simple pixel difference (inverted)
                    diff = cv2.absdiff(face_roi, ref_face)
                    pixel_similarity = 1.0 - (np.mean(diff) / 255.0)
                    
                    # Combine all methods (weighted average) with balanced validation
                    # Use template matching even if negative, but weight it less
                    template_score = max(0, max_val)  # Ensure non-negative
                    
                    # Weight the scores more carefully for better recognition - focus on histogram and SSIM
                    combined_confidence = (template_score * 0.2) + (hist_corr * 0.4) + (ssim_score * 0.3) + (edge_corr * 0.1)
                    
                    # Boost confidence if histogram and SSIM are decent
                    if hist_corr > 0.2 and ssim_score > 0.05:
                        combined_confidence *= 1.2  # Boost for decent histogram and SSIM
                    
                    # More lenient validation: only penalize very low scores
                    if template_score < 0.05 and hist_corr < 0.15 and ssim_score < 0.05:
                        combined_confidence *= 0.8  # Less harsh penalty
                    
                    print(f"Matching with {person_name}: template={max_val:.3f}, hist={hist_corr:.3f}, ssim={ssim_score:.3f}, edge={edge_corr:.3f}, pixel={pixel_similarity:.3f}, combined={combined_confidence:.3f}")
                    
                    if combined_confidence > best_confidence:
                        best_confidence = combined_confidence
                        best_match = person_name
                        
                except Exception as e:
                    print(f"Error matching with {person_name}: {e}")
                    continue
            
            # Apply threshold - if confidence is too low, mark as unknown
            if best_confidence < 0.25:  # Balanced threshold for better recognition
                best_match = "Unknown"
                best_confidence = 0.1
                print(f"❌ Low confidence match: {best_match} (confidence: {best_confidence:.3f}) - marking as Unknown")
            else:
                print(f"✅ Successfully matched face to: {best_match} (confidence: {best_confidence:.3f})")
            
            return best_match, best_confidence
            
        except Exception as e:
            print(f"Error in face matching: {e}")
            return "Unknown", 0.5


# Global instance for Django integration
face_recognition_system = None

def get_face_recognition_system():
    """Get or create the global face recognition system instance"""
    global face_recognition_system

    if face_recognition_system is None:
        try:
            face_recognition_system = FaceRecognation()
        except Exception as e:
            print(f"Failed to initialize face recognition system: {e}")
            return None

    return face_recognition_system

def recognize_violence_frame(frame):
    """
    Convenience function to recognize faces in a violence frame
    This is the main function that YOLO detector will call
    """
    system = get_face_recognition_system()
    if system is None:
        return {
            "error": "Face recognition system not available",
            "recognition_results": {"success": False}
        }

    return system.recognize_face_in_frame(frame)

def recognize_latest_test_image():
    """
    Convenience function to recognize faces in the latest captured captured image
    """
    system = get_face_recognition_system()
    if system is None:
        return {
            "error": "Face recognition system not available",
            "recognition_results": {"success": False}
        }

    return system.recognize_latest_captured_image()

def get_latest_captured_image_path():
    """
    Get the path to the latest captured captured image
    """
    system = get_face_recognition_system()
    if system is None:
        return None

    return system.get_latest_test_image()

def recognize_live_camera():
    """
    Convenience function to recognize faces using live ESP32 camera
    """
    system = get_face_recognition_system()
    if system is None:
        return {
            "error": "Face recognition system not available",
            "recognition_results": {"success": False}
        }

    return system.recognize_live_camera_frame()