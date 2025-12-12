from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import time
from collections import defaultdict, deque
import os
import urllib.request
import threading
import logging
from datetime import datetime
# from detection_engine.face_recognation import recognize_live_camera  # Circular import - commented out
#from facerecognation.milvus_manager import MilvusFaceManager

# Set up logging
yolo_logger = logging.getLogger('yolo_detection')
debug_logger = logging.getLogger('detection_engine')

FACE_RECOGNITION_AVAILABLE = True

class YOLODetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        if model_path is None:
            # Get the absolute path to the model file
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.model_path = os.path.join(project_root, "models", "dataset", "violence_detection_run", "weights", "best.pt")
        else:
            self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(self.model_path)
        self.frame_count = 0
        self.camera_stream_url = 'http://10.15.186.3/cam-hi.jpg'  # Updated to match your network
        self.stream_active = True
        self.violence_alert_count = 0
        self.last_violence_time = 0
        self.detection_history = deque(maxlen=30)  # Keep last 30 frames
        # Image capture settings
        self.captured_images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media', 'captured_db')
        self.last_image_capture_time = 0
        self.image_capture_cooldown = 0.1  # seconds between image captures (reduced for testing)
        self.captured_images = []  # Track captured images
        self.max_images_in_dir = 10  # Maximum images in test_images directory (reduced for face recognition)
        # Threading control
        self.is_running = False
        self.detection_thread = None
        # Set current violence status for immediate dashboard response
        self.current_violence_status = False
        self.violence_alert_active = False
        self.violence_alert_start_time = 0
        # Face recognition integration - optimized for instant recognition
        self.face_recognition_enabled = FACE_RECOGNITION_AVAILABLE
        self.last_face_recognition_time = 0
        self.face_recognition_cooldown = 1  # Reduced to 1 second for instant recognition


        # Color palette for different classes
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0),
            (0, 128, 128), (128, 0, 0), (192, 192, 192), (255, 20, 147), (32, 178, 170)
        ]
        self.class_names = {
            0: 'Violence',
            1: 'Weapon',
            2: 'Knife',
            3: 'NonViolence'
        }
        # Special colors for violence detection classes
        self.violence_colors = {
            'Violence': (0, 0, 255),      # Red for violence
            'Weapon': (255, 0, 0),        # Bright red for weapons
            'Knife': (255, 0, 0),         # Bright red for knives
            'NonViolence': (0, 255, 0)    # Green for non-violence
        }

        # Initialize missing attributes that views.py expects
        self.detected_objects = defaultdict(int)
        self.is_violence_model = True  # This is a violence detection model
        self.enhance_quality = True
        self.jpeg_quality = 95
        self.milvus_manager = None  # Placeholder for Milvus integration
        
        print(f"Model loaded successfully! Confidence threshold: {confidence_threshold}")
        if self.is_violence_model:
            print("⚠️  Violence detection mode activated!")
        print(f"📸 Violence frames will be saved to: {self.captured_images_dir}")
    
     
     # Draw Bounding Boxes
    def draw_custom_box(self, img, x1, y1, x2, y2, class_name, confidence, color):
        # Box thickness based on confidence
        thickness = max(2, int(confidence * 4))
        
        # Draw main bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner decorations
        corner_length = 20
        corner_thickness = 3
        
        # Top-left corner
        cv2.line(img, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # Top-right corner
        cv2.line(img, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(img, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Draw label background
        label_bg_color = tuple(int(c * 0.7) for c in color)  # Darker version of box color
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), label_bg_color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), text_thickness)
        
        return img
     
     # Draw Violence Alert
    def draw_violence_alert(self, img, alert_message):
        
        if not alert_message:
            return img
        
        height, width = img.shape[:2]
        
        # Create alert overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        # Draw alert text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        color = (255, 255, 255)
        thickness = 3
        
        # Center the text
        text_size = cv2.getTextSize(alert_message, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50
        
        cv2.putText(img, alert_message, (text_x, text_y), font, font_scale, color, thickness)
        
        return img
     
     # Save violence frame
    def save_violence_frame(self, frame, violence_type, confidence):
        try:
            current_time = time.time()
            
            
            # Check and manage captured_db directory before capturing
            self._check_and_manage_images_before_capture()

            # Generate unique filename in format YYYY-MM-DD_HH-MM-SS.jpg
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp_str}.jpg"
            filepath = os.path.join(self.captured_images_dir, filename)
            
            # Save the frame
            success = cv2.imwrite(filepath, frame)
            
            if success:
                self.last_image_capture_time = current_time
                self.captured_images.append({
                    'path': filepath,
                    'timestamp': current_time,
                    'violence_type': violence_type,
                    'confidence': confidence
                })
                print(f"📸 Violence frame saved to file: {filename}")
                # Keep only last 10 images in memory (matching directory limit)
                if len(self.captured_images) > 10:
                    self.captured_images.pop(0)
                
            else:
                print(f"❌ Failed to save violence frame: {filename}")
                return None
                
        except Exception as e:
            print(f"❌ Error saving violence frame: {e}")
            return None
     
     # Check for violence-related detections and generate alerts
    def violence_alert(self, objects, frame=None):

        violence_detected = False
        alert_message = ""
        current_time = time.time()
        violence_type = ""
        violence_confidence = 0.0

        # Check for immediate violence in current frame
        frame_has_violence = False
        for obj in objects:
            class_name = obj['class_name']
            confidence = obj['confidence']
            if class_name in ['Violence', 'Weapon', 'Knife'] and confidence > 0.6:
                frame_has_violence = True
                violence_type = class_name
                violence_confidence = confidence
                break
        
        # If violence detected in current frame, immediately capture and alert
        if frame_has_violence:
            violence_detected = True
            
            # Set current violence status for immediate dashboard response
            self.current_violence_status = True
            self.violence_alert_active = True
            self.violence_alert_start_time = current_time
            self.violence_alert_count += 1
            self.last_violence_time = current_time
            
            # Create appropriate alert message based on violence type
            if violence_type == 'Violence':
                alert_message = f"🚨 VIOLENCE DETECTED! (Confidence: {violence_confidence:.1%})"
            elif violence_type == 'Weapon':
                alert_message = f"🔫 WEAPON DETECTED! (Confidence: {violence_confidence:.1%})"
            elif violence_type == 'Knife':
                alert_message = f"🔪 KNIFE DETECTED! (Confidence: {violence_confidence:.1%})"
            else:
                alert_message = f"🚨 VIOLENCE DETECTED! (Confidence: {violence_confidence:.1%})"
            
            print(f"{alert_message} (Alert #{self.violence_alert_count})")
            
            # Capture frame immediately when violence is detected
            if frame is not None:
                captured_path = self.save_violence_frame(frame, violence_type, violence_confidence)
                if captured_path:
                    print(f"📸 Violence frame captured: {captured_path}")
        else:
            # No violence in current frame
            if self.violence_alert_active:
                # Check if we should clear the alert (simplified for immediate response)
                # You can add more sophisticated logic here if needed
                recent_violence = sum(list(self.violence_frame_buffer)[-10:])  # Check last 10 frames
                if recent_violence == 0:  # No violence in last 10 frames
                    self.violence_alert_active = False
                    self.current_violence_status = False
                    alert_message = f"✅ Violence alert cleared."
                    print(alert_message)
        
        # Statistics for return
        stats = {
            'trigger_count': self.violence_alert_count,
            'violence_alert_active': self.violence_alert_active,
            'current_violence_status': self.current_violence_status,
            'last_detection_time': self.last_violence_time,
            'alert_start_time': self.violence_alert_start_time
        }

        return violence_detected, alert_message, stats

     # Clear Captured Images
    def clear_all_captured_images(self):
        try:
            if not os.path.exists(self.captured_images_dir):
                print("Test images directory does not exist")
                return True

            # Get all image files in the directory
            image_files = []
            for file in os.listdir(self.captured_images_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    file_path = os.path.join(self.captured_images_dir, file)
                    image_files.append(file_path)

            if not image_files:
                print("No images to delete in test_images directory")
                return True

            # Delete all images
            deleted_count = 0
            for file_path in image_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted image: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

            # Clear the captured images tracking list
            self.captured_images = []
            
            print(f"Successfully cleared {deleted_count} images from test_images directory")
            return True

        except Exception as e:
            print(f"Error clearing test_images directory: {e}")
            return False

     # Tregure Face Recognition
    def trigger_face_recognition(self, frame):

        if not self.face_recognition_enabled:
            print("Face recognition not available")
            return None

        current_time = time.time()

        # Use the configured cooldown for instant recognition
        if current_time - self.last_face_recognition_time < self.face_recognition_cooldown:
            print("Face recognition cooldown active")
            return None

        try:
            print("🚀 Triggering INSTANT face recognition for violence frame")
            start_time = time.time()

            # Call fast face recognition system
            # recognition_result = recognize_live_camera()  # Commented out due to circular import
            recognition_result = None  # Placeholder until circular import is resolved

            # Update last recognition time
            self.last_face_recognition_time = current_time
            
            processing_time = time.time() - start_time

            # Log results with timing
            if recognition_result.get("recognition_results", {}).get("success", False):
                best_match = recognition_result["recognition_results"].get("best_match", {})
                person = best_match.get("person", "Unknown")
                confidence = best_match.get("confidence", 0)
                model_used = best_match.get("model_used", "Unknown")
                print(f"⚡ INSTANT FACE RECOGNITION: {person} ({confidence:.1%} confidence, {processing_time:.2f}s, {model_used})")
            else:
                if recognition_result.get("recognition_results", {}).get("no_faces_detected"):
                    print(f"⚡ Instant face detection: No faces found ({processing_time:.2f}s)")
                else:
                    print(f"⚡ Instant face recognition: No matches found ({processing_time:.2f}s)")

            # Add timing information to results
            if "recognition_results" in recognition_result:
                recognition_result["recognition_results"]["processing_time"] = processing_time
                recognition_result["recognition_results"]["recognition_mode"] = "instant"

            return recognition_result

        except Exception as e:
            print(f"Error in instant face recognition: {e}")
            return {
                "error": str(e),
                "recognition_results": {"success": False, "recognition_mode": "instant"}
            }

     # Process detections
    def process_detections(self, results):
        
        current_objects = []
        
        for result in results:
            # Handle pose detection results
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # Process pose detection
                keypoints = result.keypoints
                if keypoints is not None:
                    for i, kpts in enumerate(keypoints):
                        if kpts is not None and len(kpts) > 0:
                            # Get bounding box from keypoints
                            x_coords = kpts[:, 0]
                            y_coords = kpts[:, 1]
                            x1, x2 = int(x_coords.min()), int(x_coords.max())
                            y1, y2 = int(y_coords.min()), int(y_coords.max())
                            
                            # Add some padding
                            padding = 20
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(result.orig_shape[1], x2 + padding)
                            y2 = min(result.orig_shape[0], y2 + padding)
                            
                            current_objects.append({
                                'bbox': (x1, y1, x2, y2),
                                'class_name': 'person',
                                'confidence': 0.8,  # Default confidence for pose
                                'class_id': 0,
                                'keypoints': kpts.cpu().numpy() if hasattr(kpts, 'cpu') else kpts
                            })
                            self.detected_objects['person'] += 1
            
            # Handle object detection results
            elif hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    if confidence >= self.confidence_threshold:
                        current_objects.append({
                            'bbox': (x1, y1, x2, y2),
                            'class_name': class_name,
                            'confidence': confidence,
                            'class_id': class_id
                        })
                        
                        # Update detection count
                        self.detected_objects[class_name] += 1
        
        return current_objects

     # Process Frame Main function of the YOLO detection
    def process_frame(self, frame):

        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Process detections
        objects = self.process_detections(results)
        
        # Check for violence alerts
        violence_detected, alert_message, violence_stats = self.violence_alert(objects, frame)

        # Capture frame and trigger face recognition if violence detected
        captured_image_path = None
        face_recognition_result = None
        if violence_detected:
            # Capture the violence frame first
            violence_type = "violence"
            max_confidence = 0.0

            # Find the highest confidence violence-related detection
            for obj in objects:
                if obj['class_name'] in ['Violence', 'Weapon', 'Knife'] and obj['confidence'] > 0.6:
                    if obj['confidence'] > max_confidence:
                        max_confidence = obj['confidence']
                        violence_type = obj['class_name'].lower()

            # Capture the frame
            captured_image_path = self.capture_violence_frame(frame, violence_type, max_confidence)

            # Trigger face recognition
            face_recognition_result = self.trigger_face_recognition(frame)

        # Draw bounding boxes and labels
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            class_id = obj['class_id']
            
            # Get color for this class
            if class_name in self.violence_colors:
                color = self.violence_colors[class_name]
            else:
                color = self.colors[class_id % len(self.colors)]
            
            # Draw custom bounding box
            frame = self.draw_custom_box(frame, x1, y1, x2, y2, class_name, confidence, color)
            
            # Draw pose keypoints if available
            if 'keypoints' in obj and obj['keypoints'] is not None:
                keypoints = obj['keypoints']
                for kp in keypoints:
                    if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw violence alert if detected
        if violence_detected:
            frame = self.draw_violence_alert(frame, alert_message)
        
        # Update frame counter
        self.frame_count += 1
        
        # Store detection history
        self.detection_history.append(len(objects))
        
        processing_time = time.time() - start_time

        # Prepare capture info
        capture_info = {
            'captured': captured_image_path is not None,
            'image_path': captured_image_path,
            'latest_image': self.get_frame_from_captured_image(),
            'total_captured': len(self.captured_images)
        }

        return frame, objects, violence_detected, alert_message, processing_time, violence_stats, face_recognition_result, capture_info
    

    # Get latest captured image
    def get_frame_from_captured_image(self):
               
            try:
                image_dir = self.captured_images_dir
                if not os.path.exists(image_dir):
                    return None
                # List all files in the directory, filter for image files
                files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not files:
                    return None
                # Sort files by modification time, descending
                files.sort(key=lambda f: os.path.getmtime(os.path.join(image_dir, f)), reverse=True)
                latest_image_path = os.path.join(image_dir, files[0])
                # Read image using OpenCV
                frame = cv2.imread(latest_image_path)
                return frame
            except Exception as e:
                # Optionally log the error if logger is available
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error getting latest captured image frame: {e}")
                return None

    # Get frame from camera
    def get_frame_from_camera(self, camera_url=None):
        if camera_url is None:
            camera_url = self.camera_stream_url

        try:
            # Prepare request
            request = urllib.request.Request(camera_url)
            request.add_header('User-Agent', 'YOLODetector/1.0')
            request.add_header('Accept', 'image/jpeg,image/*;q=0.9,*/*;q=0.8')
            request.add_header('Cache-Control', 'no-cache')

            # Open and read image data
            with urllib.request.urlopen(request, timeout=5) as response:
                image_data = response.read()
                if not image_data:
                    return None

            # Decode image to OpenCV frame
            imgnp = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
            if frame is None:
                return None

            return frame  # Return OpenCV frame, not JPEG bytes
        except Exception as e:
            # Optionally log the error if logger is available
            if hasattr(self, 'logger'):
                self.logger.error(f"Error getting frame from camera: {e}")
            return None

    # Missing methods that views.py expects
    def get_violence_logs(self):
        """Get violence detection logs"""
        logs = []
        for img in self.captured_images:
            logs.append({
                'timestamp': img['timestamp'],
                'type': img['violence_type'],
                'confidence': img['confidence'],
                'alert_message': f"{img['violence_type']} detected",
                'violence_percentage': img['confidence'] * 100,
                'trigger_count': self.violence_alert_count,
                'total_violence': sum(1 for i in self.captured_images if 'violence' in i['violence_type'].lower()),
                'total_weapons': sum(1 for i in self.captured_images if 'weapon' in i['violence_type'].lower()),
                'total_knives': sum(1 for i in self.captured_images if 'knife' in i['violence_type'].lower()),
                'frame_count': self.frame_count
            })
        return logs

    def clear_violence_logs(self):
        """Clear violence detection logs"""
        self.captured_images = []
        self.violence_alert_count = 0
        print("Violence logs cleared")

    def test_camera_connection(self, camera_url):
        """Test camera connection"""
        try:
            import urllib.request
            import time
            
            start_time = time.time()
            response = urllib.request.urlopen(camera_url, timeout=5)
            response_time = time.time() - start_time
            
            if response.getcode() == 200:
                image_data = response.read()
                return {
                    'success': True,
                    'url': camera_url,
                    'response_time': round(response_time, 2),
                    'image_size': len(image_data),
                    'status_code': response.getcode()
                }
            else:
                return {
                    'success': False,
                    'url': camera_url,
                    'error': f'HTTP {response.getcode()}',
                    'response_time': round(response_time, 2)
                }
        except Exception as e:
            return {
                'success': False,
                'url': camera_url,
                'error': str(e),
                'response_time': 0
            }

    def get_alternative_camera_urls(self):
        """Get alternative camera URLs to test"""
        base_ips = ['192.168.1', '192.168.0', '192.168.68', '10.0.0']
        ports = ['', ':8080', ':81', ':8081']
        paths = ['/cam-hi.jpg', '/cam.jpg', '/stream', '/mjpeg/1']
        
        urls = []
        for base_ip in base_ips:
            for port in ports:
                for path in paths:
                    for i in range(100, 120):  # Test IPs 100-119
                        url = f"http://{base_ip}.{i}{port}{path}"
                        urls.append(url)
        return urls

    def get_violence_status(self):
        """Get current violence status"""
        return {
            'violence_alert_active': self.violence_alert_active,
            'current_violence_status': self.current_violence_status,
            'violence_alert_count': self.violence_alert_count,
            'last_violence_time': self.last_violence_time,
            'alert_start_time': self.violence_alert_start_time
        }

    def get_latest_captured_image_path(self):
        """Get the latest captured image path"""
        if not self.captured_images:
            return None
        return self.captured_images[-1]['path']

    def clear_all_test_images(self):
        """Clear all test images - alias for clear_all_captured_images"""
        return self.clear_all_captured_images()

    def capture_violence_frame(self, frame, violence_type, confidence):
        """Capture violence frame - alias for save_violence_frame"""
        return self.save_violence_frame(frame, violence_type, confidence)

    def get_violence_frames_from_milvus(self, limit=10):
        """Get violence frames from Milvus - placeholder"""
        # This would integrate with Milvus database
        return []

    def _check_and_manage_images_before_capture(self):
        """Check and manage images before capture"""
        try:
            if not os.path.exists(self.captured_images_dir):
                os.makedirs(self.captured_images_dir, exist_ok=True)
                return

            # Get all image files
            image_files = []
            for file in os.listdir(self.captured_images_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    file_path = os.path.join(self.captured_images_dir, file)
                    image_files.append((file_path, os.path.getmtime(file_path)))

            # Sort by modification time (oldest first)
            image_files.sort(key=lambda x: x[1])

            # Remove oldest images if we exceed the limit
            while len(image_files) > self.max_images_in_dir:
                oldest_file = image_files.pop(0)[0]
                try:
                    os.remove(oldest_file)
                    print(f"Removed old image: {os.path.basename(oldest_file)}")
                except Exception as e:
                    print(f"Failed to remove {oldest_file}: {e}")

        except Exception as e:
            print(f"Error managing images: {e}")
