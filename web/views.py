from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse, Http404, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
#from django.contrib import messages
# Import face recognition system
# from detection_engine.face_recognation import FaceRecognation, get_face_recognition_system
# from detection_engine.yolo_detection import YOLODetection
import cv2
import json
import threading
import time
import logging
import os
import mimetypes
import base64
from datetime import datetime

# Set up specific loggers for different components
logger = logging.getLogger('web')
violence_logger = logging.getLogger('violence_alerts')
face_logger = logging.getLogger('face_recognition')
debug_logger = logging.getLogger('detection_engine')

try:
    from detection_engine.yolo_detection import YOLODetector
    YOLODetectorAvailable = True
except ImportError as e:
    debug_logger.warning(f"YOLO detector not available: {e}")
    YOLODetector = None
    YOLODetectorAvailable = False

try:
    from detection_engine.face_recognation import FaceRecognation
    FaceRecognationAvailable = True
except ImportError as e:
    face_logger.warning(f"Face recognition engine not available: {e}")
    FaceRecognation = None
    FaceRecognationAvailable = False
# Global detector instance
detector = None
recognition_engine = None
detection_active = False
detection_thread = None
violence_alert_count = 0
last_violence_alert = None


def dashboard(request):
    """
    Main dashboard view
    """
    return render(request, 'web/dashboard.html')

def get_detector():
    """
    Get or create detector instance
    """
    global detector
    if detector is None and YOLODetectorAvailable:
        try:
            print("🔧 Initializing YOLO detector...")
            detector = YOLODetector()
            debug_logger.info("YOLO detector initialized successfully")
            print("✅ YOLO detector initialized successfully")
        except Exception as e:
            debug_logger.error(f"Failed to initialize YOLO detector: {e}")
            print(f"❌ Failed to initialize YOLO detector: {e}")
            detector = None
    elif detector is None:
        print("❌ YOLO detector not available - YOLODetectorAvailable = False")
    else:
        print("✅ YOLO detector already initialized")
    return detector

def get_recognition_engine():
    """
    Get or create recognition engine instance
    """
    global recognition_engine
    if recognition_engine is None and FaceRecognationAvailable:
        try:
            recognition_engine = FaceRecognation()
            face_logger.info("Face recognition engine initialized successfully")
        except Exception as e:
            face_logger.error(f"Failed to initialize face recognition engine: {e}")
            recognition_engine = None
    return recognition_engine


def generate_frames():
    """
    Generator function for video streaming with detection
    """
    global detection_active
    detector_instance = get_detector()
    
    debug_logger.info("Starting video stream generation")
    frame_count = 0

    # Stream continuously, regardless of detection_active status
    while True:
        try:
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                debug_logger.info(f"Generated {frame_count} frames")
                
            if detector_instance is None:
                # Create a simple test frame when detector is not available
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (50, 50, 50)  # Dark gray background

                # Add text
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "DETECTOR NOT INITIALIZED"
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                x = (640 - text_size[0]) // 2
                y = (480 + text_size[1]) // 2
                cv2.putText(frame, text, (x, y), font, 1, (0, 0, 255), 2)
            else:
                # Get frame from camera
                try:
                    frame = detector_instance.get_frame_from_camera()
                except Exception as e:
                    debug_logger.error(f"Error getting frame from camera: {e}")
                    frame = None

                if frame is not None:
                    if detection_active:
                        # Process frame with YOLO detection
                        try:
                            processed_frame, objects, violence_detected, alert_message, processing_time, violence_stats, face_recognition_result, capture_info = detector_instance.process_frame(frame)
                            frame = processed_frame
                            # Log image capture results
                            if capture_info and capture_info.get('captured', False):
                                image_path = capture_info.get('image_path', '')
                                violence_logger.info(f"📸 Violence frame captured: {os.path.basename(image_path)}")

                            # Log face recognition results if available
                            if face_recognition_result and face_recognition_result.get("recognition_results", {}).get("success", False):
                                best_match = face_recognition_result["recognition_results"].get("best_match", {})
                                person = best_match.get("person", "Unknown")
                                confidence = best_match.get("confidence", 0)
                                face_logger.info(f"🔍 Face recognized in violence: {person} ({confidence:.1%})")
                            print("violence_detected",violence_detected,"alert_message",alert_message)

                            
                        except Exception as e:
                            debug_logger.error(f"Error in YOLO processing: {e}")
                            # Use original frame if processing fails
                    # If detection not active, use raw frame
                else:
                    # Create error frame when camera not available
                    import numpy as np
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    frame[:] = (50, 50, 50)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text1 = "CAMERA NOT CONNECTED"
                    text2 = "Check ESP32-CAM connection"

                    text_size1 = cv2.getTextSize(text1, font, 1, 2)[0]
                    text_size2 = cv2.getTextSize(text2, font, 0.7, 2)[0]

                    x1 = (640 - text_size1[0]) // 2
                    x2 = (640 - text_size2[0]) // 2

                    cv2.putText(frame, text1, (x1, 220), font, 1, (0, 0, 255), 2)
                    cv2.putText(frame, text2, (x2, 260), font, 0.7, (255, 255, 0), 2)

            # Encode frame as high-quality JPEG
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 95,           # High quality (95%)
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,           # Optimize encoding
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1,        # Progressive JPEG
                cv2.IMWRITE_JPEG_LUMA_QUALITY, 95,      # Luma quality
                cv2.IMWRITE_JPEG_CHROMA_QUALITY, 95     # Chroma quality
            ]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                       frame_bytes + b'\r\n')

            time.sleep(0.020)  # ~30 FPS

        except Exception as e:
            debug_logger.error(f"Error in frame generation: {e}")
            # Create error frame
            import numpy as np
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            error_frame[:] = (50, 50, 50)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"STREAMING ERROR: {str(e)[:30]}"
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
            x = (640 - text_size[0]) // 2
            cv2.putText(error_frame, text, (x, 240), font, 0.7, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(1)  # Wait longer on error

def process_frame_with_detection(frame):
    """
    Process a single frame with detection and return processed frame and alert info
    """
    try:
        detector_instance = get_detector()
        if detector_instance is None:
            return frame, [], "", False, 0, {}, {}, {}
            
        # Check if frame is valid
        if frame is None or frame.size == 0:
            debug_logger.warning("Invalid frame received for processing")
            return frame, [], "", False, 0, {}, {}, {}
            
        processed_frame, objects, violence_detected, alert_message, processing_time, violence_stats, face_recognition_result, capture_info = detector_instance.process_frame(frame)
        # Store violence alert if detected
        if violence_detected:
            global last_violence_alert
            last_violence_alert = ({
                'message': alert_message,
                'timestamp': time.time(),
                'violence_detected': True,
                'objects': objects,
                'processing_time': processing_time,
                'violence_stats': violence_stats
            })   
            violence_logger.info(f"🚨 VIOLENCE ALERT STORED: {alert_message}")
            
        return processed_frame, objects, alert_message, violence_detected, processing_time, violence_stats, face_recognition_result, capture_info
        
    except Exception as e:
        debug_logger.error(f"Error in process_frame_with_detection: {e}")
        return frame, [], "", False, 0, {}, {}, {}


def video_feed(request):
    """
    Video feed endpoint - returns single frame as JPEG (ultra-simplified and robust)
    """
    try:
        import numpy as np
        
        # Create a simple frame first (always works)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add simple text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "VIOLENCE DETECTION SYSTEM"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        x = (640 - text_size[0]) // 2
        y = (480 + text_size[1]) // 2
        cv2.putText(frame, text, (x, y), font, 1, (0, 255, 0), 2)
        
        # Try to get camera frame if detector is available
        try:
            global detection_active
            detector_instance = get_detector()
            
            if detector_instance is not None:
                camera_frame = detector_instance.get_frame_from_camera()
                if camera_frame is not None:
                    frame = camera_frame
                    
                    # Only process frame if detection is active
                    if detection_active:
                        try:
                            # Debug logging for detection process
                            if hasattr(video_feed, 'debug_count'):
                                video_feed.debug_count += 1
                            else:
                                video_feed.debug_count = 1
                                
                            if video_feed.debug_count % 30 == 0:  # Log every 30 frames
                                print(f"🔍 Processing frame {video_feed.debug_count} for violence detection...")
                            
                            # Process frame directly without the helper function
                            processed_frame, objects, violence_detected, alert_message, processing_time, violence_stats, face_recognition_result, capture_info = detector_instance.process_frame(frame)

                            frame = processed_frame
                            
                            # Debug the detection result
                            if video_feed.debug_count % 30 == 0:
                                print(f"🚨 Detection result: objects={objects}")
                            #violence_detected = True
                            # Store violence alert if detected
                            if objects["confidence"] > 0.7:
                                print(f"🚨 VIOLENCE DETECTED")
                                global last_violence_alert
                                last_violence_alert = {
                                    'message': alert_message,
                                    'timestamp': time.time(),
                                    'violence_detected': True,
                                    'objects': objects,
                                    'processing_time': processing_time,
                                    'violence_stats': violence_stats
                                }
                                violence_logger.info(f"🚨 VIOLENCE ALERT STORED: {alert_message}")
                                print(f"🚨 VIOLENCE DETECTED: {alert_message}")
                                
                                # --- Capture and store this frame in @violence_fream/ directory, max 100 frames ---
                                # Define the directory to store violence frames
                                violence_fream_dir = os.path.join(os.getcwd(), "violence_fream")
                                if not os.path.exists(violence_fream_dir):
                                    os.makedirs(violence_fream_dir)

                                # List all files in the directory (only images)
                                files = [f for f in os.listdir(violence_fream_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                # If more than 100, remove the oldest
                                if len(files) >= 100:
                                    files.sort(key=lambda f: os.path.getmtime(os.path.join(violence_fream_dir, f)))
                                    for old_file in files[:len(files)-99]:
                                        try:
                                            os.remove(os.path.join(violence_fream_dir, old_file))
                                        except Exception as e:
                                            debug_logger.warning(f"Failed to remove old violence frame: {old_file} ({e})")

                                # Save the current frame
                                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                                filename = f"violence_{timestamp_str}.jpg"
                                filepath = os.path.join(violence_fream_dir, filename)
                                try:
                                    cv2.imwrite(filepath, frame)
                                    debug_logger.info(f"Violence frame saved: {filepath}")
                                except Exception as e:
                                    debug_logger.error(f"Failed to save violence frame: {e}")
                            else:
                                # Log when no violence is detected for debugging
                                if hasattr(video_feed, 'frame_count'):
                                    video_feed.frame_count += 1
                                else:
                                    video_feed.frame_count = 1
                                    
                                if video_feed.frame_count % 100 == 0:  # Log every 100 frames
                                    print(f"🔍 Frame {video_feed.frame_count}: No violence detected")
                                
                        except Exception as e:
                            debug_logger.error(f"Error processing frame: {e}")
                            # Use original frame if processing fails
        except Exception as e:
            debug_logger.warning(f"Camera error: {e}")
            # Use default frame if camera fails
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        
        if ret:
            frame_bytes = buffer.tobytes()
            response = HttpResponse(frame_bytes, content_type='image/jpeg')
            response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
            response['Content-Length'] = str(len(frame_bytes))
            return response
        else:
            return HttpResponse("Error encoding frame", status=500)
            
    except Exception as e:
        debug_logger.error(f"Error in video_feed: {e}")
        # Return a simple error response
        return HttpResponse("Video feed error", status=500)


@csrf_exempt
@require_http_methods(["POST"])
def start_detection(request):
    """
    Start the detection system (optimized to prevent stacking)
    """
    global detection_active, detection_thread
    try:
        if detection_active:
            return JsonResponse({
                'success': False,
                'message': 'Detection system is already running'
            })

        # Initialize detector with timeout protection
        detector_instance = get_detector()
        if detector_instance is None:
            return JsonResponse({
                'success': False,
                'message': 'Failed to initialize detector'
            })

        # Test detector before starting
        try:
            test_frame = detector_instance.get_frame_from_camera()
            if test_frame is None:
                debug_logger.warning("Detector initialized but camera not responding")
        except Exception as e:
            debug_logger.warning(f"Camera test failed: {e}")

        detection_active = True
        debug_logger.info("✅ Detection system started successfully")
        print("✅ Detection system started successfully")
        print(f"🔍 Detection status: detection_active={detection_active}, detector_available={detector_instance is not None}")

        return JsonResponse({
            'success': True,
            'message': 'Detection system started successfully'
        })
        
    except Exception as e:
        debug_logger.error(f"❌ Error starting detection: {e}")
        detection_active = False  # Ensure state is clean
        return JsonResponse({
            'success': False,
            'message': f'Error starting detection: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def stop_detection(request):
    """
    Stop the detection system (optimized to prevent stacking)
    """
    global detection_active, detection_thread, last_violence_alert

    try:
        if not detection_active:
            return JsonResponse({
                'success': False,
                'message': 'Detection system is not running'
            })

        # Clean shutdown
        detection_active = False
        last_violence_alert = None  # Clear any pending alerts
        
        # Clear any detection thread if it exists
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=1)  # Wait max 1 second
            detection_thread = None

        debug_logger.info("✅ Detection system stopped successfully")
        print("✅ Detection system stopped successfully")

        return JsonResponse({
            'success': True,
            'message': 'Detection system stopped successfully'
        })
        
    except Exception as e:
        debug_logger.error(f"❌ Error stopping detection: {e}")
        # Force stop even if there's an error
        detection_active = False
        last_violence_alert = None
        return JsonResponse({
            'success': False,
            'message': f'Error stopping detection: {str(e)}'
        })

def detection_status(request):
    """
    Get current detection status
    """
    global detection_active, detector

    status = {
        'active': detection_active,
        'detector_loaded': detector is not None,
        'frame_count': detector.frame_count if detector else 0,
        'violence_alerts': detector.violence_alert_count if detector else 0
    }

    return JsonResponse(status)

def detection_stats(request):
    """
    Get detection statistics
    """
    global detector

    if detector is None:
        return JsonResponse({
            'error': 'Detector not initialized'
        })

    stats = {
        'frame_count': detector.frame_count,
        'violence_alerts': detector.violence_alert_count,
        'detected_objects': dict(detector.detected_objects),
        'detection_history': list(detector.detection_history),
        'is_violence_model': detector.is_violence_model
    }

    return JsonResponse(stats)


@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """
    Health check endpoint to monitor system status
    """
    global detection_active, detector, last_violence_alert
    
    try:
        status = {
            'success': True,
            'timestamp': time.time(),
            'detection_active': detection_active,
            'detector_available': detector is not None,
            'last_alert': last_violence_alert is not None,
            'system_status': 'healthy'
        }
        
        # Check if system is responsive
        if detector:
            try:
                test_frame = detector.get_frame_from_camera()
                status['camera_responding'] = test_frame is not None
            except:
                status['camera_responding'] = False
        else:
            status['camera_responding'] = False
            
        return JsonResponse(status)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'system_status': 'unhealthy'
        })

@csrf_exempt
@require_http_methods(["POST"])
def test_violence_alert(request):
    """
    Test endpoint to simulate a violence alert
    """
    global last_violence_alert
    
    try:
        data = json.loads(request.body)
        test_message = data.get('message', 'Test violence alert!')
        
        last_violence_alert = {
            'message': test_message,
            'timestamp': time.time(),
            'violence_detected': True,
            'objects': ['test_object'],
            'processing_time': 0.1,
            'violence_stats': {'test_count': 1}
        }
        
        print(f"🧪 Test violence alert set: {test_message}")
        violence_logger.info(f"🧪 Test violence alert set: {test_message}")
        print(f"🧪 Test violence alert stored successfully")
        return JsonResponse({
            'success': True,
            'message': f'Test alert set: {test_message}',
            'alert': last_violence_alert
        })
    except Exception as e:
        print(f"❌ Error setting test violence alert: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

# Removed redundant background worker - violence detection happens in video_feed function

@csrf_exempt
@require_http_methods(["POST", "GET"])
def get_violence_alert(request):
    """
    Get the latest violence alert message
    """
    global last_violence_alert

    try:
        # Debug logging
        print(f"🔍 get_violence_alert called - detection_active: {detection_active}, last_violence_alert: {last_violence_alert is not None}")

        # Check if we have a recent violence alert
        if last_violence_alert is not None:
            # Check if alert is recent (within last 10 seconds)
            current_time = time.time()
            time_diff = current_time - last_violence_alert['timestamp']
            print(f"🔍 Alert age: {time_diff:.2f} seconds")

            if time_diff <= 10:
                violence_logger.info(f"🚨 Returning violence alert: {last_violence_alert['message']}")
                print(f"🚨 Returning violence alert: {last_violence_alert['message']}")
                return JsonResponse({
                    'success': True,
                    'alert': last_violence_alert
                })
            else:
                violence_logger.info("⏰ Violence alert expired")
                print("⏰ Violence alert expired")
                # Clear expired alert
                last_violence_alert = None

        # Log occasionally when no alerts
        if hasattr(get_violence_alert, 'call_count'):
            get_violence_alert.call_count += 1
        else:
            get_violence_alert.call_count = 1

        if get_violence_alert.call_count % 50 == 0:  # Log every 50 calls
            violence_logger.info("🔍 No violence alerts available")
            print("🔍 No violence alerts available")

        return JsonResponse({
            'success': False,
            'message': 'No violence alerts'
        })

    except Exception as e:
        violence_logger.error(f"Error in get_violence_alert: {e}")
        print(f"❌ Error in get_violence_alert: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error getting violence alert: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def run_face_recognition(request):
    """
    Run face recognition on the current live ESP32 camera frame
    """
    try:
        system = get_recognition_engine()
        if system is None:
            return JsonResponse({
                'success': False,
                'error': 'Face recognition system not available'
            })
        print("system",system)
        # Use direct live camera feed for face recognition
        result = system.recognize_live_camera_frame()

        
        if result.get('error'):
            return JsonResponse({
                'success': False,
                'result': result           
            })
        
        return JsonResponse({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        face_logger.error(f"Error in face recognition: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Face recognition failed: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def add_camera(request):
    """
    Update camera settings
    """
    global detector

    try:
        data = json.loads(request.body)
        camera_name = data.get('cameraName')
        camera_ip = data.get('cameraIP')
        camera_port = data.get('cameraPort', '')

        # Construct camera URL
        if camera_port:
            camera_url = f"http://{camera_ip}:{camera_port}/cam-hi.jpg"
        else:
            camera_url = f"http://{camera_ip}/cam-hi.jpg"

        # Initialize detector if not already initialized
        detector_instance = get_detector()
        
        if detector_instance and camera_url:
            detector_instance.camera_stream_url = camera_url
            debug_logger.info(f"Camera URL updated to: {camera_url}")

            return JsonResponse({
                'success': True,
                'message': 'Camera added successfully'
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Invalid camera URL or detector not initialized'
            })
    except Exception as e:
        debug_logger.error(f"Error adding camera: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error adding camera: {str(e)}'
        })

def violence_logs(request):
    """
    Get violence detection logs
    """
    global detector

    if detector is None:
        return JsonResponse({
            'logs': [],
            'message': 'Detector not initialized'
        })

    try:
        logs = detector.get_violence_logs()
        # Format logs for frontend
        formatted_logs = []
        for log in logs:
            formatted_logs.append({
                'timestamp': log['timestamp'],
                'time_str': time.strftime('%H:%M:%S', time.localtime(log['timestamp'])),
                'type': log['type'],
                'confidence': log['confidence'],
                'alert_message': log['alert_message'],
                'violence_percentage': log['violence_percentage'],
                'trigger_count': log['trigger_count'],
                'total_violence': log['total_violence'],
                'total_weapons': log['total_weapons'],
                'total_knives': log['total_knives'],
                'frame_count': log['frame_count']
            })

        return JsonResponse({
            'logs': formatted_logs,
            'total_logs': len(formatted_logs)
        })
    except Exception as e:
        debug_logger.error(f"Error getting violence logs: {e}")
        return JsonResponse({
            'logs': [],
            'error': str(e)
        })

@csrf_exempt
@require_http_methods(["POST"])
def clear_violence_logs(request):
    """
    Clear violence detection logs
    """
    global detector

    if detector is None:
        return JsonResponse({
            'success': False,
            'message': 'Detector not initialized'
        })

    try:
        detector.clear_violence_logs()
        violence_logger.info("Violence logs cleared")
        return JsonResponse({
            'success': True,
            'message': 'Violence logs cleared successfully'
        })
    except Exception as e:
        debug_logger.error(f"Error clearing violence logs: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error clearing logs: {str(e)}'
        })

def test_camera_connection(request):
    """
    Test camera connection and return diagnostics
    """
    global detector

    camera_url = request.GET.get('url', 'http://192.168.68.101/cam-hi.jpg')

    if detector is None:
        # Create temporary detector for testing
        try:
            if YOLODetectorAvailable:
                temp_detector = YOLODetector()
                result = temp_detector.test_camera_connection(camera_url)
            else:
                result = {
                    'success': False,
                    'error': 'YOLODetector not available',
                    'url': camera_url
                }
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'url': camera_url
            }
    else:
        result = detector.test_camera_connection(camera_url)

    return JsonResponse(result)

def camera_diagnostics(request):
    """
    Run comprehensive camera diagnostics
    """
    global detector

    if detector is None and YOLODetectorAvailable:
        try:
            detector = YOLODetector()
        except Exception as e:
            return JsonResponse({
                'error': f'Failed to initialize detector: {str(e)}'
            })

    if detector is None:
        return JsonResponse({
            'error': 'Detector not available'
        })

    # Get network info
    import subprocess
    import re

    try:
        result = subprocess.run(['ifconfig'], capture_output=True, text=True)
        network_match = re.search(r'inet (192\.168\.\d+\.\d+)', result.stdout)
        current_ip = network_match.group(1) if network_match else "Unknown"
        network_base = ".".join(current_ip.split(".")[:-1]) if network_match else "192.168.1"
    except Exception:
        current_ip = "Unknown"
        network_base = "192.168.1"

    # Test multiple camera URLs
    alternatives = detector.get_alternative_camera_urls()
    results = []
    working_cameras = []

    debug_logger.info(f"Testing {len(alternatives)} camera URLs...")

    for i, url in enumerate(alternatives[:20]):  # Limit to first 20 to avoid timeout
        debug_logger.info(f"Testing camera {i+1}/20: {url}")
        result = detector.test_camera_connection(url)
        results.append(result)
        if result['success']:
            working_cameras.append(result)
            debug_logger.info(f"✅ Found working camera: {url}")

    return JsonResponse({
        'current_url': detector.camera_stream_url,
        'current_ip': current_ip,
        'network_base': network_base,
        'test_results': results,
        'working_cameras': working_cameras,
        'total_tested': len(results),
        'recommendations': generate_camera_recommendations(results, current_ip, network_base)
    })

def generate_camera_recommendations(test_results, current_ip="Unknown", network_base="192.168.1"):
    """
    Generate recommendations based on test results
    """
    recommendations = []

    successful_urls = [r for r in test_results if r['success']]

    recommendations.append(f"🖥️ Your computer IP: {current_ip}")
    recommendations.append(f"🌐 Your network: {network_base}.x")

    if not successful_urls:
        recommendations.extend([
            "",
            "❌ No ESP32-CAM found on your network",
            "🔧 Troubleshooting steps:",
            "   1. Check ESP32-CAM power supply (5V recommended)",
            "   2. Verify ESP32-CAM WiFi connection",
            "   3. Ensure ESP32-CAM is on same network as computer",
            "   4. Try restarting ESP32-CAM device",
            "   5. Check ESP32-CAM serial monitor for IP address",
            "",
            f"💡 Expected ESP32-CAM IP range: {network_base}.100-120",
            "🔍 Manual check: Try these URLs in browser:",
            f"   • http://{network_base}.101/cam-hi.jpg",
            f"   • http://{network_base}.100/cam-hi.jpg",
            f"   • http://{network_base}.102/cam-hi.jpg"
        ])
    else:
        best_url = min(successful_urls, key=lambda x: x['response_time'])
        recommendations.extend([
            "",
            f"✅ Found {len(successful_urls)} working camera(s)!",
            f"🏆 Best camera: {best_url['url']}",
            f"⚡ Response time: {best_url['response_time']}s",
            f"📊 Image size: {best_url.get('image_size', 'Unknown')} bytes",
            "",
            "🔧 Next steps:",
            "   1. Click 'Update Camera URL' to use this camera",
            "   2. Test the video feed",
            "   3. Start violence detection"
        ])

        if len(successful_urls) > 1:
            recommendations.append("")
            recommendations.append("📋 Other working cameras:")
            for cam in successful_urls[1:]:
                recommendations.append(f"   • {cam['url']} ({cam['response_time']}s)")

    return recommendations

@csrf_exempt
@require_http_methods(["POST"])
def auto_configure_camera(request):
    """
    Automatically find and configure the best camera
    """
    global detector

    if detector is None:
        return JsonResponse({
            'success': False,
            'message': 'Detector not initialized'
        })

    try:
        # Test alternative camera URLs
        alternatives = detector.get_alternative_camera_urls()

        for url in alternatives[:10]:  # Test first 10 URLs
            result = detector.test_camera_connection(url)
            if result['success']:
                # Update camera URL
                detector.camera_stream_url = url
                debug_logger.info(f"Auto-configured camera URL to: {url}")

                return JsonResponse({
                    'success': True,
                    'message': f'Camera auto-configured successfully',
                    'camera_url': url,
                    'response_time': result['response_time']
                })

        return JsonResponse({
            'success': False,
            'message': 'No working cameras found during auto-configuration'
        })

    except Exception as e:
        debug_logger.error(f"Error in auto-configuration: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Auto-configuration failed: {str(e)}'
        })

def stream_status(request):
    """
    Get video stream status
    """
    global detector, detection_active

    status = {
        'detector_available': detector is not None,
        'detection_active': detection_active,
        'camera_url': detector.camera_stream_url if detector else 'Not set',
        'stream_active': True  # Stream is always active now
    }

    return JsonResponse(status)

@csrf_exempt
@require_http_methods(["POST"])
def update_image_quality(request):
    """
    Update image quality settings
    """
    global detector

    if detector is None:
        return JsonResponse({
            'success': False,
            'message': 'Detector not initialized'
        })

    try:
        data = json.loads(request.body)

        # Update quality settings
        enhance_quality = data.get('enhance_quality', True)
        jpeg_quality = data.get('jpeg_quality', 95)

        # Store settings (you can add these as detector attributes)
        detector.enhance_quality = enhance_quality
        detector.jpeg_quality = max(50, min(100, jpeg_quality))  # Clamp between 50-100

        debug_logger.info(f"Image quality updated: enhance={enhance_quality}, jpeg_quality={jpeg_quality}")

        return JsonResponse({
            'success': True,
            'message': 'Image quality settings updated',
            'settings': {
                'enhance_quality': enhance_quality,
                'jpeg_quality': detector.jpeg_quality
            }
        })

    except Exception as e:
        debug_logger.error(f"Error updating image quality: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error updating quality settings: {str(e)}'
        })


def latest_violence_image(request):
    """
    Get information about the latest captured violence image from Milvus database
    """
    try:
        detector_instance = get_detector()
        if not detector_instance:
            return JsonResponse({
                'success': False,
                'message': 'Detector not available'
            })
        
        # First try to get from Milvus database
        latest_frame = detector_instance.get_latest_violence_frame_from_milvus()
        
        if latest_frame:
            image_path = latest_frame.get('image_path')
            if image_path and os.path.exists(image_path):
                stat = os.stat(image_path)
                image_info = {
                    'filename': latest_frame.get('filename'),
                    'path': image_path,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'url': f'/test_images/{latest_frame.get("filename")}',
                    'capture_datetime': latest_frame.get('capture_datetime'),
                    'violence_type': latest_frame.get('violence_type'),
                    'confidence_score': latest_frame.get('confidence_score'),
                    'frame_width': latest_frame.get('frame_width'),
                    'frame_height': latest_frame.get('frame_height'),
                    'detection_model': latest_frame.get('detection_model'),
                    'camera_source': latest_frame.get('camera_source'),
                    'source': 'milvus'
                }
                
                return JsonResponse({
                    'success': True,
                    'image': image_info
                })
        
        # Fallback to filesystem-based method
        from detection_engine.face_recognation import get_latest_captured_image_path
        
        image_path = get_latest_captured_image_path()
        
        if not image_path:
            return JsonResponse({
                'success': False,
                'message': 'No violence images found'
            })
        
        # Get image info from filesystem
        import os
        from django.conf import settings
        
        if os.path.exists(image_path):
            stat = os.stat(image_path)
            image_info = {
                'filename': os.path.basename(image_path),
                'path': image_path,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'url': f'/test_images/{os.path.basename(image_path)}',
                'source': 'filesystem'
            }
            
            return JsonResponse({
                'success': True,
                'image': image_info
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Image file not found'
            })
            
    except Exception as e:
        debug_logger.error(f"Error getting latest violence image: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })

def milvus_stats(request):
    """
    Get comprehensive Milvus database statistics
    """
    try:
        from detection_engine.face_recognation import get_face_recognition_system
        
        # Get comprehensive database statistics
        stats = {
            'available': False,
            'face_recognition_system': False,
            'collections': {},
            'total_entities': 0,
            'database_health': 'Unknown',
            'last_updated': None,
            'storage_info': {},
            'performance_metrics': {},
            'error_log': []
        }
        
        # Try to get face recognition system stats
        try:
            system = get_face_recognition_system()
            if system:
                face_stats = system.get_milvus_stats()
                stats.update(face_stats)
                stats['face_recognition_system'] = True
        except Exception as e:
            stats['error_log'].append(f"Face recognition system error: {str(e)}")
        
        # Try to get detector Milvus stats
        global detector
        if detector and hasattr(detector, 'milvus_manager') and detector.milvus_manager:
            try:
                # Get violence frame collection stats
                try:
                    violence_stats = detector.milvus_manager.get_collection_stats()
                    stats['collections']['violence_frames'] = violence_stats
                except Exception as e:
                    stats['collections']['violence_frames'] = {'entity_count': 0, 'error': str(e)}
                stats['available'] = True
                stats['database_health'] = 'Connected'
                
                # Get recent violence frames
                recent_frames = detector.get_violence_frames_from_milvus(limit=10)
                stats['recent_violence_frames'] = len(recent_frames)
                stats['latest_violence_frame'] = recent_frames[0] if recent_frames else None
                
            except Exception as e:
                stats['error_log'].append(f"Violence frames collection error: {str(e)}")
        else:
            stats['error_log'].append("Detector or Milvus manager not available")
        
        # Add system information
        import os
        
        try:
            import psutil
            stats['system_info'] = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if os.path.exists('/') else 'N/A',
                'process_count': len(psutil.pids())
            }
        except ImportError:
            stats['system_info'] = {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 'N/A',
                'process_count': 0
            }
            stats['error_log'].append("psutil not available - system metrics disabled")
        
        # Add timestamp
        import time
        stats['last_updated'] = time.time()
        stats['last_updated_str'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        return JsonResponse(stats)
        
    except Exception as e:
        debug_logger.error(f"Error getting comprehensive Milvus stats: {e}")
        return JsonResponse({
            'available': False,
            'error': str(e),
            'message': f'Error getting database statistics: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def rebuild_milvus(request):
    """
    Rebuild Milvus database from image files
    """
    try:
        from detection_engine.face_recognation import get_face_recognition_system
        
        system = get_face_recognition_system()
        if not system:
            return JsonResponse({
                'success': False,
                'message': 'Face recognition system not available'
            })
        
        result = system.rebuild_milvus_database()
        return JsonResponse(result)
        
    except Exception as e:
        debug_logger.error(f"Error rebuilding Milvus: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })

def simple_video_feed(request):
    """
    Simple video feed without detection (fallback)
    """
    camera_url = 'http://192.168.68.101/cam-hi.jpg'
    return HttpResponse(
        f"<img src='{camera_url}' alt='Violence Detection' class='video-feed' id='videoFeed' autoplay muted controls>"
    )

def violence_status(request):
    """
    Get current violence detection status
    """
    global detector
    
    if detector is None:
        return JsonResponse({
            'success': False,
            'message': 'Detector not initialized',
            'violence_alert_active': False,
            'current_violence_status': False
        })
    
    try:
        status = detector.get_violence_status()
        return JsonResponse({
            'success': True,
            **status
        })
    except Exception as e:
        debug_logger.error(f"Error getting violence status: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error getting violence status: {str(e)}',
            'violence_alert_active': False,
            'current_violence_status': False
        })

def get_latest_violence_frame(request):
    """
    Get the latest captured violence frame
    """
    import os
    from datetime import datetime
    from django.http import JsonResponse
    from django.conf import settings
    
    global detector
    
    if detector is None:
        return JsonResponse({
            'success': False,
            'message': 'Detector not initialized'
        })
    
    try:
        
        # Get the latest captured image path
        latest_image_path = detector.get_latest_captured_image_path()
        
        if latest_image_path and os.path.exists(latest_image_path):
            # Get relative path for URL
            image_filename = os.path.basename(latest_image_path)
            
            # Get image info from captured images list
            image_info = None
            for img in reversed(detector.captured_images):
                if img['path'] == latest_image_path:
                    image_info = {
                        'filename': image_filename,
                        'timestamp': img['timestamp'],
                        'violence_type': img['violence_type'],
                        'confidence': img['confidence'],
                        'capture_time_str': datetime.fromtimestamp(img['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    break
            
            if not image_info:
                # Fallback info from filename if not in captured_images
                image_info = {
                    'filename': image_filename,
                    'timestamp': os.path.getmtime(latest_image_path),
                    'violence_type': 'unknown',
                    'confidence': 0.0,
                    'capture_time_str': datetime.fromtimestamp(os.path.getmtime(latest_image_path)).strftime('%Y-%m-%d %H:%M:%S')
                }
            
            return JsonResponse({
                'success': True,
                'has_image': True,
                'image_path': f'/test_images/{image_filename}',
                'image_info': image_info,
                'total_captured': len(detector.captured_images)
            })
        else:
            return JsonResponse({
                'success': True,
                'has_image': False,
                'message': 'No violence frames captured yet',
                'total_captured': len(detector.captured_images) if hasattr(detector, 'captured_images') else 0
            })
            
    except Exception as e:
        debug_logger.error(f"Error getting latest violence frame: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error getting latest violence frame: {str(e)}'
        })

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

@csrf_exempt
def clear_test_images(request):
    """
    Clear all images from test_images directory (refresh functionality)
    """
    global detector
    
    if detector is None:
        return JsonResponse({
            'success': False,
            'message': 'Detector not initialized'
        })
    
    try:
        success = detector.clear_all_test_images()
        
        if success:
            return JsonResponse({
                'success': True,
                'message': 'All test images cleared successfully',
                'total_captured': 0
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Failed to clear test images'
            })
            
    except Exception as e:
        debug_logger.error(f"Error clearing test images: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error clearing test images: {str(e)}'
        })

def serve_test_image(request, filename):
    """
    Serve test images from the test_images directory
    """
    try:
        # Build the full path to the test image
        test_images_dir = os.path.join(settings.BASE_DIR, 'test_images')
        file_path = os.path.join(test_images_dir, filename)
        
        # Security check: ensure the file is within test_images directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(test_images_dir)):
            raise Http404("File not found")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise Http404("File not found")
        
        # Check if it's an image file
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
            raise Http404("Invalid file type")
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = 'image/jpeg'
        
        # Return the file
        response = FileResponse(
            open(file_path, 'rb'),
            content_type=content_type
        )
        
        # Add cache control headers for better performance
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        
        return response
        
    except Exception as e:
        debug_logger.error(f"Error serving test image {filename}: {e}")
        raise Http404("File not found")

@csrf_exempt
@require_http_methods(["POST"])
def capture_image(request):
    """
    Handle image capture from webcam and save to captured_db folder
    """
    try:
        # Parse JSON data
        data = json.loads(request.body)
        image_data = data.get('image')
        
        if not image_data:
            return JsonResponse({'error': 'No image data provided'}, status=400)
        
        # Remove data URL prefix (data:image/png;base64,)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Create captured_db directory if it doesn't exist
        captured_dir = os.path.join(settings.MEDIA_ROOT, 'captured_db')
        os.makedirs(captured_dir, exist_ok=True)
        
        # Generate unique filename
        import time
        timestamp = int(time.time() * 1000)  # milliseconds
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(captured_dir, filename)
        
        # Convert to PIL Image and save as JPEG
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_bytes))
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(filepath, 'JPEG', quality=85)
        
        return JsonResponse({
            'success': True, 
            'message': 'Image captured successfully',
            'filename': filename
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
