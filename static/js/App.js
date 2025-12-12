// Global variables for the application
let detectionActive = false;
let lastAlertCheck = 0;
let lastAlertMessage = '';
let videoImg = null;
let videoStreamInterval = null;
let alertCheckInterval = null;
let isVideoStreamActive = false;
let isAlertCheckActive = false;
const ALERT_CHECK_INTERVAL = 3000; // Check every 3 seconds when detection is active
const VIDEO_UPDATE_INTERVAL = 500; // Update video every 500ms for smoother streaming (2 FPS)
// A variable to store the count for our specific log message.
let violenceAlertsCheckCount = 0;

// Function to check for violence alerts (optimized)
function checkViolenceAlerts() {
    if (!isAlertCheckActive) {
        return;
    }
    
    const now = Date.now();
    
    // Only check if detection is active and enough time has passed
    if (!detectionActive) {
        return;
    }
    
    if ((now - lastAlertCheck) < ALERT_CHECK_INTERVAL) {
        return;
    }
    
    lastAlertCheck = now;

    console.log('🔍 Checking for violence alerts...');

    // Use our new counting function instead of the original console.log
    //logAndCountViolenceAlerts();
  
    
    const csrfToken = getCookie('csrftoken');
    //const csrfToken = document.getElementById('csrfToken').value;
    
    fetch('/get_violence_alert/', {
        method: 'GET', 
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
    }).then(response => {
        console.log('📡 Violence alert response status:', response.status);
        return response.json();
    })
      .then(data => {
        //console.log('📡 Violence alert response data:', data);
        
        // Check if we have a successful response with alert data
        if (data.success && data.alert) {
            console.log('🚨 VIOLENCE DETECTED:', data.alert.message);
            console.log('📊 Alert details:', data.alert);
            
            // Prevent duplicate alerts
            if (data.alert.message !== lastAlertMessage) {
                // Show alert to user
                if (data.alert.violence_detected) {
                    alert('🚨 VIOLENCE DETECTED: ' + data.alert.message);
                    lastAlertMessage = data.alert.message;
                }
            } else {
                console.log('🔄 Duplicate alert ignored:', data.alert.message);
            }
        } else {
            // Only log occasionally when no alerts
            if (Math.random() < 0.1) { // Log 10% of the time
                console.log('✅ No violence alerts detected - Response:', data.message || 'No message');
            }
        }
    })
    .catch(error => {
        console.error('❌ Error checking violence alerts:', error);
    });
}

// A wrapper function to count and log the message.
const logAndCountViolenceAlerts = () => {
    //console.log('🔍 Checking for violence alerts...', violenceAlertsCheckCount);
    violenceAlertsCheckCount++;
     
};

// Function to start alert checking
function startAlertChecking() {
    if (isAlertCheckActive) {
        return;
    }
    
    console.log('🔍 Starting alert checking');
    isAlertCheckActive = true;
    
    if (alertCheckInterval) {
        clearInterval(alertCheckInterval);
    }
    alertCheckInterval = setInterval(checkViolenceAlerts, ALERT_CHECK_INTERVAL);
}

// Function to stop alert checking
function stopAlertChecking() {
    if (!isAlertCheckActive) {
        return;
    }
    
    console.log('🔍 Stopping alert checking');
    isAlertCheckActive = false;
    
    if (alertCheckInterval) {
        clearInterval(alertCheckInterval);
        alertCheckInterval = null;
    }
}

// This script displays live camera feed using image refresh
document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const cameraIP = document.getElementById('cameraIP');
    
    console.log('🚀 Violence Detection App initialized');
    
    // Function to check server connectivity
    function checkServerConnectivity() {
        return fetch('/health_check/')
            .then(response => response.ok)
            .catch(() => false);
    }
    
    // Function to start video stream using image refresh
    function startVideoStream() {
        if (isVideoStreamActive) {
            console.log('📹 Video stream already active, skipping restart');
            return;
        }
        
        console.log('📹 Starting video stream using image refresh');
        
        // Check server connectivity first
        checkServerConnectivity().then(serverAvailable => {
            if (!serverAvailable) {
                console.warn('⚠️ Server not available, will retry video stream when server is ready');
                setTimeout(startVideoStream, 5000); // Retry in 5 seconds
                return;
            }
            
            isVideoStreamActive = true;
            
            // Create an img element for the video feed
            videoImg = document.getElementById('video-img');
            if (!videoImg) {
                videoImg = document.createElement('img');
                videoImg.id = 'video-img';
                videoImg.style.width = '100%';
                videoImg.style.height = '100%';
                videoImg.style.objectFit = 'cover';
                video.parentNode.replaceChild(videoImg, video);
            }
            
            // Function to update the image
            function updateImage() {
                if (!isVideoStreamActive) {
                    return; // Stop if video stream is disabled
                }
                
                const timestamp = new Date().getTime();
                videoImg.src = `/video_feed/?t=${timestamp}`;
                
                videoImg.onload = function() {
                    // Only log occasionally to avoid spam
                    if (Math.random() < 0.05) { // Log 5% of the time
                        console.log('✅ Video frame loaded successfully');
                    }
                };
                
                videoImg.onerror = function(e) {
                    // Only log errors occasionally to avoid spam
                    if (Math.random() < 0.1) { // Log 10% of errors
                        console.warn('⚠️ Video frame loading error (retrying in 5s)');
                    }
                    
                    // Retry after 5 seconds to prevent overwhelming the server
                    setTimeout(() => {
                        if (isVideoStreamActive) {
                            updateImage();
                        }
                    }, 5000);
                };
            }
            
            // Start updating the image
            updateImage();
            
            // Update image at a reasonable interval for smooth video
            if (videoStreamInterval) {
                clearInterval(videoStreamInterval);
            }
            videoStreamInterval = setInterval(updateImage, VIDEO_UPDATE_INTERVAL);
            
            console.log('📹 Video stream started with', VIDEO_UPDATE_INTERVAL + 'ms interval');
        });
    }
    
    // Function to stop video stream
    function stopVideoStream() {
        if (!isVideoStreamActive) {
            return;
        }
        
        console.log('📹 Stopping video stream');
        isVideoStreamActive = false;
        
        if (videoStreamInterval) {
            clearInterval(videoStreamInterval);
            videoStreamInterval = null;
        }
        
        if (videoImg) {
            videoImg.src = '';
        }
    }
    
    
    
    // Start the video stream
    startVideoStream();
    
    // Update video stream when camera IP changes (after adding camera)
    if (cameraIP) {
        cameraIP.addEventListener('change', function() {
            console.log('📹 Camera IP changed, refreshing video stream...');
            stopVideoStream();
            setTimeout(() => {
                startVideoStream();
            }, 1000);
        });
    }
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        stopVideoStream();
        stopAlertChecking();
    });
    
    console.log('🔍 Alert checking started with', ALERT_CHECK_INTERVAL + 'ms interval');
});

function startApplication() {
    const startBtn = document.getElementById('startBtn');
    startBtn.innerHTML = 'Starting...';
    startBtn.disabled = true;

    const csrfToken = document.getElementById('csrfToken').value;
    console.log('🚀 Starting violence detection application...');
    
    fetch('/start_detection/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
    }).then(response => response.json()).then(data => {
        console.log('📡 Start detection response:', data);
        if (data.success) {
            detectionActive = true;
            startAlertChecking();
            console.log('✅ Detection started - violence alerts enabled');
            alert('✅ Violence detection started successfully!');
        } else {
            console.error('❌ Failed to start detection:', data.message);
            alert('❌ Failed to start detection: ' + data.message);
        }
    }).catch(error => {
        console.error('❌ Error starting detection:', error);
        alert('❌ Error starting detection: ' + error.message);
    }).finally(() => {
        startBtn.innerHTML = 'Start Application';
        startBtn.disabled = false;
    });
}

function stopApplication() {
    const stopBtn = document.getElementById('stopBtn');
    stopBtn.innerHTML = 'Stopping...';
    stopBtn.disabled = true;

    const csrfToken = document.getElementById('csrfToken').value;
    console.log('🛑 Stopping violence detection application...');
    
    fetch('/stop_detection/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
    }).then(response => response.json()).then(data => {
        console.log('📡 Stop detection response:', data);
        if (data.success) {
            detectionActive = false;
            stopAlertChecking();
            console.log('✅ Detection stopped - violence alerts disabled');
            alert('✅ Violence detection stopped successfully!');
        } else {
            console.error('❌ Failed to stop detection:', data.message);
            alert('❌ Failed to stop detection: ' + data.message);
        }
    }).catch(error => {
        console.error('❌ Error stopping detection:', error);
        alert('❌ Error stopping detection: ' + error.message);
    }).finally(() => {
        stopBtn.innerHTML = 'Stop Application';
        stopBtn.disabled = false;
    });
}

function addCamera() {
    const cameraName = document.getElementById('cameraName').value;
    const cameraIP = document.getElementById('cameraIP').value;
    const cameraPort = document.getElementById('cameraPort').value;
    const csrfToken = document.getElementById('csrfToken').value;
    
    console.log('📹 Adding camera:', { cameraName, cameraIP, cameraPort });
    
    fetch('/add_camera/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ cameraName: cameraName, cameraIP: cameraIP, cameraPort: cameraPort }),
    }).then(response => response.json()).then(data => {
        console.log('📡 Add camera response:', data);
        if (data.success) {
            // Update video stream with new camera
            updateVideoStream();
            alert('✅ Camera added successfully! Video stream updated.');
        } else {
            alert('❌ Failed to add camera: ' + data.message);
        }
    }).catch(error => {
        console.error('❌ Error adding camera:', error);
        alert('❌ Error adding camera. Please check console for details.');
    });
}

// Function to update video stream
function updateVideoStream() {
    const video = document.getElementById('video');
    
    if (!video) {
        console.error('❌ Video element not found');
        return;
    }
    
    console.log('🔄 Updating video stream from Django video feed');
    
    // Use Django video feed endpoint which handles camera connection
    video.src = '/video_feed/';
    video.load();
}

// Helper function to get CSRF token from cookies (for Django)
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Missing functions to prevent errors
function refreshCameras() {
    console.log('🔄 refreshCameras - function not implemented yet');
    alert('Refresh Cameras function not implemented yet');
}

function testCameraConnection() {
    console.log('🔍 testCameraConnection - function not implemented yet');
    alert('Test Camera Connection function not implemented yet');
}

function runCameraDiagnostics() {
    console.log('🔧 runCameraDiagnostics - function not implemented yet');
    alert('Camera Diagnostics function not implemented yet');
}

function autoConfigureCamera() {
    console.log('⚙️ autoConfigureCamera - function not implemented yet');
    alert('Auto-Configure Camera function not implemented yet');
}

function fullscreenVideo() {
    console.log('📺 fullscreenVideo - function not implemented yet');
    alert('Fullscreen Video function not implemented yet');
}

function refreshViolenceLogs() {
    console.log('📋 refreshViolenceLogs - function not implemented yet');
    alert('Refresh Violence Logs function not implemented yet');
}

function clearViolenceLogs() {
    console.log('🗑️ clearViolenceLogs - function not implemented yet');
    alert('Clear Violence Logs function not implemented yet');
}

function runFaceRecognition() {
    const runFaceRecognitionBtn = document.getElementById('runFaceRecognitionBtn');
    if (!runFaceRecognitionBtn) {
        console.error('❌ runFaceRecognitionBtn element not found');
        alert('Face recognition button not found');
        return;
    }

    runFaceRecognitionBtn.innerHTML = 'Running...';
    runFaceRecognitionBtn.disabled = true;
    
    const csrfToken = document.getElementById('csrfToken').value;
    console.log('🔍 Running face recognition...');
    
    fetch('/run_face_recognition/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
    }).then(response => response.json()).then(data => {
        console.log('🔍 Face recognition result:', data);
        if (data.error) {
            alert('❌ Face recognition error: ' + data.error);
        } else {
            alert('✅ Face recognition completed successfully!');
        }
    }).catch(error => {
        console.error('❌ Face recognition error:', error);
        alert('❌ Error running face recognition: ' + error.message);
    }).finally(() => {
        runFaceRecognitionBtn.innerHTML = 'Run Face Recognition';
        runFaceRecognitionBtn.disabled = false;
    });
}

function refreshEvidence() {
    console.log('🔄 refreshEvidence - function not implemented yet');
    alert('Refresh Evidence function not implemented yet');
}

function getMilvusStats() {
    console.log('📊 getMilvusStats - function not implemented yet');
    alert('Get Milvus Stats function not implemented yet');
}

function rebuildMilvus() {
    console.log('🔧 rebuildMilvus - function not implemented yet');
    alert('Rebuild Milvus function not implemented yet');
}

function clearAllTestImages() {
    console.log('🗑️ clearAllTestImages - function not implemented yet');
    alert('Clear All Test Images function not implemented yet');
}

function loadLatestViolenceFrame() {
    console.log('📸 loadLatestViolenceFrame - function not implemented yet');
    alert('Load Latest Violence Frame function not implemented yet');
}

function testViolenceAlert() {
    const csrfToken = getCookie('csrftoken');
    console.log('🧪 Testing violence alert system...');
    
    fetch('/test_violence_alert/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ message: 'Test violence detection - Knife detected!' })
    }).then(response => response.json())
      .then(data => {
        console.log('🧪 Test violence alert response:', data);
        if (data.success) {
            alert('✅ Test violence alert set! Check console for detection.');
        } else {
            alert('❌ Failed to set test alert: ' + data.error);
        }
    }).catch(error => {
        console.error('❌ Error testing violence alert:', error);
        alert('❌ Error testing violence alert: ' + error.message);
    });
}