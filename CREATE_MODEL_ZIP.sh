#!/bin/bash

# Script to create model files ZIP for Google Drive upload
# Run this script from the project root directory

set -e

echo "================================================"
echo "Creating Model Files ZIP for Google Drive Upload"
echo "================================================"
echo ""

# Define source directory (original project location)
SOURCE_DIR="/Users/sajibmacmini/Downloads/automate_violence_detection/violence_detection/models"

# Define output file
OUTPUT_FILE="violence_detection_models.zip"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Source directory not found: $SOURCE_DIR"
    echo "Please update SOURCE_DIR in this script to point to your original project location"
    exit 1
fi

echo "📂 Source directory: $SOURCE_DIR"
echo "📦 Output file: $OUTPUT_FILE"
echo ""

# Create a temporary directory for organizing files
TEMP_DIR=$(mktemp -d)
echo "📁 Creating temporary directory: $TEMP_DIR"

# Copy model files to temp directory
echo ""
echo "📋 Copying model files..."

# Base YOLO models
echo "  - Copying yolo11n.pt..."
cp "$SOURCE_DIR/yolo11n.pt" "$TEMP_DIR/" 2>/dev/null || echo "    ⚠️  File not found"

echo "  - Copying yolo11n-pose.pt..."
cp "$SOURCE_DIR/yolo11n-pose.pt" "$TEMP_DIR/" 2>/dev/null || echo "    ⚠️  File not found"

echo "  - Copying yolo11s.pt..."
cp "$SOURCE_DIR/yolo11s.pt" "$TEMP_DIR/" 2>/dev/null || echo "    ⚠️  File not found"

# Violence detection models
mkdir -p "$TEMP_DIR/weights"

echo "  - Copying best.pt..."
cp "$SOURCE_DIR/dataset/violence_detection_run/weights/best.pt" "$TEMP_DIR/weights/" 2>/dev/null || echo "    ⚠️  File not found"

echo "  - Copying last.pt..."
cp "$SOURCE_DIR/dataset/violence_detection_run/weights/last.pt" "$TEMP_DIR/weights/" 2>/dev/null || echo "    ⚠️  File not found"

echo ""
echo "✅ Files copied to temporary directory"

# List files with sizes
echo ""
echo "📊 Files to be included in ZIP:"
echo ""
ls -lh "$TEMP_DIR/" | tail -n +2
if [ -d "$TEMP_DIR/weights" ]; then
    echo ""
    echo "Weights directory:"
    ls -lh "$TEMP_DIR/weights/" | tail -n +2
fi

# Calculate total size
TOTAL_SIZE=$(du -sh "$TEMP_DIR" | cut -f1)
echo ""
echo "📏 Total size: $TOTAL_SIZE"

# Create README for the ZIP
echo ""
echo "📝 Creating README.txt for the ZIP..."
cat > "$TEMP_DIR/README.txt" << 'EOF'
# Violence Detection System - Model Files

This ZIP file contains the pre-trained model files for the Violence Detection System.

## Files Included:

1. yolo11n.pt (5.4 MB)
   - YOLO11 Nano base model
   - Place in: models/

2. yolo11n-pose.pt (6.0 MB)
   - YOLO11 Nano Pose model (optional)
   - Place in: models/

3. yolo11s.pt (18 MB)
   - YOLO11 Small base model
   - Place in: models/

4. weights/best.pt (5.2 MB)
   - Custom trained violence detection model (REQUIRED)
   - Place in: models/dataset/violence_detection_run/weights/

5. weights/last.pt (5.2 MB)
   - Last training checkpoint (optional)
   - Place in: models/dataset/violence_detection_run/weights/

## Installation Instructions:

1. Extract this ZIP file
2. Place model files in the correct directories (see above)
3. Run: python verify_models.py
4. If all checks pass, you're ready to go!

For detailed setup instructions, see:
https://github.com/YOUR_USERNAME/violence_detection_with_facerecognation

## Quick Setup:

```bash
# Extract ZIP
unzip violence_detection_models.zip -d /tmp/models

# Navigate to your project
cd violence_detection_with_facerecognation

# Copy base models
cp /tmp/models/yolo11n.pt models/
cp /tmp/models/yolo11n-pose.pt models/
cp /tmp/models/yolo11s.pt models/

# Copy violence detection models
mkdir -p models/dataset/violence_detection_run/weights
cp /tmp/models/weights/best.pt models/dataset/violence_detection_run/weights/
cp /tmp/models/weights/last.pt models/dataset/violence_detection_run/weights/

# Verify installation
python verify_models.py
```

## Support:

For issues or questions, please open an issue on GitHub:
https://github.com/YOUR_USERNAME/violence_detection_with_facerecognation/issues

---
Generated: $(date)
Version: 1.0
EOF

echo "✅ README.txt created"

# Create the ZIP file
echo ""
echo "🗜️  Creating ZIP file..."
cd "$TEMP_DIR"
zip -r -q "$OLDPWD/$OUTPUT_FILE" .

# Get back to original directory
cd "$OLDPWD"

# Clean up temp directory
echo "🧹 Cleaning up temporary directory..."
rm -rf "$TEMP_DIR"

# Get ZIP file size
ZIP_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "================================================"
echo "✅ SUCCESS!"
echo "================================================"
echo ""
echo "📦 ZIP file created: $OUTPUT_FILE"
echo "📏 File size: $ZIP_SIZE"
echo ""
echo "Next steps:"
echo "1. Upload $OUTPUT_FILE to Google Drive"
echo "2. Set sharing permissions to 'Anyone with the link'"
echo "3. Copy the shareable link"
echo "4. Update the following files with the Google Drive link:"
echo "   - README.md"
echo "   - MODELS_README.md"
echo "   - QUICKSTART.md"
echo "   - verify_models.py"
echo ""
echo "For detailed upload instructions, see:"
echo "GOOGLE_DRIVE_UPLOAD_GUIDE.md"
echo ""

