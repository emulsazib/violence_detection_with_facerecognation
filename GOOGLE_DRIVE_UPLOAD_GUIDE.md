# Google Drive Upload Guide for Model Files

This guide explains how to upload the model files to Google Drive and share them with users.

## 📦 Files to Upload

You need to upload the following model files from your source project to Google Drive:

### Required Files:

1. **yolo11n.pt** (5.4 MB)
   - Location: `/Users/sajibmacmini/Downloads/automate_violence_detection/violence_detection/models/yolo11n.pt`

2. **yolo11s.pt** (18 MB)
   - Location: `/Users/sajibmacmini/Downloads/automate_violence_detection/violence_detection/models/yolo11s.pt`

3. **yolo11n-pose.pt** (6.0 MB)
   - Location: `/Users/sajibmacmini/Downloads/automate_violence_detection/violence_detection/models/yolo11n-pose.pt`

4. **best.pt** (5.2 MB) - MOST IMPORTANT
   - Location: `/Users/sajibmacmini/Downloads/automate_violence_detection/violence_detection/models/dataset/violence_detection_run/weights/best.pt`

5. **last.pt** (5.2 MB) - Optional checkpoint
   - Location: `/Users/sajibmacmini/Downloads/automate_violence_detection/violence_detection/models/dataset/violence_detection_run/weights/last.pt`

## 📋 Step-by-Step Upload Process

### Option 1: Upload as a ZIP File (Recommended)

#### Step 1: Create ZIP Archive

On macOS/Linux:
```bash
cd /Users/sajibmacmini/Downloads/automate_violence_detection/violence_detection/models

# Create a zip file with all models
zip -r violence_detection_models.zip \
  yolo11n.pt \
  yolo11n-pose.pt \
  yolo11s.pt \
  dataset/violence_detection_run/weights/best.pt \
  dataset/violence_detection_run/weights/last.pt
```

On Windows (PowerShell):
```powershell
Compress-Archive -Path @(
  "yolo11n.pt",
  "yolo11n-pose.pt",
  "yolo11s.pt",
  "dataset\violence_detection_run\weights\best.pt",
  "dataset\violence_detection_run\weights\last.pt"
) -DestinationPath "violence_detection_models.zip"
```

#### Step 2: Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Click **New** → **File upload**
3. Select `violence_detection_models.zip`
4. Wait for upload to complete (total ~40 MB)

#### Step 3: Get Shareable Link

1. Right-click on `violence_detection_models.zip`
2. Click **Share** or **Get link**
3. Set permissions to **Anyone with the link** → **Viewer**
4. Click **Copy link**

#### Step 4: Update Repository Documentation

Update the following files with your Google Drive link:

1. **README.md** (Line ~75):
   ```markdown
   📥 [Download All Models (ZIP)](YOUR_GOOGLE_DRIVE_LINK_HERE)
   ```

2. **MODELS_README.md** (Line ~22):
   ```markdown
   🔗 **Google Drive Link**: [Download Models Here](YOUR_GOOGLE_DRIVE_SHARE_LINK)
   ```

3. **QUICKSTART.md** (Line ~52):
   ```markdown
   1. Go to: [Google Drive Models Link](YOUR_GOOGLE_DRIVE_LINK)
   ```

### Option 2: Upload Individual Files

If you prefer to upload files individually:

#### Step 1: Create a Folder

1. Go to Google Drive
2. Click **New** → **Folder**
3. Name it: `violence_detection_models`

#### Step 2: Upload Each File

Upload each model file to the folder:
- yolo11n.pt
- yolo11n-pose.pt
- yolo11s.pt
- best.pt
- last.pt

#### Step 3: Share the Folder

1. Right-click on the folder
2. Click **Share** or **Get link**
3. Set to **Anyone with the link** → **Viewer**
4. Copy the link

#### Step 4: Get Individual File IDs (Optional)

For automation with `gdown`:

1. Right-click each file → **Get link**
2. Copy the file ID from the URL
3. Update MODELS_README.md with file IDs for gdown commands

Example URL:
```
https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
```

## 🔄 Alternative: Use GitHub Releases (for smaller models)

For models under GitHub's size limits:

1. Go to your GitHub repository
2. Click **Releases** → **Create a new release**
3. Tag version (e.g., `v1.0-models`)
4. Upload model files as release assets
5. Update documentation with release links

**Note:** GitHub has a 2GB limit per file and 100MB recommended size.

## 📝 Update Documentation Checklist

After uploading to Google Drive, update these files:

- [ ] README.md - Main download link
- [ ] MODELS_README.md - Detailed download instructions
- [ ] QUICKSTART.md - Quick start guide link
- [ ] verify_models.py - Error message link (line ~)

## 🧪 Test the Download

Before pushing to GitHub:

1. Download the file from your Google Drive link
2. Extract the files
3. Place them in the models/ directory
4. Run `python verify_models.py`
5. Ensure all models are detected correctly

## 📊 Expected File Sizes

Verify file sizes match:

```
yolo11n.pt:      5.4 MB  (5,361,104 bytes)
yolo11n-pose.pt: 6.0 MB  (6,036,808 bytes)
yolo11s.pt:     18.0 MB (18,887,840 bytes)
best.pt:         5.2 MB  (5,215,136 bytes)
last.pt:         5.2 MB  (5,215,136 bytes)
-------------------------------------------
Total ZIP:      ~40 MB
```

## 🔐 Security Notes

1. **Do NOT upload:**
   - Personal face recognition images
   - Database files with user data
   - Configuration files with secrets
   - Log files

2. **Set proper permissions:**
   - Use "Anyone with the link" (not public)
   - Set to "Viewer" only (not editor)

3. **Keep backup:**
   - Keep original model files as backup
   - Consider multiple storage locations

## 📧 Communication

After uploading, you may want to:

1. Create a pinned issue on GitHub with the download link
2. Add download instructions to repository description
3. Include link in README badges section

## 🔄 Updating Models

When you update models:

1. Upload new version to Google Drive
2. Update version number in documentation
3. Update CHANGELOG.md with changes
4. Notify users of updates

## 📞 Support

If users have trouble downloading:

1. Check link permissions
2. Verify link is not expired
3. Consider mirrors (Dropbox, OneDrive, etc.)
4. Provide direct support via GitHub issues

---

## ✅ Final Checklist

Before committing to GitHub:

- [ ] Models uploaded to Google Drive
- [ ] Shareable link obtained
- [ ] README.md updated with link
- [ ] MODELS_README.md updated with link
- [ ] QUICKSTART.md updated with link
- [ ] verify_models.py updated with link
- [ ] Tested download from link
- [ ] Verified extraction works
- [ ] Confirmed models load correctly
- [ ] .gitignore excludes model files
- [ ] Documentation is clear and complete

---

**Your Google Drive Link:**
```
https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing
```

Replace all instances of `YOUR_GOOGLE_DRIVE_LINK` in the documentation with this link!

