#!/usr/bin/env python3
"""
Model Verification Script for Violence Detection System
This script checks if all required model files are properly installed.
"""

import os
from pathlib import Path
import sys


def verify_models():
    """Verify all required model files are present"""
    
    project_root = Path(__file__).resolve().parent
    
    required_models = {
        'YOLO11 Nano': project_root / 'models' / 'yolo11n.pt',
        'YOLO11 Small': project_root / 'models' / 'yolo11s.pt',
        'Violence Detection (best.pt)': project_root / 'models' / 'dataset' / 'violence_detection_run' / 'weights' / 'best.pt'
    }
    
    optional_models = {
        'YOLO11 Nano Pose': project_root / 'models' / 'yolo11n-pose.pt',
        'Training Checkpoint (last.pt)': project_root / 'models' / 'dataset' / 'violence_detection_run' / 'weights' / 'last.pt'
    }
    
    print("=" * 70)
    print("🔍 Violence Detection System - Model Verification")
    print("=" * 70)
    print()
    
    # Check required models
    print("📋 Checking REQUIRED model files...\n")
    
    all_present = True
    missing_models = []
    
    for name, path in required_models.items():
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"  ✅ {name:<35} Found ({size:.1f} MB)")
        else:
            print(f"  ❌ {name:<35} NOT FOUND")
            print(f"     Expected at: {path}")
            all_present = False
            missing_models.append(name)
    
    # Check optional models
    print("\n📋 Checking OPTIONAL model files...\n")
    
    for name, path in optional_models.items():
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name:<35} Found ({size:.1f} MB)")
        else:
            print(f"  ⚠️  {name:<35} Not found (optional)")
    
    # Summary
    print("\n" + "=" * 70)
    
    if all_present:
        print("✅ SUCCESS: All required models are installed!")
        print()
        print("🚀 You're ready to run the application!")
        print()
        print("To start the server, run:")
        print("  python manage.py runserver")
        print("=" * 70)
        return True
    else:
        print("❌ ERROR: Some required models are missing!")
        print()
        print("Missing models:")
        for model in missing_models:
            print(f"  - {model}")
        print()
        print("📥 Please download the models from Google Drive:")
        print("   🔗 See MODELS_README.md for download instructions")
        print()
        print("Quick setup:")
        print("  1. Download models.zip from Google Drive link in MODELS_README.md")
        print("  2. Extract and place files in the models/ directory")
        print("  3. Run this script again to verify")
        print("=" * 70)
        return False


def check_directories():
    """Check if necessary directories exist"""
    project_root = Path(__file__).resolve().parent
    
    required_dirs = [
        project_root / 'models' / 'dataset' / 'violence_detection_run' / 'weights',
        project_root / 'logs',
        project_root / 'media' / 'captured_db',
        project_root / 'media' / 'face_db',
        project_root / 'media' / 'results'
    ]
    
    print("\n📂 Checking required directories...\n")
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  ✅ {dir_path.relative_to(project_root)}")
        else:
            print(f"  ❌ {dir_path.relative_to(project_root)} - Creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"     Created: {dir_path.relative_to(project_root)}")
    
    return all_exist


def main():
    """Main verification function"""
    print()
    
    # Check directories first
    check_directories()
    
    print()
    
    # Verify models
    success = verify_models()
    
    print()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

