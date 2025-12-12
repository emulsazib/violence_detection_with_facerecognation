# Contributing to Violence Detection System

Thank you for considering contributing to the Violence Detection System! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

1. **Clear title**: Describe the bug concisely
2. **Description**: Detailed description of the issue
3. **Steps to reproduce**: Step-by-step instructions
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Environment**: OS, Python version, GPU/CPU
7. **Screenshots**: If applicable
8. **Error logs**: Include relevant error messages

### Suggesting Enhancements

For feature requests or enhancements:

1. Check existing issues to avoid duplicates
2. Clearly describe the feature and its benefits
3. Provide examples or mockups if possible
4. Explain why this feature would be useful

### Pull Requests

We actively welcome pull requests! Here's the process:

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/violence_detection_with_facerecognation.git
cd violence_detection_with_facerecognation
```

#### 2. Create a Branch

```bash
# Create a new branch for your feature/fix
git checkout -b feature/amazing-feature
# Or for bug fixes
git checkout -b fix/bug-description
```

#### 3. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model files (see MODELS_README.md)
```

#### 4. Make Your Changes

- Write clean, readable code
- Follow Python PEP 8 style guidelines
- Add comments for complex logic
- Update documentation if needed
- Add tests for new features

#### 5. Test Your Changes

```bash
# Run the application
python manage.py runserver

# Test manually
# Add automated tests if applicable
```

#### 6. Commit Your Changes

```bash
# Add your changes
git add .

# Commit with a descriptive message
git commit -m "Add feature: description of feature"

# Use conventional commit messages:
# feat: New feature
# fix: Bug fix
# docs: Documentation changes
# style: Code style changes (formatting)
# refactor: Code refactoring
# test: Adding tests
# chore: Maintenance tasks
```

#### 7. Push and Create PR

```bash
# Push to your fork
git push origin feature/amazing-feature

# Go to GitHub and create a Pull Request
```

### Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what changes you made and why
- **Link issues**: Reference related issues (Fixes #123)
- **Screenshots**: Include for UI changes
- **Testing**: Describe how you tested your changes
- **Documentation**: Update README.md or other docs if needed

## 📝 Coding Standards

### Python Code Style

Follow PEP 8 guidelines:

```python
# Good
def detect_violence(frame, confidence_threshold=0.5):
    """
    Detect violence in a frame.
    
    Args:
        frame: Input frame/image
        confidence_threshold: Detection confidence threshold
        
    Returns:
        Detection results
    """
    results = model(frame)
    return results

# Use type hints when possible
def process_frame(frame: np.ndarray) -> dict:
    pass
```

### Django Code Style

- Follow Django best practices
- Use Django's built-in features when possible
- Keep views simple, move logic to models or services
- Use Django ORM properly

### Documentation

- Add docstrings to all functions and classes
- Update README.md for new features
- Comment complex algorithms
- Keep MODELS_README.md updated for model changes

### Testing

- Test new features manually
- Add automated tests when possible
- Ensure existing tests still pass
- Test on different environments if possible

## 🎯 Areas for Contribution

Here are some areas where we especially welcome contributions:

### Features

- [ ] Additional detection classes (e.g., fire, accidents)
- [ ] Multi-camera support
- [ ] Real-time alerts (email, SMS, webhooks)
- [ ] Mobile app integration
- [ ] Cloud deployment options
- [ ] Video file processing (not just live stream)
- [ ] Advanced analytics dashboard
- [ ] User authentication and authorization
- [ ] Database backend (PostgreSQL, MongoDB)

### Improvements

- [ ] Performance optimization
- [ ] Better face recognition accuracy
- [ ] Reduced false positives
- [ ] Better documentation
- [ ] More comprehensive tests
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] API improvements

### Bug Fixes

- Check the issues page for reported bugs
- Fix known issues
- Improve error handling

## 🔧 Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool
- CUDA (optional, for GPU support)

### Setup Steps

1. Fork and clone the repository
2. Run the setup script:
   ```bash
   ./setup.sh
   ```
3. Download model files (see MODELS_README.md)
4. Configure your camera in `detection_engine/yolo_detection.py`
5. Start development!

### Project Structure

```
violence_detection_with_facerecognation/
├── detection_engine/      # Core detection logic
├── web/                   # Web application
├── templates/             # HTML templates
├── static/                # CSS, JS files
├── models/                # ML models (not in repo)
├── media/                 # User data (not in repo)
├── logs/                  # Application logs
└── violence_detection/    # Django settings
```

## 📚 Resources

- [Django Documentation](https://docs.djangoproject.com/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV Documentation](https://docs.opencv.org/)

## ❓ Questions?

- Check existing issues and discussions
- Open a new issue with the "question" label
- Reach out to maintainers

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing private information without consent

## ⚖️ Legal and Ethical Considerations

When contributing to this project, please be mindful of:

1. **Privacy**: Don't include personal data in contributions
2. **Biometric Data**: Handle facial recognition ethically
3. **Surveillance**: Consider the implications of surveillance technology
4. **Testing**: Use appropriate test data, not real personal images
5. **Documentation**: Include warnings about legal requirements

## 🎉 Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Credited in release notes
- Thanked in the community

## 📞 Contact

- GitHub Issues: For bugs and features
- GitHub Discussions: For questions and discussions
- Email: [project-email@example.com]

---

Thank you for contributing! 🙏

Your efforts help make this project better for everyone.

