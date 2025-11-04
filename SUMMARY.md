# River's AI Image Detection Project

This project emerged from my work as a Laboratory Assistant in the Computer Science department's work-study program, where I explored the practical applications of artificial intelligence in computer vision. By combining cutting-edge technologies like TensorFlow and OpenCV with Python programming, I developed an interactive platform that demonstrates the potential of AI vision systems in real-world applications. The project not only served as a valuable learning experience but also provides a foundation for future students to understand and experiment with computer vision technologies in an engaging, hands-on manner.

In the spirit of embracing modern development tools, this documentation was created with the assistance of GitHub Copilot, showcasing how AI can enhance not only the project's core functionality but also its documentation process. This collaboration between human insight and AI assistance helped ensure comprehensive, well-structured documentation while significantly streamlining the documentation process.

## Core Technologies

### TensorFlow
TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive ecosystem of tools, libraries, and community resources for:

#### Real-World Applications
- **Computer Vision**: Object detection in self-driving cars, medical image analysis
- **Natural Language Processing**: Language translation, chatbots, text analysis
- **Recommendation Systems**: Product recommendations on e-commerce sites, content suggestions
- **Speech Recognition**: Voice assistants, transcription services
- **Gaming**: AI opponents, procedural content generation

#### In This Project
TensorFlow is used for:
- Running pre-trained object detection models (SSD architecture)
- Processing real-time video feeds for object detection
- Managing model inference and prediction
- Handling neural network computations for object classification

### OpenCV (Open Source Computer Vision Library)
OpenCV is a powerful computer vision and image processing library used extensively in both academic and commercial applications.

#### Real-World Applications
- **Manufacturing**: Quality control, defect detection
- **Security**: Motion detection, facial recognition systems
- **Healthcare**: Medical imaging analysis, patient monitoring
- **Retail**: Customer tracking, inventory management
- **Automotive**: Lane detection, parking assistance
- **Photography**: Image enhancement, filters, feature detection

#### In This Project
OpenCV is utilized for:
- Real-time video capture and processing
- Face detection and tracking
- Image preprocessing and enhancement
- Drawing detection boxes and visual overlays
- Frame manipulation and resizing
- Color space conversions

This project consists of two main components that leverage these technologies for interactive gaming and detection applications:

## 1. Object Detection Game

### Overview
A real-time object detection system that uses TensorFlow and OpenCV to create an interactive gaming experience. The system can detect various objects through a webcam feed and integrate them into gameplay.

### Key Features
- Real-time object detection using TensorFlow models
- Interactive gameplay based on detected objects
- Web interface for game interaction and control
- Pre-trained model support (using frozen_inference_graph.pb)
- Configurable detection parameters

### Technical Components
- TensorFlow for object detection
- OpenCV for image processing
- Web interface using HTML/JavaScript
- Model checkpoints and configurations
- SSD (Single Shot Detector) architecture implementation

## 2. Person Recognition Game

### Overview
A face detection and recognition system that creates interactive experiences based on identifying and tracking people in real-time video feeds.

### Key Features
- Face enrollment system for registering new users
- Real-time face detection and tracking
- Multiple game modes based on face recognition
- Web-based user interface for game control
- Support for multiple users/players

### Technical Components
- Face detection and recognition algorithms
- User enrollment system
- Multiple web interfaces for different game modes
- Progress tracking and visualization
- Video processing capabilities

## Common Features
- Real-time video processing
- Web-based interfaces
- Modular architecture
- Configuration options
- Progress tracking and logging

## Use Cases

### Educational
1. Interactive learning environments
2. Student engagement tracking
3. Attendance monitoring systems
4. Educational gaming platforms

### Entertainment
1. Interactive gaming experiences
2. Augmented reality applications
3. Multi-player face-based games
4. Object interaction games

### Security
1. Object detection systems
2. Face recognition security
3. Movement tracking
4. Presence detection

### Research
1. Human-computer interaction studies
2. Computer vision research
3. Machine learning model testing
4. User behavior analysis

## Technical Requirements

### Core Dependencies
1. **Python Environment**
   - Python 3.x for running the applications
   - Virtual environment recommended for package management

2. **TensorFlow Requirements**
   - TensorFlow 2.x
   - CUDA-compatible GPU (optional, for improved performance)
   - Minimum 4GB RAM (8GB+ recommended)
   - Compatible processor architecture

3. **OpenCV Requirements**
   - OpenCV-Python (cv2)
   - Working webcam/camera device
   - Sufficient CPU for real-time processing
   - DirectX/OpenGL support for display

4. **Additional Requirements**
   - Modern web browser with JavaScript enabled
   - Stable internet connection for web interface
   - Webcam/Camera with minimum 720p resolution
   - Required Python packages (specified in requirements.txt)

### Hardware Recommendations
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 8GB minimum, 16GB recommended
- GPU: NVIDIA GPU with CUDA support (optional)
- Camera: HD webcam with good low-light performance
- Display: 1080p or higher resolution

## Project Structure
```
ai-img-detection/
├── object-detection/      # Object detection game implementation
│   ├── saved_model/      # TensorFlow model files
│   ├── web/             # Web interface files
│   └── game related files
├── person-recognition/   # Person recognition game implementation
│   ├── gameWeb/         # Main game interface
│   ├── gameWebNewUser/  # New user registration interface
│   └── web/            # General web interface
└── various utility scripts
```

## Getting Started
1. Install required Python packages
2. Configure camera settings
3. Run the appropriate game script
4. Access web interface for interaction

## Future Enhancements
- Multiple camera support
- Advanced game modes
- Enhanced user interfaces
- Additional object detection models
- Improved face recognition accuracy
- Mobile device support

## Notes
- Requires appropriate hardware for optimal performance
- Camera access permissions needed
- Model files must be present for object detection
- Internet connection required for web interface
