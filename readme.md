## user setup

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

## 1. Object Detection

### Overview

A real-time object detection system that uses TensorFlow and OpenCV to detect various objects through a webcam feed.

### Key Features

- Real-time object detection using TensorFlow models
- Web interface for interaction and control

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
- Web-based user interface for game control
- Support for multiple users/players

### Technical Components

- Face detection and recognition algorithms
- User enrollment system
- Multiple web interfaces for different playing the game on one and registering a name to the face on the other
- Progress tracking and visualization
- Video processing capabilities

## Common Features

- Real-time video processing
- Web-based interfaces
- Modular architecture
- Configuration options

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
│   └── game related files
├── person-recognition/   # Person recognition game implementation
│   ├── gameWeb/         # Main game interface
│   ├── gameWebNewUser/  # New user registration interface
├── various utility scripts
└── web/             # Web interface files for all the projects here
```

## Getting Started

1. run this in powershell after installing visual studio build tools with desktop development with c++, python 3.13, and python 3.9, and ffmpeg if using the "from video" script

2. run

```pwsh
git clone https://github.com/rsa17826/ai-img-detection.git ./ai-img-detection
cd ./ai-img-detection/object-detection
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz -o ./zip.tar.gz
tar -xvzf zip.tar.gz --strip-components=1
rm ./zip.tar.gz
py -3.13 -m venv ./.venv
Set-ExecutionPolicy Bypass -Scope Process -Force
./.venv/Scripts/Activate.ps1
python -m pip install -r ./requirements.txt
python tf_text_graph_ssd.py --input frozen_inference_graph.pb --config pipeline.config --output graph.pbtxt

cd ..

cd ./person-recognition
py -3.9 -m venv ./.venv
./.venv/Scripts/Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install face_recognition facenet-pytorch opencv-python flask pandas numpy eel

cd ..

```

3. Run the appropriate game script
4. Access web interface for interaction

## Future Enhancements

- Enhanced user interfaces
- Additional object detection models
- Improved face recognition accuracy

## Notes

- Requires appropriate hardware for optimal performance
- Camera access permissions needed
- Model files must be present for object detection
- Internet connection not required for web interface
<!--

either run object-detection.ps1, person-recognition.ps1, or "from video.bat" "pathToVideo"

- if running object-detection.ps1, person-recognition.ps1

  - you should see http://127.0.0.1:15674 open in your default browser with ui used to control the python script
    ![image showing the web ui](image.png)

- if running person-recognition.ps1

  - to start using person-recognition first enable the correct camera, then press "add face to list" and enter the faces name
  - you can add more than 1 face under each name to make the detection more reliable
  - once a face is added it will be able to start detecting any faces that are in the face list and unknown for those that arnt in the face list -->
