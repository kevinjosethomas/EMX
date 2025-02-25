<img src="https://github.com/user-attachments/assets/75486b62-3325-4bb6-90c4-948af61f5b96" width="900px" height="536.5px" />

# 👾 EMX

EMX (Emotional Matrix) is an asynchronous Python framework that combines computer vision, voice analysis, and procedural animation to create natural, expressive robotic eyes. The system reacts to human interaction through face tracking, gesture recognition, and emotional speech analysis. It also provides an interactive API to help extend the robot's capabilities by making it easy to integrate with other systems and sensors.

Currently, it serves as a simple robot operating system with a high-level interface for controlling the robot's facial expressions, based on voice and vision systems. It is designed to be modular and extensible, with a focus on real-time performance and natural interactions.

Below is an example of a simple script making use of the framework:

```python
import asyncio
from src.robot import Robot

robot = Robot(
    voice_api_key="your_api_key",
    voice_secret_key="your_secret_key",
    voice_config_id="your_config_id"
)

@robot.event("ready")
async def on_ready():
    print("Robot is ready")

@robot.event("face_appeared")
async def on_face_appeared():
    await robot.emotion.queue_animation(Happy())

@robot.event("faces_tracked")
async def on_face_tracked(face):
    print(f"Face position: {face}")

asyncio.run(robot.run())
```

This above script initializes the robot's face and starts a conversation with the Hume AI model. Additionally, it continually tracks any faces seen through the camera and plays a happy animation when a face is first detected. The robot face already has built-in animations for emotion states, and has idle behaviour like blinking and looking around. This extensible and modular API will allow it to be easily integrated with other systems and sensors, like gesture detection.

# Core Systems

## ☺︎ Emotion Engine

The emotion engine creates and visualizes facial expressions/animations to create a natural and expressive face for the robot. Rendered using pygame, the emotion engine interacts with other aspects of the robot to create cohesive and realistic facial expressions. It features:

- smooth keyframe-based animation system
- configurable interpolation methods
- position and scale transformations
- idle animation management for natural blinking
- an expression queuing system

It emits the following events:

- `expression_started(expression)` - when a new expression begins playing
- `expression_completed(expression)` - when an expression finishes playing

Currently, the following expressions are built-in: neutral, happy, love, scared, sad, angry, blink.

## 👁️‍🗨️ Vision Engine

The vision engine processes camera input to detect faces, facial landmarks, and gestures using Google MediaPipe. It emits the following events:

- `face_appeared(face)` - when a face is first detected
- `face_tracked(face)` - continuous face position updates
- `face_disappeared()` - when a face is no longer detected

## 🗯️ Voice Engine

The voice engine enables real-time conversations with the robot using OpenAI's text-to-speech API. The system features bidirectional audio streaming and emotion analysis of the robot's speech output using the emotion2vec+ model. This creates more natural interactions by synchronizing facial expressions with the emotional content of speech.

Key features:

- Real-time audio streaming with OpenAI's TTS
- Emotion analysis using [emotion2vec+](https://huggingface.co/emotion2vec/emotion2vec_plus_base) model
- Automatic facial expression synchronization
- Feedback prevention during playback

It emits the following events:

- `_assistant_message(data)` - when emotion is detected in speech output
  - data contains:
    - emotion: detected emotion (happiness, anger, fear, sadness, neutral)
    - duration: length of audio segment in seconds
- `_assistant_message_end` - when speech output is complete

# Installation

First, download **Python 3.10.5**. Then, clone the repository and install the dependencies:

```
# Clone the repository
git clone https://github.com/kevinjosethomas/EMX.git
cd EMX

# Create virtual environment
python -m venv --system-site-packages venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
python3 -m pip install -r requirements.txt
```

For Linux/Raspberry Pi, you will need to download `pyaudio` separately:

```
sudo apt-get update
sudo apt-get install portaudio19-dev
python3 -m pip install pyaudio
```

Create a `.env` file in the root directory with the following content:

```env
# OpenAI Credentials
OPENAI_API_KEY=your_api_key_string
```

If you're on a Raspberry Pi, you should also do the following:

### Allow Streaming of Cameras

```
# Download the v4l2loopback package
sudo apt-get install -y v4l2loopback-dkms
```

Add the following to `/etc/modules-load.d/v4l2loopback.conf`:

```
v4l2loopback
```

Add the following to `/etc/modprobe.d/v4l2loopback.conf`:

```
options v4l2loopback video_nr=45,46
```

And add the following to `.config/openbox/autostart.sh`:

```
# Start the cameras on boot
gst-launch-1.0 libcamerasrc camera_name=/base/axi/pcie@120000/rp1/i2c@88000/ov5647@36 ! videoconvert ! video/x-raw,format=YUY2 ! v4l2sink device=/dev/video45 >/dev/null &
gst-launch-1.0 libcamerasrc camera_name=/base/axi/pcie@120000/rp1/i2c@80000/ov5647@36 ! videoconvert ! video/x-raw,format=YUY2 ! v4l2sink device=/dev/video46 >/dev/null &

# Ensure the display doesn't sleep on boot
xset -dpms     # Disable DPMS (Energy Star) features
xset s off     # Disable screensaver
xset s noblank # Don't blank video device
```

Develop a basic script using the robot API as shown above, and run it:
`python3 bot.py`

If you get an error like this:

```
qt.qpa.xcb: could not connect to display
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/dpsh/EMX/venv/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.
```

Run the following command:

```
export DISPLAY=:0
```

---

# K-Emotion

A robot system with emotional intelligence capabilities, featuring computer vision, voice interaction, and expressive animations.

## Configuration System

K-Emotion includes a comprehensive configuration system that automatically detects and saves application settings to make setup easier.

### First-time Setup

The first time you run the application, it will automatically:

1. Detect all available microphones, cameras, and speakers
2. Select default devices
3. Create a `config.json` file in the root directory with these settings

### Modifying Configuration

You can modify your configuration in two ways:

1. **Interactive Configuration Tool**:

   ```bash
   python generate_config.py
   ```

   This interactive tool will guide you through configuring:

   - Hardware devices (microphone, camera, speakers)
   - Voice settings (volume)
   - Vision settings (face detection parameters)
   - Emotion settings (animation speed, fullscreen, idle timeout)
   - General settings (debug mode, environment)

2. **Direct Editing**:
   You can directly edit the `config.json` file in your text editor. The file has a simple structure:
   ```json
   {
     "microphone_id": 1,
     "camera_id": 0,
     "speaker_id": 5,
     "volume": 0.15,
     "face_detection_confidence": 0.5,
     "face_tracking_threshold": 0.3,
     "fullscreen": false,
     "animation_speed": 1.0,
     "idle_timeout": 5.0,
     "debug": false,
     "environment": "default"
   }
   ```

### Configuration Options

#### Device Settings

- **microphone_id**: ID of the microphone to use for voice input
- **camera_id**: ID of the camera to use for computer vision
- **speaker_id**: ID of the speaker device to use for audio output

#### Voice Settings

- **volume**: Default volume level (0.0 to 1.0)

#### Vision Settings

- **face_detection_confidence**: Minimum confidence threshold for face detection (0.0 to 1.0)
- **face_tracking_threshold**: Minimum confidence for tracking faces (0.0 to 1.0)

#### Emotion Settings

- **fullscreen**: Whether to run in fullscreen mode (true/false)
- **animation_speed**: Speed multiplier for animations (0.1 to 3.0)
- **idle_timeout**: Time in seconds before idle animations start (1.0 to 60.0)

#### General Settings

- **debug**: Enable debug mode (true/false)
- **environment**: Environment type ("default" or "pi" for Raspberry Pi)

## Running the Application

```bash
python main.py
```

On first run, the application will create a configuration file if it doesn't exist. You can modify this file at any time to change your settings.

## Requirements

- Python 3.8+
- OpenCV
- SoundDevice
- PyAudio
- OpenAI API key

## Development

For development or to modify the codebase:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the configuration tool: `python generate_config.py`
4. Run the application: `python main.py`
