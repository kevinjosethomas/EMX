# 👾

? is an asynchronous Python framework that combines computer vision, voice analysis, and procedural animation to create natural, expressive robotic eyes. The system reacts to human interaction through face tracking, gesture recognition, and emotional speech analysis. It also provides an interactive API to help extend the robot's capabilities by making it easy to integrate with other systems and sensors.

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

@robot.event("face_tracked")
async def on_face_tracked(face):
    print(f"Face position: {face}")

asyncio.run(robot.run())
```

This above script initializes the robot's face and starts a conversation with the Hume AI model. Additionally, it continually tracks any faces seen through the camera and plays a happy animation when a face is first detected. The robot face already has built-in animations for emotion states, and has idle behaviour like blinking and looking around. This extensible and modular API will allow it to be easily integrated with other systems and sensors, like gesture detection.

## Core Systems

### Emotion Engine

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

### Vision Engine

The vision engine processes camera input to detect faces, facial landmarks, and gestures using Google MediaPipe. It emits the following events:

- `face_appeared(face)` - when a face is first detected
- `face_disappeared(face)` - when a face is no longer detected
- `face_tracked(gesture)` - continuous face position updates

### Voice Engine

The voice engine allows for realtime conversations with the robot using Hume AI's Evi API. Besides generating text and audio, it provides sentiment/emotions for the robot's outputs, which are propagated to the above emotion engine to create more natural interactions.

It emits the following events:

- `_assistant_message(emotion)` - when the robot is about to start speaking
- `_assistant_message_end(emotion)` - when the robot has finished speaking

## Installation

First, clone the repository and install the dependencies:

```
# Clone the repository
git clone https://github.com/kevinjosethomas/k-emotion.git
cd k-emotion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
python3 -m pip install -r requirements.txt
```

Create a `.env` file in the root directory with the following content:

```env
HUME_API_KEY=your_api_key
HUME_SECRET_KEY=your_secret_key
HUME_CONFIG_ID=your_config_id
```

Develop a basic script using the robot API as shown above, and run it:
`python3 bot.py`
