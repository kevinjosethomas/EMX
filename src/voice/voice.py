import io
import os
import json
import time
import base64
import random
import asyncio
import sounddevice as sd
import concurrent.futures
from typing import cast, Any
from funasr import AutoModel
from pydub import AudioSegment
from openai import AsyncOpenAI
from pyee.asyncio import AsyncIOEventEmitter
from .audio import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from .recorder import AudioRecorder
from .processor import AudioProcessor
from .audio import AudioPlayer

SYSTEM_PROMPT = """You are the voice of K-Bot, an open-source humanoid robot by K-Scale Labs. You are currently in K-Scale's AI Day where we are showcasing our robot and you (the head unit) are the one speaking. Communicate as the robot itself, never breaking character or referencing anything beyond this role. You can see through the connected cameras by using the describe_vision function, which will tell you what the camera currently sees. Your audio is played with a cute avatar that emulates your facial expressions. Users may ask you to toggle camera view by using the toggle_camera_view function, which will replace the facial avatar on the screen with a view of the cameras so users can see what you see. Be as concise as possible. You always speak in English unless explicitly asked otherwise."""


class Voice(AsyncIOEventEmitter):
    """Manages voice interactions with OpenAI's API.

    This class coordinates between the audio recorder, processor, and player
    components to create a complete voice interaction system. It handles the
    flow of audio data between these components and manages events.

    Attributes:
        recorder (AudioRecorder): Microphone input capture
        processor (AudioProcessor): OpenAI API communication and emotion analysis
        player (AudioPlayer): Audio playback

    Events emitted:
        - _assistant_message: When emotion is detected in response audio
        - _assistant_message_end: When response is complete
    """

    def __init__(
        self,
        openai_api_key,
        robot=None,
        microphone_id=None,
        debug=False,
        volume=0.15,
    ):
        """Initialize the Voice system.

        Args:
            openai_api_key (str): API key for OpenAI
            robot (Robot, optional): Reference to main robot instance
            microphone_id (str, optional): Specific microphone device ID
            debug (bool, optional): Enable debug mode
            volume (float, optional): Initial playback volume
        """
        super().__init__()

        # Create components
        self.recorder = AudioRecorder(microphone_id=microphone_id, debug=debug)
        self.processor = AudioProcessor(
            openai_api_key=openai_api_key, robot=robot, debug=debug
        )
        self.player = AudioPlayer(volume=volume)

        # Set up event handling
        self._setup_event_handling()

        # Debug mode
        self.debug = debug
        if debug:
            os.makedirs("debug_audio", exist_ok=True)
            os.makedirs("debug_audio/input", exist_ok=True)
            os.makedirs("debug_audio/output", exist_ok=True)

    def _setup_event_handling(self):
        """Set up event handling between components."""

        # Recorder -> Processor
        self.recorder.on("audio_captured", self._handle_audio_captured)

        # Processor -> Player
        self.processor.on("audio_to_play", self._handle_audio_to_play)
        self.processor.on("emotion_detected", self._handle_emotion_detected)
        self.processor.on(
            "processing_complete", self._handle_processing_complete
        )
        self.processor.on(
            "session_ready", lambda: self.recorder.start_recording()
        )
        self.processor.on("set_volume", self._handle_set_volume)

        # Player -> Voice
        self.player.on("queue_empty", self._handle_queue_empty)

    async def _handle_audio_captured(self, data):
        """Handle audio captured from microphone.

        Args:
            data (dict): Contains audio_bytes and sample_rate
        """
        await self.processor.process_audio(data["audio_bytes"])

    def _handle_audio_to_play(self, audio_bytes):
        """Handle processed audio to play.

        Args:
            audio_bytes (bytes): Audio data to play
        """
        self.player.add_data(audio_bytes)

        # Stop recording while assistant is speaking to prevent feedback
        self.recorder.stop_recording()

    def _handle_emotion_detected(self, data):
        """Handle emotion detected in assistant's response.

        Args:
            data (dict): Emotion data with emotion and duration
        """
        self.emit("_assistant_message", data)

    def _handle_processing_complete(self):
        """Handle completion of OpenAI response."""
        self.emit("_assistant_message_end")
        asyncio.create_task(self._wait_for_audio_completion())

    def _handle_queue_empty(self):
        """Handle audio queue becoming empty."""
        # Allow recording again when playback is done
        self.recorder.start_recording()

    def _handle_set_volume(self, volume):
        """Handle volume change request.

        Args:
            volume (float): New volume level
        """
        self.player.set_volume(volume)

    async def _wait_for_audio_completion(self):
        """Wait for audio playback to complete."""
        await self.player.wait_for_queue_empty()
        await asyncio.sleep(0.1)  # Small buffer period
        self.recorder.start_recording()

    async def run(self):
        """Run the voice system.

        Starts all components and connects them together.
        """
        # Initialize components
        await self.recorder.start()

        # Start queue monitor in player
        queue_monitor_task = asyncio.create_task(
            self.player.start_queue_monitor()
        )

        # Connect to OpenAI and process events
        await self.processor.connect()

        # Cleanup
        queue_monitor_task.cancel()

    def set_volume(self, volume: float):
        """Set the audio playback volume.

        Args:
            volume (float): Volume level between 0.0 and 1.0
        """
        self.player.set_volume(volume)
