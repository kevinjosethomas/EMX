import os
import asyncio
from .audio import AudioPlayer
from .recorder import AudioRecorder
from .processor import AudioProcessor
from pyee.asyncio import AsyncIOEventEmitter


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
        config=None,
    ):
        """Initialize the Voice system.

        Args:
            openai_api_key (str): API key for OpenAI
            robot (Robot, optional): Reference to main robot instance
            config (dict, optional): Configuration dictionary (if None, loads from file)
        """
        super().__init__()

        if config is None:
            from src.config import get_config

            self.config = get_config()
        else:
            self.config = config

        microphone_id = self.config.get("microphone_id")
        speaker_id = self.config.get("speaker_id")
        volume = self.config.get("volume", 0.15)
        debug = self.config.get("debug", False)

        self.recorder = AudioRecorder(microphone_id=microphone_id, debug=debug)
        self.processor = AudioProcessor(
            openai_api_key=openai_api_key, robot=robot, debug=debug
        )
        self.player = AudioPlayer(device_id=speaker_id, volume=volume)
        self._setup_event_handling()

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
        # self.processor.on("emotion_detected", self._handle_emotion_detected)
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
        await self.recorder.start()

        queue_monitor_task = asyncio.create_task(
            self.player.start_queue_monitor()
        )
        await self.processor.connect()

        queue_monitor_task.cancel()

    def set_volume(self, volume: float):
        """Set the audio playback volume.

        Args:
            volume (float): Volume level between 0.0 and 1.0
        """
        self.player.set_volume(volume)

        self.config["volume"] = volume
        from src.config import save_config

        save_config(self.config)
