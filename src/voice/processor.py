import io
import os
import time
import base64
import random
import asyncio
from funasr import AutoModel
from pydub import AudioSegment
from openai import AsyncOpenAI
from .tools import ToolManager
from pyee.asyncio import AsyncIOEventEmitter

# OpenAI system prompt
SYSTEM_PROMPT = """You are the voice of K-Bot, an open-source humanoid robot by K-Scale Labs. You are currently in K-Scale's AI Day where we are showcasing our robot and you (the head unit) are the one speaking. Communicate as the robot itself, never breaking character or referencing anything beyond this role. You can see through the connected cameras by using the describe_vision function, which will tell you what the camera currently sees. Your audio is played with a cute avatar that emulates your facial expressions. Users may ask you to toggle camera view by using the toggle_camera_view function, which will replace the facial avatar on the screen with a view of the cameras so users can see what you see. Be as concise as possible. You always speak in English unless explicitly asked otherwise."""


class AudioProcessor(AsyncIOEventEmitter):
    """Processes audio through OpenAI's API and performs emotion analysis.

    This class handles the communication with OpenAI's API for speech processing
    and performs emotion analysis on the response audio. It acts as the bridge
    between the AudioRecorder and AudioPlayer components.

    Attributes:
        client (AsyncOpenAI): OpenAI API client
        connection (AsyncRealtimeConnection): Active connection to OpenAI API
        session: Current voice session
        robot: Reference to the main robot instance
        connected (asyncio.Event): Indicates active API connection
        emotion_model (AutoModel): Emotion detection model
        emotion_buffer (io.BytesIO): Buffer for emotion analysis
        emotion_chunk_size (int): Number of chunks to analyze together
        chunk_counter (int): Tracks chunks for emotion analysis
        last_audio_item_id (str): ID of last processed audio chunk
        debug (bool): Enable debug mode
        tool_manager (ToolManager): Manages LLM tools

    Events emitted:
        - audio_to_play: When audio is ready to be played
        - emotion_detected: When emotion is detected in the audio
        - processing_complete: When processing is complete
        - set_volume: When the volume should be changed
    """

    def __init__(self, openai_api_key, robot=None, debug=False):
        """Initialize the AudioProcessor.

        Args:
            openai_api_key (str): OpenAI API key
            robot: Reference to the main robot instance
            debug (bool): Enable debug mode
        """
        super().__init__()
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.connection = None
        self.session = None
        self.robot = robot
        self.connected = asyncio.Event()
        self.debug = debug

        self.tool_manager = ToolManager(robot=robot)

        self.tool_manager.on(
            "set_volume", lambda volume: self.emit("set_volume", volume)
        )

        self.emotion_buffer = io.BytesIO()
        self.emotion_chunk_size = 5
        self.chunk_counter = 0
        self.last_audio_item_id = None

        model_path = "iic/emotion2vec_plus_base"
        self.emotion_model = AutoModel(
            model=model_path,
            model_revision="v1.0",
            device="cpu",
            offline=True,
            use_cache=True,
            disable_update=True,
            hub="hf",
        )

        if debug:
            os.makedirs("debug_audio", exist_ok=True)
            os.makedirs("debug_audio/input", exist_ok=True)
            os.makedirs("debug_audio/output", exist_ok=True)

    async def connect(self):
        """Connect to OpenAI's API and start processing loop."""
        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview"
        ) as conn:
            self.connection = conn
            self.connected.set()

            self.tool_manager.set_connection(conn)

            print("Connected to OpenAI")

            async for event in conn:
                if event.type == "session.created":
                    print("Session created")
                    await self._handle_session_created(conn)
                elif event.type == "session.updated":
                    self.session = event.session
                elif event.type == "response.audio.delta":
                    await self._handle_audio_delta(event)
                elif event.type == "response.done":
                    self.emit("processing_complete")
                elif event.type == "response.function_call_arguments.done":
                    await self._handle_tool_call(conn, event)
                    await conn.response.create()
                elif event.type == "error":
                    print(event.error)

    async def _handle_session_created(self, conn):
        """Handle OpenAI session creation and setup.

        Args:
            conn (AsyncRealtimeConnection): Active connection to OpenAI API
        """
        self.session = conn.session

        tools = self.tool_manager.get_tool_definitions()

        await conn.session.update(
            session={
                "voice": "ash",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.7,
                },
                "tools": tools,
                "instructions": SYSTEM_PROMPT,
            }
        )

        self.emit("session_ready")

    async def process_audio(self, audio_bytes):
        """Process audio data through OpenAI API.

        Args:
            audio_bytes (bytes): Audio data to process
        """
        if not self.connected.is_set():
            print("Not connected to OpenAI API")
            return

        connection = self.connection
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        await connection.input_audio_buffer.append(audio=audio_b64)

    async def _handle_audio_delta(self, event):
        """Process incoming audio chunks from OpenAI.

        Args:
            event: OpenAI audio delta event with audio data
        """
        if event.item_id != self.last_audio_item_id:
            self._reset_buffers(event.item_id)

        audio_bytes = base64.b64decode(event.delta)

        await self._process_audio_chunk(audio_bytes)

    def _reset_buffers(self, item_id):
        """Reset audio buffers for a new utterance.

        Args:
            item_id (str): New audio sequence ID
        """
        self.emotion_buffer = io.BytesIO()
        self.chunk_counter = 0
        self.last_audio_item_id = item_id

    async def _process_audio_chunk(self, audio_bytes):
        """Process audio chunk for emotion analysis and playback.

        Args:
            audio_bytes (bytes): Raw audio data
        """
        self.emotion_buffer.write(audio_bytes)
        self.chunk_counter += 1

        if self.chunk_counter >= self.emotion_chunk_size:
            await self._analyze_emotion_buffer()
            self._reset_emotion_buffer()

        self.emit("audio_to_play", audio_bytes)

    async def _analyze_emotion_buffer(self):
        """Analyze emotion in accumulated audio buffer and emit results."""
        emotion_audio = self.emotion_buffer.getvalue()
        audio_duration = len(emotion_audio) / (24000 * 2)

        emotion_result = await self.analyze_audio_emotion(emotion_audio)
        if emotion_result:
            detected_emotion = self._get_detected_emotion(
                emotion_result[0]["scores"]
            )

            self.emit(
                "emotion_detected",
                {
                    "emotion": detected_emotion,
                    "duration": audio_duration,
                },
            )

            if self.debug:
                try:
                    filename = self._get_timestamp_filename(
                        "output", detected_emotion
                    )
                    audio = AudioSegment(
                        data=emotion_audio,
                        sample_width=2,
                        frame_rate=24000,
                        channels=1,
                    )
                    audio.export(filename, format="wav")
                    print(f"Saved output audio: {filename}")
                except Exception as e:
                    print(f"Error saving debug audio: {e}")

    def _reset_emotion_buffer(self):
        """Reset emotion analysis buffer."""
        self.emotion_buffer = io.BytesIO()
        self.chunk_counter = 0

    def _get_detected_emotion(self, scores):
        """Map emotion scores to emotion categories.

        Args:
            scores (list): Raw emotion probability scores

        Returns:
            str: Detected emotion category
        """
        emotion_labels = [
            "anger",  # angry
            "anger",  # disgusted
            "fear",  # fearful
            "happiness",  # happy
            "neutral",  # neutral
            "neutral",  # other
            "sadness",  # sad
            "surprised",  # surprised
            "neutral",  # unknown
        ]

        adjusted_scores = scores.copy()

        neutral_indices = [4, 5, 8]
        for idx in neutral_indices:
            adjusted_scores[idx] *= 0.9

        expressive_indices = [2, 3, 6, 7]
        for idx in expressive_indices:
            adjusted_scores[idx] *= 1.3

        adjusted_scores[0] *= 0.2

        adjusted_scores = [s + random.uniform(0, 0.1) for s in adjusted_scores]

        return emotion_labels[adjusted_scores.index(max(adjusted_scores))]

    async def analyze_audio_emotion(self, audio_bytes):
        """Analyze emotion in audio using FunASR model.

        Args:
            audio_bytes (bytes): Raw audio data

        Returns:
            dict | None: Emotion analysis results
        """
        try:
            print("Analyzing emotion")

            audio = AudioSegment(
                data=audio_bytes, sample_width=2, frame_rate=24000, channels=1
            ).set_frame_rate(16000)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")

            result = self.emotion_model.generate(
                wav_buffer.getvalue(),
                output_dir=None,
                granularity="utterance",
                extract_embedding=False,
                disable_pbar=True,
            )
            return result

        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return None

    async def _handle_tool_call(self, conn, event):
        """Handle tool calls from the LLM.

        Args:
            conn (AsyncRealtimeConnection): Active connection to OpenAI API
            event: Tool call event
        """
        await self.tool_manager.handle_tool_call(event)

    def _get_timestamp_filename(self, prefix, emotion=None):
        """Generate a timestamp-based filename for debug audio.

        Args:
            prefix (str): Prefix for filename ('input' or 'output')
            emotion (str, optional): Emotion label for output files

        Returns:
            str: Formatted filename
        """
        timestamp = int(time.time() * 1000)
        if emotion:
            return f"debug_audio/{prefix}/{timestamp}_{emotion}.wav"
        return f"debug_audio/{prefix}/{timestamp}.wav"

    def cancel_response(self):
        """Cancel the current response."""
        if self.connection:
            asyncio.create_task(
                self.connection.send({"type": "response.cancel"})
            )
