import io
import base64
import asyncio
import sounddevice as sd
from typing import cast, Any
from funasr import AutoModel
from pydub import AudioSegment
from openai import AsyncOpenAI
from pyee.asyncio import AsyncIOEventEmitter
from .audio import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection


class Voice(AsyncIOEventEmitter):
    """Handles real-time voice interactions with OpenAI's API.

    This class manages bidirectional audio streaming between the local microphone
    and OpenAI's API, including:
    - Microphone input capture and streaming
    - Audio playback of API responses
    - Emotion analysis of assistant responses
    - Event emission for emotion changes

    Attributes:
        client (AsyncOpenAI): OpenAI API client
        microphone_id (str): ID of input microphone device
        connection (AsyncRealtimeConnection): Active connection to OpenAI
        session (Session): Current voice session
        audio_player (AudioPlayerAsync): Audio output manager
        should_send_audio (Event): Controls when to send mic audio
        connected (Event): Indicates active API connection
        last_audio_item_id (str): ID of last processed audio chunk
        emotion_buffer (BytesIO): Buffer for emotion analysis
        emotion_chunk_size (int): Number of chunks to analyze together
        chunk_counter (int): Tracks chunks for emotion analysis
        emotion_model (AutoModel): Emotion detection model
    """

    def __init__(self, openai_api_key, microphone_id=None):
        """Initialize the Voice system with OpenAI API and audio settings.

        This class manages the bidirectional audio stream between local microphone
        and OpenAI's API, including emotion analysis of responses.

        Args:
            openai_api_key (str): API key for OpenAI authentication
            microphone_id (str, optional): Specific microphone device ID. Defaults to None.

        Attributes:
            client: OpenAI API client instance
            connection: Active connection to OpenAI's realtime API
            audio_player: Handles async audio playback
            emotion_model: FunASR model for emotion detection
            emotion_buffer: Accumulates audio for emotion analysis
            emotion_chunk_size: Number of chunks to analyze together
        """

        super().__init__()
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.microphone_id = microphone_id
        self.connection = None
        self.session = None
        self.audio_player = AudioPlayerAsync()
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.last_audio_item_id = None

        self.emotion_buffer = io.BytesIO()
        self.emotion_chunk_size = 8
        self.chunk_counter = 0
        self.emotion_model = AutoModel(model="iic/emotion2vec_plus_base")

    async def _handle_session_created(self, conn):
        """Handle OpenAI session creation and setup.

        Configures the voice session with specific parameters and enables
        audio transmission.

        Args:
            conn (AsyncRealtimeConnection): Active connection to OpenAI API

        Settings:
            voice: "ash" - The voice model to use
            turn_detection: Server-side VAD with 0.7 threshold
        """

        self.session = conn.session
        self.should_send_audio.set()
        await conn.session.update(
            session={
                "voice": "ash",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.7,
                },
            }
        )

    async def _handle_audio_delta(self, event):
        """Process incoming audio chunks from OpenAI's response stream.

        Manages the audio buffer state and triggers processing of new audio chunks.
        Pauses microphone input while processing assistant's speech.

        Args:
            event: OpenAI audio delta event containing:
                - item_id: Unique ID for the audio sequence
                - delta: Base64 encoded audio data
        """

        self.should_send_audio.clear()

        if event.item_id != self.last_audio_item_id:
            self._reset_buffers(event.item_id)

        audio_bytes = base64.b64decode(event.delta)
        await self._process_audio_chunk(audio_bytes)

    def _reset_buffers(self, item_id):
        """Reset audio and emotion analysis buffers for a new utterance.

        Clears both raw audio and emotion analysis buffers and resets the
        chunk counter when a new audio sequence begins.

        Args:
            item_id (str): New audio sequence ID to track
        """

        self.audio_buffer = io.BytesIO()
        self.emotion_buffer = io.BytesIO()
        self.chunk_counter = 0
        self.last_audio_item_id = item_id

    async def _process_audio_chunk(self, audio_bytes):
        """Process an audio chunk for both emotion analysis and playback.

        Accumulates audio data for emotion analysis and sends it to the audio
        player. When sufficient chunks are collected, triggers emotion analysis.

        Args:
            audio_bytes (bytes): Raw PCM audio data at 24kHz, 16-bit, mono

        The method:
        1. Adds chunk to emotion analysis buffer
        2. Increments chunk counter
        3. Triggers emotion analysis if buffer is full
        4. Sends audio to playback system
        """

        self.emotion_buffer.write(audio_bytes)
        self.chunk_counter += 1

        if self.chunk_counter >= self.emotion_chunk_size:
            await self._analyze_emotion_buffer()
            self._reset_emotion_buffer()

        self.audio_player.add_data(audio_bytes)

    async def _analyze_emotion_buffer(self):
        """Analyze emotion in accumulated audio buffer and emit results.

        Processes the collected audio buffer to detect emotions using the FunASR model.
        Calculates audio duration based on buffer size and sample rate.
        Emits '_assistant_message' event with emotion and duration data.

        The method:
        1. Gets raw audio data from buffer
        2. Calculates duration from buffer size and audio format (24kHz, 16-bit)
        3. Analyzes emotion using FunASR model
        4. Maps emotion scores to simplified categories
        5. Emits results to emotion engine

        Event data format:
            {
                'emotion': str,  # Simplified emotion category
                'duration': float  # Audio duration in seconds
            }
        """

        emotion_audio = self.emotion_buffer.getvalue()
        audio_duration = len(emotion_audio) / (24000 * 2)

        emotion = await self.analyze_audio_emotion(emotion_audio)
        if emotion:
            detected_emotion = self._get_detected_emotion(emotion[0]["scores"])
            self.emit(
                "_assistant_message",
                {
                    "emotion": detected_emotion,
                    "duration": audio_duration,
                },
            )

    def _reset_emotion_buffer(self):
        """Reset emotion analysis state for new utterance.

        Clears the emotion analysis buffer and resets chunk counter
        to prepare for a new sequence of audio chunks.

        The method:
        1. Creates new empty BytesIO buffer
        2. Resets chunk counter to 0
        """

        self.emotion_buffer = io.BytesIO()
        self.chunk_counter = 0

    def _get_detected_emotion(self, scores):
        """Map raw emotion scores to simplified categories.

        Takes raw emotion scores from FunASR model and maps them to a reduced set
        of core emotions for more stable expression changes.

        Args:
            scores (list): Raw emotion probability scores from model

        Returns:
            str: Simplified emotion category with highest probability
                One of: anger, fear, happiness, neutral, sadness

        Emotion mapping:
            - angry, disgusted -> anger
            - fearful -> fear
            - happy, surprised -> happiness
            - neutral, other, unknown -> neutral
            - sad -> sadness
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
        return emotion_labels[scores.index(max(scores))]

    async def wait_for_audio_completion(self):
        """Wait for audio playback to complete before re-enabling microphone.

        This method ensures that the assistant's speech is fully played
        before allowing new microphone input, preventing feedback loops
        where the assistant hears its own output.

        The method polls the audio player's queue length and only re-enables
        the microphone once the queue is empty and a small buffer period
        has passed.
        """

        while len(self.audio_player.queue) > 0:
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.1)
        self.should_send_audio.set()

    async def _get_connection(self) -> AsyncRealtimeConnection:
        """Get the current OpenAI API connection.

        Waits for an active connection to be established before returning.

        Returns:
            AsyncRealtimeConnection: The active connection to OpenAI's API

        Raises:
            AssertionError: If no connection exists after waiting
        """

        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def run(self):
        """Main voice processing loop for OpenAI API interaction.

        Manages the bidirectional audio stream between local microphone and
        OpenAI's API, including emotion analysis of responses.

        The method:
        1. Starts microphone capture task in background
        2. Establishes real-time connection to OpenAI API
        3. Processes events from API:
            - session.created: Configures voice and VAD settings
            - session.updated: Updates session state
            - response.audio.delta: Processes audio chunks for emotion/playback
            - response.done: Waits for audio completion before re-enabling mic
        4. Handles error conditions and connection cleanup

        Events emitted:
            - _assistant_message: On emotion detection with emotion/duration data
            - _assistant_message_end: When response is complete

        Note:
            Audio chunks are accumulated for emotion analysis before playback.
            Microphone input is paused during assistant speech to prevent feedback.
        """

        asyncio.create_task(self.send_mic_audio())

        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview"
        ) as conn:
            self.connection = conn
            self.connected.set()
            self.should_send_audio.clear()

            async for event in conn:
                if event.type == "session.created":
                    await self._handle_session_created(conn)
                elif event.type == "session.updated":
                    self.session = event.session
                elif event.type == "response.audio.delta":
                    await self._handle_audio_delta(event)
                elif event.type == "response.done":
                    self.emit("_assistant_message_end")
                    asyncio.create_task(self.wait_for_audio_completion())
                elif event.type == "error":
                    print(event.error)

    async def analyze_audio_emotion(self, audio_bytes):
        """Analyze emotion in raw audio bytes using FunASR model.

        Converts raw audio data to the format required by the emotion model
        and performs emotion analysis.

        Args:
            audio_bytes (bytes): Raw PCM audio data at 24kHz, 16-bit, mono

        Returns:
            dict | None: Dictionary containing emotion analysis results with:
                - scores: List of emotion probability scores
                - labels: Corresponding emotion labels
            Returns None if analysis fails

        Processing steps:
        1. Converts 24kHz audio to 16kHz for emotion model
        2. Converts to WAV format in memory
        3. Runs FunASR emotion analysis model
        4. Returns raw emotion scores and labels

        Raises:
            Exception: If audio processing or emotion analysis fails
                Exception is caught and logged, returning None
        """

        try:

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

    async def send_mic_audio(self) -> None:
        """Capture and stream microphone audio to OpenAI's API.

        This method:
        1. Initializes a real-time audio input stream
        2. Continuously captures audio chunks when enabled
        3. Sends chunks to OpenAI's API for processing
        4. Manages the session state including canceling existing responses
           when new audio input begins

        The audio capture is controlled by the should_send_audio event,
        allowing other parts of the system to pause/resume the microphone
        input as needed.

        Audio is captured in chunks of 0.1 seconds using the configured
        sample rate and format settings.
        """

        sent_audio = False
        read_size = int(SAMPLE_RATE * 0.1)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=read_size,
            latency="low",
        )
        stream.start()

        try:
            while True:
                await asyncio.sleep(0.05)

                if not self.should_send_audio.is_set():
                    continue

                if stream.read_available >= read_size:
                    data, _ = stream.read(read_size)
                    connection = await self._get_connection()

                    if not sent_audio:
                        await connection.send({"type": "response.cancel"})
                        sent_audio = True

                    audio_b64 = base64.b64encode(cast(Any, data)).decode(
                        "utf-8"
                    )
                    await connection.input_audio_buffer.append(audio=audio_b64)

        finally:
            stream.stop()
            stream.close()
