import os
import io
import time
import asyncio
import sounddevice as sd
import concurrent.futures
from pydub import AudioSegment
from pyee.asyncio import AsyncIOEventEmitter
from .audio import CHANNELS, SAMPLE_RATE


class AudioRecorder(AsyncIOEventEmitter):
    """Handles microphone input capture and streaming.

    This class is responsible for capturing audio from the microphone and
    emitting events with the captured audio data. It runs in a separate thread
    to avoid blocking the main thread.

    Attributes:
        microphone_id (str): ID of input microphone device
        should_record (asyncio.Event): Controls when to record
        audio_thread_pool (ThreadPoolExecutor): Thread pool for audio capture
        input_sample_rate (int): Actual sample rate of the input device

    Events emitted:
        - audio_captured: When audio is captured, with the audio data
    """

    def __init__(self, microphone_id=None, debug=False):
        """Initialize the AudioRecorder.

        Args:
            microphone_id (str, optional): Specific microphone device ID. Defaults to None.
            debug (bool, optional): Enable debug mode. Defaults to False.
        """
        super().__init__()
        self.microphone_id = microphone_id
        self.should_record = asyncio.Event()
        self.debug = debug
        self.debug_mic_buffer = io.BytesIO() if debug else None
        self.input_sample_rate = SAMPLE_RATE  # Will be updated when starting

        # Create thread pool for audio capture
        self.audio_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="audio_recorder"
        )

        # Try to set high priority for audio thread if possible
        if hasattr(os, "sched_get_priority_max"):
            try:
                audio_priority = os.sched_get_priority_max(os.SCHED_FIFO)
                os.sched_setscheduler(
                    0, os.SCHED_FIFO, os.sched_param(audio_priority)
                )
            except Exception as e:
                print(f"Could not set audio thread priority: {e}")

    async def start(self):
        """Start the audio recorder in a separate thread."""
        self._main_loop = asyncio.get_running_loop()
        asyncio.create_task(self._start_recording())

    async def _start_recording(self):
        """Start the audio recording thread."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.audio_thread_pool, self._capture_audio
            )
        except Exception as e:
            print(f"Error in mic audio capture: {e}")

    def _capture_audio(self):
        """Audio capture function running in separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Initialize microphone
        device = self.microphone_id if self.microphone_id is not None else None
        device_info = sd.query_devices(device, "input")
        actual_input_sr = int(device_info["default_samplerate"])
        self.input_sample_rate = actual_input_sr
        print(
            f"Using input sample rate {actual_input_sr} instead of {SAMPLE_RATE}"
        )

        read_size = int(actual_input_sr * 0.1)

        stream = sd.InputStream(
            device=device,
            channels=CHANNELS,
            samplerate=actual_input_sr,
            dtype="int16",
            blocksize=read_size,
            latency="high",
            callback=None,
        )
        stream.start()

        try:
            while True:
                if not self.should_record.is_set():
                    time.sleep(0.1)
                    continue

                try:
                    data, _ = stream.read(read_size)
                except sd.PortAudioError as e:
                    print(f"PortAudio error: {e}")
                    time.sleep(0.1)
                    continue

                # Process audio in main thread
                asyncio.run_coroutine_threadsafe(
                    self._process_captured_audio(data), self._main_loop
                )
        finally:
            stream.stop()
            stream.close()

    async def _process_captured_audio(self, data):
        """Process captured audio data and emit event.

        Args:
            data (numpy.ndarray): Raw audio data from microphone
        """
        raw_bytes = data.tobytes()

        # Resample if needed to match target sample rate
        if self.input_sample_rate != SAMPLE_RATE:
            segment = AudioSegment(
                data=raw_bytes,
                sample_width=2,
                frame_rate=self.input_sample_rate,
                channels=CHANNELS,
            )
            segment = segment.set_frame_rate(SAMPLE_RATE)
            audio_bytes = segment.raw_data
        else:
            audio_bytes = raw_bytes

        # Save debug audio if enabled
        if self.debug and self.debug_mic_buffer is not None:
            self.debug_mic_buffer.write(audio_bytes)

        # Emit audio_captured event with processed audio
        self.emit(
            "audio_captured",
            {"audio_bytes": audio_bytes, "sample_rate": SAMPLE_RATE},
        )

    def start_recording(self):
        """Start recording audio from microphone."""
        self.should_record.set()

    def stop_recording(self):
        """Stop recording audio from microphone."""
        self.should_record.clear()

    def close(self):
        """Close the audio recorder and release resources."""
        self.stop_recording()
        self.audio_thread_pool.shutdown()
