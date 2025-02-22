"""
This code is directly ported from https://github.com/openai/openai-python/blob/main/examples/realtime/audio_util.py
"""

from __future__ import annotations

import io
import base64
import pyaudio
import asyncio
import threading
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from typing import Callable, Awaitable
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

CHANNELS = 1
SAMPLE_RATE = 24000
FORMAT = pyaudio.paInt16
CHUNK_LENGTH_S = 0.05  # 100ms


def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
    # load the audio file from the byte stream
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    print(
        f"Loaded audio: {audio.frame_rate=} {audio.channels=} {audio.sample_width=} {audio.frame_width=}"
    )
    # resample to 24kHz mono pcm16
    pcm_audio = (
        audio.set_frame_rate(SAMPLE_RATE)
        .set_channels(CHANNELS)
        .set_sample_width(2)
        .raw_data
    )
    return pcm_audio


class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        
        # Query available devices
        devices = sd.query_devices()
        print("Available audio devices:")
        print(devices)
        
        # Try to find a valid default output device
        try:
            default_device = sd.query_devices(kind='output')
            device_id = default_device['index']
            print(f"Using output device: {default_device['name']}")
            
            # Get supported sample rates
            supported_rates = default_device.get('default_samplerate')
            print(f"Device default sample rate: {supported_rates}")
            
            # Store both rates for conversion
            self.input_rate = SAMPLE_RATE
            self.output_rate = int(supported_rates) if supported_rates else SAMPLE_RATE
            print(f"Input rate: {self.input_rate}, Output rate: {self.output_rate}")
            
        except Exception as e:
            print(f"Warning: Error querying output device: {e}")
            device_id = None
            self.input_rate = SAMPLE_RATE
            self.output_rate = SAMPLE_RATE

        try:
            self.stream = sd.OutputStream(
                device=device_id,
                callback=self.callback,
                samplerate=self.output_rate,
                channels=CHANNELS,
                dtype=np.int16,
                blocksize=int(CHUNK_LENGTH_S * self.output_rate),
            )
            self.playing = False
            self._frame_count = 0
        except Exception as e:
            print(f"Warning: Could not initialize audio output: {e}")
            self.stream = None
            self.playing = False
            self._frame_count = 0

    def callback(self, outdata, frames, time, status):  # noqa
        with self.lock:
            data = np.empty(0, dtype=np.int16)

            # get next item from queue if there is still space in the buffer
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)

            # fill the rest of the frames with zeros if there is no more data
            if len(data) < frames:
                data = np.concatenate(
                    (data, np.zeros(frames - len(data), dtype=np.int16))
                )

        outdata[:] = data.reshape(-1, 1)

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def add_data(self, data: bytes):
        if self.stream is None:
            return
        with self.lock:
            # Convert incoming audio data to the correct sample rate
            if self.input_rate != self.output_rate:
                # Convert bytes to AudioSegment
                audio_segment = AudioSegment(
                    data=data,
                    sample_width=2,
                    frame_rate=self.input_rate,
                    channels=CHANNELS
                )
                # Resample to output rate
                resampled = audio_segment.set_frame_rate(self.output_rate)
                np_data = np.frombuffer(resampled.raw_data, dtype=np.int16)
            else:
                np_data = np.frombuffer(data, dtype=np.int16)
                
            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        if self.stream is None:
            print("Warning: Audio output not available")
            return
        self.playing = True
        self.stream.start()

    def stop(self):
        if self.stream is None:
            return
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []

    def terminate(self):
        if self.stream is not None:
            self.stream.close()


async def send_audio_worker_sounddevice(
    connection: AsyncRealtimeConnection,
    should_send: Callable[[], bool] | None = None,
    start_send: Callable[[], Awaitable[None]] | None = None,
):
    sent_audio = False

    device_info = sd.query_devices()
    print(device_info)

    read_size = int(SAMPLE_RATE * 0.02)

    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16",
    )
    stream.start()

    try:
        while True:
            if stream.read_available < read_size:
                await asyncio.sleep(0)
                continue

            data, _ = stream.read(read_size)

            if should_send() if should_send else True:
                if not sent_audio and start_send:
                    await start_send()
                await connection.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode("utf-8"),
                    }
                )
                sent_audio = True

            elif sent_audio:
                print("Done, triggering inference")
                await connection.send({"type": "input_audio_buffer.commit"})
                await connection.send(
                    {"type": "response.create", "response": {}}
                )
                sent_audio = False

            await asyncio.sleep(0)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
