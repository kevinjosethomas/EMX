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
CHUNK_LENGTH_S = 0.1 


def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
    
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    print(
        f"Loaded audio: {audio.frame_rate=} {audio.channels=} {audio.sample_width=} {audio.frame_width=}"
    )
    
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
        self.volume = 0.15
        self.buffer_size = 4096
        self.min_buffer_fill = 0.5

        devices = sd.query_devices()
        print("Available audio devices:")
        print(devices)

        try:
            default_device = sd.query_devices(kind="output")
            device_id = default_device["index"]
            print(f"Using output device: {default_device['name']}")

            supported_rates = default_device.get("default_samplerate")
            print(f"Device default sample rate: {supported_rates}")

            self.input_rate = SAMPLE_RATE
            self.output_rate = (
                int(supported_rates) if supported_rates else SAMPLE_RATE
            )
            print(
                f"Input rate: {self.input_rate}, Output rate: {self.output_rate}"
            )

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

            data = (data * self.volume).astype(np.int16)

        outdata[:] = data.reshape(-1, 1)

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def add_data(self, data: bytes):
        if self.stream is None:
            return
            
        with self.lock:
            # Convert incoming audio data
            if self.input_rate != self.output_rate:
                audio_segment = AudioSegment(
                    data=data,
                    sample_width=2,
                    frame_rate=self.input_rate,
                    channels=CHANNELS,
                )
                resampled = audio_segment.set_frame_rate(self.output_rate)
                np_data = np.frombuffer(resampled.raw_data, dtype=np.int16)
            else:
                np_data = np.frombuffer(data, dtype=np.int16)
                
            # Add to queue with buffering
            self.queue.append(np_data)
            buffer_fill = sum(len(x) for x in self.queue) / self.buffer_size
            
            if not self.playing and buffer_fill >= self.min_buffer_fill:
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

    def set_volume(self, volume: float):
        """Set the playback volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
