import base64
import asyncio
import soundfile as sf
import sounddevice as sd
from typing import cast, Any
from openai import AsyncOpenAI
from pyee.asyncio import AsyncIOEventEmitter
from .audio import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection


from funasr import AutoModel
import io
from pydub import AudioSegment


class Voice(AsyncIOEventEmitter):
    def __init__(
        self,
        openai_api_key,
        microphone_id=None,
    ):
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
        self.emotion_chunk_size = 3
        self.chunk_counter = 0

        self.emotion_model = AutoModel(model="iic/emotion2vec_plus_base")

    async def analyze_audio_emotion(self, audio_bytes):
        """Analyze emotion in audio bytes by converting to correct format first."""

        try:

            audio = AudioSegment(
                data=audio_bytes,
                sample_width=2,
                frame_rate=24000,
                channels=1,
            )

            audio = audio.set_frame_rate(16000)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_bytes = wav_buffer.getvalue()

            result = self.emotion_model.generate(
                wav_bytes,
                output_dir=None,
                granularity="utterance",
                extract_embedding=False,
                disable_pbar=True,
            )

            return result

        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return None

    async def run(self):
        asyncio.create_task(self.send_mic_audio())

        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview"
        ) as conn:
            self.connection = conn
            self.connected.set()
            self.should_send_audio.clear()
            audio_buffer = io.BytesIO()

            async for event in conn:
                # print(f"Received event type: {event.type}")

                if event.type == "session.created":
                    print("session.created")
                    self.session = event.session
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
                    continue

                if event.type == "session.updated":
                    print("session.updated")
                    self.session = event.session
                    continue

                if event.type == "response.audio.delta":

                    self.should_send_audio.clear()

                    if event.item_id != self.last_audio_item_id:
                        audio_buffer = io.BytesIO()
                        self.emotion_buffer = io.BytesIO()
                        self.chunk_counter = 0
                        self.last_audio_item_id = event.item_id

                    audio_bytes = base64.b64decode(event.delta)

                    audio_buffer.write(audio_bytes)
                    self.emotion_buffer.write(audio_bytes)
                    self.chunk_counter += 1

                    if self.chunk_counter >= self.emotion_chunk_size:
                        emotion_audio = self.emotion_buffer.getvalue()

                        audio_duration = len(emotion_audio) / (24000 * 2)

                        emotion = await self.analyze_audio_emotion(
                            emotion_audio
                        )
                        if emotion:
                            scores = emotion[0]["scores"]

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

                            detected_emotion = emotion_labels[
                                scores.index(max(scores))
                            ]

                            # Emit both emotion and duration
                            self.emit(
                                "_assistant_message",
                                {
                                    "emotion": detected_emotion,
                                    "duration": audio_duration,
                                },
                            )

                        self.emotion_buffer = io.BytesIO()
                        self.chunk_counter = 0

                        self.emotion_buffer = io.BytesIO()
                        self.chunk_counter = 0

                    self.audio_player.add_data(audio_bytes)

                if event.type == "response.done":
                    self.emit("_assistant_message_end")
                    asyncio.create_task(self.wait_for_audio_completion())
                    continue

                if event.type == "error":
                    print(event.error)
                    continue

    async def wait_for_audio_completion(self):
        """Wait for audio player to finish playing queued audio before re-enabling mic"""
        while len(self.audio_player.queue) > 0:
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.1)
        self.should_send_audio.set()

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        sent_audio = False
        read_size = int(SAMPLE_RATE * 0.1)

        debug_file = sf.SoundFile(
            "debug_mic_input.wav",
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            format="WAV",
            subtype="PCM_16",
        )

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

                frames_available = stream.read_available
                if frames_available >= read_size:
                    data, _ = stream.read(read_size)

                    debug_file.write(data)

                    connection = await self._get_connection()

                    if not sent_audio:
                        await connection.send({"type": "response.cancel"})
                        sent_audio = True

                    await connection.input_audio_buffer.append(
                        audio=base64.b64encode(cast(Any, data)).decode("utf-8")
                    )

        except Exception as e:
            print(f"Error in audio capture: {e}")
        finally:
            stream.stop()
            stream.close()
