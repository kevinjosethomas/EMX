import asyncio
import traceback
from pyee.asyncio import AsyncIOEventEmitter
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import play


class Voice(AsyncIOEventEmitter):
    """Voice synthesis engine using ElevenLabs API.

    Handles text-to-speech conversion and audio playback using the ElevenLabs API.
    Provides asynchronous methods for fetching and playing synthesized speech.

    Events:
        speech_ready: Emitted when audio data is ready to play with data:
            - audio: The raw audio bytes
        speech_error: Emitted when an error occurs with data:
            - error: Error message with traceback

    Attributes:
        api_key (str): ElevenLabs API key
        voice_id (str): ID of the voice to use for synthesis
        client (AsyncElevenLabs): Async client for ElevenLabs API
    """

    def __init__(self, api_key, voice_id):
        """Initialize the voice engine.

        Args:
            api_key (str): ElevenLabs API key
            voice_id (str): ID of the voice to use for synthesis
        """

        super().__init__()
        self.api_key = api_key
        self.voice_id = voice_id
        self.client = AsyncElevenLabs(api_key=self.api_key)

    def _format_error(self, e: Exception) -> str:
        """Format an exception with traceback information.

        Args:
            e (Exception): The exception to format

        Returns:
            str: Formatted error message with filename, line number and traceback
        """

        tb = traceback.extract_tb(e.__traceback__)
        error_location = f"{tb[-1].filename}:{tb[-1].lineno}"
        return f"Error in {error_location}: {str(e)}\n{traceback.format_exc()}"

    async def fetch_voice(self, text):
        """Fetch synthesized voice data from ElevenLabs API.

        Args:
            text (str): The text to convert to speech

        Returns:
            AsyncGenerator: Generator yielding audio data chunks
            None: If an error occurs during synthesis

        Emits:
            speech_error: If an error occurs during API call
        """

        try:

            return self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_flash_v2_5",
                output_format="mp3_44100_128",
            )
        except Exception as e:
            error_msg = self._format_error(e)
            print(error_msg)
            # self.emit("speech_error", error_msg)
            return None

    async def collect_audio(self, generator):
        """Collect audio data from async generator.

        Args:
            generator (AsyncGenerator): Generator yielding audio chunks

        Returns:
            bytes: Concatenated audio data
            None: If an error occurs during collection

        Emits:
            speech_error: If an error occurs while collecting audio chunks
        """

        try:
            audio_bytes = b""
            async for chunk in generator:
                audio_bytes += chunk
            return audio_bytes
        except Exception as e:
            error_msg = self._format_error(e)
            print(error_msg)
            # self.emit("speech_error", error_msg)
            return None

    async def play_voice(self, audio):
        """Play collected audio data.

        Args:
            audio (bytes): Audio data to play

        Emits:
            speech_ready: When audio playback starts
            speech_error: If an error occurs during playback
        """

        try:
            await asyncio.to_thread(play, audio)
            # self.emit("speech_ready", audio)
        except Exception as e:
            error_msg = self._format_error(e)
            print(error_msg)
            # self.emit("speech_error", error_msg)

    async def speak(self, text):
        """Convert text to speech and play it.

        High-level method that handles the complete text-to-speech pipeline:
        1. Fetch synthesized voice from API
        2. Collect audio data chunks
        3. Play the audio

        Args:
            text (str): The text to convert to speech

        Events are emitted by the individual methods called in the pipeline.
        """

        generator = await self.fetch_voice(text)
        if generator:
            audio = await self.collect_audio(generator)
            if audio:
                await self.play_voice(audio)
