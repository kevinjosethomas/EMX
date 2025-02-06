from pydub import AudioSegment


def save_raw_to_wav(raw_file: str, wav_file: str):
    """Convert raw PCM16 audio file to WAV format"""
    with open(raw_file, "rb") as f:
        raw_data = f.read()

    audio = AudioSegment(
        data=raw_data,
        sample_width=2,
        frame_rate=24000,
        channels=1,
    )
    audio.export(wav_file, format="wav")
