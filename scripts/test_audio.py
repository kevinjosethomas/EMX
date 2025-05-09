import pyaudio
import wave
import time
from datetime import datetime
import os


def test_audio_recording():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 10

    os.makedirs("debug_audio", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    WAVE_OUTPUT_FILENAME = f"debug_audio/recording_{timestamp}.wav"

    p = pyaudio.PyAudio()

    print("* Recording will start in 3 seconds...")
    time.sleep(3)
    print("* Recording...")

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Done recording")

    stream.stop_stream()
    stream.close()

    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print("* Playing back the recording...")

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

    wf = wave.open(WAVE_OUTPUT_FILENAME, "rb")
    data = wf.readframes(CHUNK)

    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("* Playback finished")


if __name__ == "__main__":
    try:
        test_audio_recording()
    except KeyboardInterrupt:
        print("\n* Recording interrupted by user")
    except Exception as e:
        print(f"* Error occurred: {str(e)}")
