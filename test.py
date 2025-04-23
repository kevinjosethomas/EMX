import sounddevice as sd
import numpy as np
import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def list_devices():
    devices = sd.query_devices()
    
    print("Available Audio Devices:")
    print("-----------------------")
    
    mics = []
    speakers = []
    
    for i, device in enumerate(devices):
        device_info = f"{i}: {device['name']} "
        if device['max_input_channels'] > 0:
            device_info += "(Microphone)"
            mics.append(i)
        if device['max_output_channels'] > 0:
            device_info += "(Speaker)"
            speakers.append(i)
        print(device_info)
    
    return mics, speakers

def select_device(device_type, available_devices):
    while True:
        try:
            selection = int(input(f"\nSelect a {device_type} (enter number): "))
            if selection in available_devices:
                return selection
            else:
                print(f"Invalid selection. Please choose a valid {device_type}.")
        except ValueError:
            print("Please enter a number.")

def record_audio(input_device, output_device, duration=10, sample_rate=44100):
    print(f"\nRecording for {duration} seconds... Speak now!")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Recording...")
    
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=input_device
    )
    
    sd.wait()
    
    print("Recording complete!")
    
    return recording, sample_rate

def play_audio(recording, sample_rate, output_device):
    print("\nPlaying back recording...")
    
    sd.play(recording, sample_rate, device=output_device)
    
    sd.wait()
    
    print("Playback complete!")

def main():
    clear_screen()
    print("Audio Recording and Playback Tool")
    print("================================\n")
    
    mics, speakers = list_devices()
    
    if not mics:
        print("No microphones found.")
        return
    
    if not speakers:
        print("No speakers found.")
        return
    
    input_device = select_device("microphone", mics)
    output_device = select_device("speaker", speakers)
    
    print(f"\nSelected microphone: {sd.query_devices(input_device)['name']}")
    print(f"Selected speaker: {sd.query_devices(output_device)['name']}")
    
    recording, sample_rate = record_audio(input_device, output_device)
    
    play_audio(recording, sample_rate, output_device)
    
    print("\nDone! Thanks for using the audio tool.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
