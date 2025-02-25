import os
import json
import sounddevice as sd
import cv2
from pathlib import Path

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    # Device IDs
    "microphone_id": None,
    "camera_id": None,
    "speaker_id": None,
    # Voice settings
    "volume": 0.15,
    # Vision settings
    "face_detection_confidence": 0.5,
    "face_tracking_threshold": 0.3,
    # Emotion settings
    "fullscreen": False,
    "animation_speed": 1.0,
    "idle_timeout": 5.0,
    # General settings
    "debug": False,
    "environment": "default",  # or "pi" for Raspberry Pi
}


def get_available_microphones():
    """Get a list of available microphone devices.

    Returns:
        list: List of dicts containing device info with 'id', 'name', and 'channels'
    """
    devices = []
    device_list = sd.query_devices()

    for i, device in enumerate(device_list):
        if device["max_input_channels"] > 0:
            devices.append(
                {
                    "id": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                }
            )

    return devices


def get_available_speakers():
    """Get a list of available speaker devices.

    Returns:
        list: List of dicts containing device info with 'id', 'name', and 'channels'
    """
    devices = []
    device_list = sd.query_devices()

    for i, device in enumerate(device_list):
        if device["max_output_channels"] > 0:
            devices.append(
                {
                    "id": i,
                    "name": device["name"],
                    "channels": device["max_output_channels"],
                }
            )

    return devices


def get_available_cameras():
    """Get a list of available camera devices.

    Returns:
        list: List of dicts containing camera info with 'id'
    """
    cameras = []
    index = 0

    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cameras.append({"id": index})
            cap.release()
        index += 1

    return cameras


def select_default_device(devices, device_type):
    """Select a default device based on available devices.

    Args:
        devices (list): List of available devices
        device_type (str): Type of device ('microphone', 'speaker', or 'camera')

    Returns:
        int or None: Device ID or None if no devices available
    """
    if not devices:
        return None

    if device_type == "microphone":
        for device in devices:
            if "default" in device["name"].lower():
                return device["id"]

    if device_type == "speaker":
        for device in devices:
            if "default" in device["name"].lower():
                return device["id"]

    return devices[0]["id"]


def create_config():
    """Create a new configuration file with default values.

    Returns:
        dict: The newly created configuration
    """
    config = DEFAULT_CONFIG.copy()

    microphones = get_available_microphones()
    speakers = get_available_speakers()
    cameras = get_available_cameras()

    config["microphone_id"] = select_default_device(microphones, "microphone")
    config["speaker_id"] = select_default_device(speakers, "speaker")
    config["camera_id"] = select_default_device(cameras, "camera")

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Created configuration file: {CONFIG_FILE}")
    print(f"Selected microphone: {config['microphone_id']}")
    print(f"Selected speaker: {config['speaker_id']}")
    print(f"Selected camera: {config['camera_id']}")

    return config


def load_config():
    """Load configuration from file or create default if it doesn't exist.

    Returns:
        dict: Configuration values
    """
    config_path = Path(CONFIG_FILE)

    if not config_path.exists():
        print(f"Configuration file not found. Creating new config.")
        return create_config()

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        for key in DEFAULT_CONFIG:
            if key not in config:
                config[key] = DEFAULT_CONFIG[key]

        return config

    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Creating new configuration file.")
        return create_config()


def save_config(config):
    """Save configuration to file.

    Args:
        config (dict): Configuration to save
    """
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {CONFIG_FILE}")


def get_config():
    """Get the current configuration or create if needed.

    Returns:
        dict: The current configuration
    """
    return load_config()


if __name__ == "__main__":
    config = create_config()
    print(
        "Configuration created. You can edit the config.json file to change settings."
    )
