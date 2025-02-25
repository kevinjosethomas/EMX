#!/usr/bin/env python3
"""
Configuration Generator for K-Emotion

This script creates or updates the configuration file for the K-Emotion application.
It detects available audio and video devices and creates a config.json file
with the appropriate settings.

Usage:
    python generate_config.py
"""

import os
import sys
import json
from src.config import create_config, load_config, save_config


def display_options(devices, device_type):
    """Display available device options to the user.

    Args:
        devices (list): List of device dictionaries
        device_type (str): Type of device (microphone, speaker, camera)
    """
    print(f"\nAvailable {device_type} devices:")
    for i, device in enumerate(devices):
        if device_type == "camera":
            print(f"  [{i}] Camera ID: {device['id']}")
        else:
            print(f"  [{i}] {device['name']} (ID: {device['id']})")

    print(f"  [s] Skip {device_type} selection (use default)")


def get_user_selection(devices, device_type, current_id=None):
    """Get user selection for a device.

    Args:
        devices (list): List of device dictionaries
        device_type (str): Type of device (microphone, speaker, camera)
        current_id (int, optional): Current device ID

    Returns:
        int or None: Selected device ID or None if skipped
    """
    while True:
        display_options(devices, device_type)

        if current_id is not None:
            current_name = "Unknown"
            for device in devices:
                if device["id"] == current_id:
                    if device_type == "camera":
                        current_name = f"Camera ID {current_id}"
                    else:
                        current_name = device["name"]
            print(
                f"\nCurrent {device_type}: {current_name} (ID: {current_id})"
            )

        choice = input(
            f"\nSelect {device_type} [0-{len(devices)-1}, s to skip]: "
        )

        if choice.lower() == "s":
            return current_id

        try:
            idx = int(choice)
            if 0 <= idx < len(devices):
                return devices[idx]["id"]
            else:
                print(
                    f"Invalid selection. Please choose between 0 and {len(devices)-1}."
                )
        except ValueError:
            print("Invalid input. Please enter a number or 's'.")


def get_float_input(prompt, current_value, min_val=0.0, max_val=1.0):
    """Get a float input from the user.

    Args:
        prompt (str): Prompt to display to user
        current_value (float): Current value
        min_val (float): Minimum allowed value
        max_val (float): Maximum allowed value

    Returns:
        float: User input or current value if unchanged
    """
    while True:
        user_input = input(
            f"{prompt} [{min_val}-{max_val}, current: {current_value}]: "
        )

        if not user_input.strip():
            return current_value

        try:
            value = float(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_bool_input(prompt, current_value):
    """Get a boolean input from the user.

    Args:
        prompt (str): Prompt to display to user
        current_value (bool): Current value

    Returns:
        bool: User input or current value if unchanged
    """
    user_input = input(
        f"{prompt} (y/n) [current: {'y' if current_value else 'n'}]: "
    )

    if not user_input.strip():
        return current_value

    return user_input.lower() == "y"


def get_environment_input(current_value):
    """Get environment type from the user.

    Args:
        current_value (str): Current environment value

    Returns:
        str: Selected environment
    """
    while True:
        print("\nEnvironment options:")
        print("  [1] default - Standard desktop/laptop")
        print("  [2] pi - Raspberry Pi")

        print(f"\nCurrent environment: {current_value}")

        choice = input("\nSelect environment [1-2, Enter to keep current]: ")

        if not choice.strip():
            return current_value

        if choice == "1":
            return "default"
        elif choice == "2":
            return "pi"
        else:
            print("Invalid selection. Please choose 1 or 2.")


def main():
    """Main configuration generator function."""
    print("\n" + "=" * 60)
    print("K-Emotion Configuration Generator")
    print("=" * 60)

    config_exists = os.path.exists("config.json")

    if config_exists:
        print("\nExisting configuration found.")
        config = load_config()
        print("Current configuration:")
        print("\nDevice Settings:")
        print(f"  Microphone ID: {config.get('microphone_id')}")
        print(f"  Speaker ID: {config.get('speaker_id')}")
        print(f"  Camera ID: {config.get('camera_id')}")

        print("\nVoice Settings:")
        print(f"  Volume: {config.get('volume', 0.15)}")

        print("\nVision Settings:")
        print(
            f"  Face Detection Confidence: {config.get('face_detection_confidence', 0.5)}"
        )
        print(
            f"  Face Tracking Threshold: {config.get('face_tracking_threshold', 0.3)}"
        )

        print("\nEmotion Settings:")
        print(f"  Fullscreen: {config.get('fullscreen', False)}")
        print(f"  Animation Speed: {config.get('animation_speed', 1.0)}")
        print(f"  Idle Timeout: {config.get('idle_timeout', 5.0)} seconds")

        print("\nGeneral Settings:")
        print(f"  Debug Mode: {config.get('debug', False)}")
        print(f"  Environment: {config.get('environment', 'default')}")

        if (
            input(
                "\nDo you want to update this configuration? (y/n): "
            ).lower()
            != "y"
        ):
            print("Configuration unchanged. Exiting.")
            return
    else:
        print("\nNo configuration found. Creating new configuration.")
        config = create_config()

    from src.config import (
        get_available_microphones,
        get_available_speakers,
        get_available_cameras,
    )

    print("\n" + "=" * 60)
    print("DEVICE CONFIGURATION")
    print("=" * 60)

    microphones = get_available_microphones()
    speakers = get_available_speakers()
    cameras = get_available_cameras()

    if microphones:
        mic_id = get_user_selection(
            microphones, "microphone", config.get("microphone_id")
        )
        config["microphone_id"] = mic_id
    else:
        print("No microphones detected!")
        config["microphone_id"] = None

    if speakers:
        speaker_id = get_user_selection(
            speakers, "speaker", config.get("speaker_id")
        )
        config["speaker_id"] = speaker_id
    else:
        print("No speakers detected!")
        config["speaker_id"] = None

    if cameras:
        camera_id = get_user_selection(
            cameras, "camera", config.get("camera_id")
        )
        config["camera_id"] = camera_id
    else:
        print("No cameras detected!")
        config["camera_id"] = None

    print("\n" + "=" * 60)
    print("VOICE SETTINGS")
    print("=" * 60)

    config["volume"] = get_float_input(
        "\nSet volume", config.get("volume", 0.15)
    )

    print("\n" + "=" * 60)
    print("VISION SETTINGS")
    print("=" * 60)

    config["face_detection_confidence"] = get_float_input(
        "\nFace detection confidence threshold",
        config.get("face_detection_confidence", 0.5),
    )

    config["face_tracking_threshold"] = get_float_input(
        "\nFace tracking confidence threshold",
        config.get("face_tracking_threshold", 0.3),
    )

    print("\n" + "=" * 60)
    print("EMOTION SETTINGS")
    print("=" * 60)

    config["fullscreen"] = get_bool_input(
        "\nEnable fullscreen mode", config.get("fullscreen", False)
    )

    config["animation_speed"] = get_float_input(
        "\nAnimation speed multiplier",
        config.get("animation_speed", 1.0),
        min_val=0.1,
        max_val=3.0,
    )

    config["idle_timeout"] = get_float_input(
        "\nIdle timeout (seconds)",
        config.get("idle_timeout", 5.0),
        min_val=1.0,
        max_val=60.0,
    )

    print("\n" + "=" * 60)
    print("GENERAL SETTINGS")
    print("=" * 60)

    config["debug"] = get_bool_input(
        "\nEnable debug mode", config.get("debug", False)
    )

    config["environment"] = get_environment_input(
        config.get("environment", "default")
    )

    save_config(config)
    print("\nConfiguration saved successfully!")
    print(f"Configuration file: {os.path.abspath('config.json')}")


if __name__ == "__main__":
    main()
