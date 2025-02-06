import os
import asyncio
import dotenv
from src.robot import Robot
from src.emotion.expressions import Happy, Neutral

dotenv.load_dotenv()

robot = Robot(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    debug=True,
)


@robot.event("ready")
async def on_ready():
    print("Robot is ready")
    # await robot.voice.speak("Hello, I am ready to assist you.")


@robot.event("face_appeared")
async def on_face_appeared():
    pass
    # await robot.emotion.queue_animation(Happy())


@robot.event("face_disappeared")
async def on_face_disappeared():
    pass
    # await robot.emotion.queue_animation(Sad())


# @robot.event("face_tracked")
# async def on_face_tracked(face_data):
#     pass


# @robot.event("expression_started")
# async def on_expression_started(expression):
#     print(f"Started expression: {expression.label}")


# @robot.event("expression_completed")
# async def on_expression_completed(expression):
#     print(f"Completed expression: {expression.label}")


@robot.event("gesture_detected")
async def on_gesture(gesture_data):
    print(f"Detected gesture: {gesture_data['gesture']}")
    # React to specific gestures
    if gesture_data["gesture"] == "thumbs_up":
        await robot.emotion.queue_animation(Happy())
    elif gesture_data["gesture"] == "wave":
        await robot.emotion.queue_animation(Neutral())


if __name__ == "__main__":
    asyncio.run(robot.run())
