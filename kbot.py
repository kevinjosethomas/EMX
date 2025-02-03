import asyncio
from src.robot import Robot
from src.emotion.expressions import Happy, Sad, Neutral

robot = Robot()


@robot.event("ready")
async def on_ready():
    print("Robot is ready")


@robot.event("face_appeared")
async def on_face_appeared():
    print("A face appeared!")
    await robot.emotion.queue_animation(Happy())


@robot.event("face_disappeared")
async def on_face_disappeared():
    print("Face disappeared!")
    await robot.emotion.queue_animation(Sad())


@robot.event("face_tracked")
async def on_face_tracked(face_data):
    pass


@robot.event("expression_started")
async def on_expression_started(data):
    print(f"Started expression: {data['label']}")


@robot.event("expression_completed")
async def on_expression_completed(data):
    print(f"Completed expression: {data['label']}")


@robot.event("idle_started")
async def on_idle_started():
    print("Entered idle state")


@robot.event("idle_ended")
async def on_idle_ended():
    print("Exited idle state")


if __name__ == "__main__":
    asyncio.run(robot.run())
