import asyncio
from emotion.engine import Engine
from emotion.expressions import Happy, Sad


async def main():
    engine = Engine()

    # Start the engine
    asyncio.create_task(engine.run())

    # await engine.queue_animation(
    #     Happy(
    #         duration=1.5,
    #         transition_duration=0.3,
    #         interpolation="ease-in-out",
    #     ),
    # )

    # await engine.queue_animation(
    #     Sad(
    #         duration=2.0,
    #         transition_duration=0.2,
    #         interpolation="linear",
    #     ),
    # )

    while engine.running:
        await asyncio.sleep(0.1)  # Keep the script alive


asyncio.run(main())
