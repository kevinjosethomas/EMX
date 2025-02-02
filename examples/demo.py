import asyncio
from expression.engine import Engine
from expression.expressions import Happy, Sad


async def main():
    engine = Engine()

    # Start the engine
    asyncio.create_task(engine.run())

    # Simulate an LLM dynamically triggering expressions
    await asyncio.sleep(1)
    await engine.queue_animation(
        Happy(),
        transition_duration=0.3,
        animation_duration=1.5,
        interpolation="ease-in-out",
    )

    await asyncio.sleep(3)
    await engine.queue_animation(
        Sad(),
        transition_duration=0.2,
        animation_duration=2.0,
        interpolation="linear",
    )

    while engine.running:
        await asyncio.sleep(0.1)  # Keep the script alive


asyncio.run(main())
