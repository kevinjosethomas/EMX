import time
import random
import asyncio
from .expressions import Neutral, Blink


class IdleAnimationManager:
    """Manages idle animations like blinking when no other expressions are active.

    Runs a background loop that periodically queues blink animations when no other
    expressions are queued. Only queues blinks when the expression queue is empty
    or contains only neutral expressions.
    """

    def __init__(self, emotion_engine):
        """Initialize the idle animation manager.

        Args:
            emotion_manager (Emotion): Reference to main emotion manager for queueing animations
        """

        self.emotion_engine = emotion_engine
        self.running = False
        self.last_squint = time.time()

    async def run_idle_loop(self):
        """Background task that periodically queues idle animations.

        Runs continuously while self.running is True. Only queues idle animations
        when no other non-neutral expressions are in the queue.
        """

        while True:
            if self.running and self.emotion_engine.expression_queue.empty():
                current_time = time.time()

                if current_time - self.last_squint >= random.uniform(10, 15):
                    loc = (
                        random.uniform(-0.5, 0.5),
                        random.uniform(-0.5, 0.5),
                    )
                    self.last_squint = current_time

                    if random.uniform(0, 1) > 0.5:
                        await self.emotion_engine.queue_animation(
                            Neutral(
                                duration=random.uniform(1, 4),
                                transition_duration=0.1,
                                interpolation="linear",
                                position=loc,
                                scale=1.1,
                            )
                        )
                    else:
                        await self.emotion_engine.queue_animation(
                            Neutral(
                                duration=random.uniform(1, 2),
                                transition_duration=0.1,
                                interpolation="linear",
                                position=loc,
                            )
                        )
                else:
                    next_expression = Blink(
                        duration=0.05,
                        transition_duration=0.05,
                        interpolation="ease_in_out",
                    )

                    await self.emotion_engine.queue_animation(next_expression)

            await asyncio.sleep(random.uniform(4, 6))
