import asyncio
import random
from .expressions import Blink


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
        self.running = True

    async def run_idle_loop(self):
        """Background task that periodically queues idle animations.

        Runs continuously while self.running is True. Only queues blink animations
        when no other non-neutral expressions are in the queue.
        """

        while self.running:
            if self.emotion_engine.expression_queue.empty():
                next_expression = Blink(
                    duration=0.05,
                    transition_duration=0.05,
                    interpolation="ease_in_out",
                )
                await self.emotion_engine.queue_animation(next_expression)
            await asyncio.sleep(random.uniform(3, 5))
