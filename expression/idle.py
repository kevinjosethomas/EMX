import random
import time
from .expressions import Neutral, Blink


class IdlingState:
    """
    Handles idling animations when no expressions are queued.
    - Randomly selects tasks such as blinking.
    - Can later include looking around, reacting to environment, etc.
    """

    def __init__(self):
        self.current_expression = Neutral()  # Default idle state
        self.last_idle_time = time.time()  # Track last idle action
        self.next_idle_action_time = self.get_next_idle_time()
        self.blinking = False

    def get_next_idle_time(self):
        """Returns a random time between 3-5 seconds for the next action."""
        return time.time() + random.uniform(5, 7)

    def get_idle_expression(self):
        """Determines what the face should do during idle time."""
        if self.blinking:
            self.blinking = False
            self.current_expression = Neutral()
        elif time.time() > self.next_idle_action_time:
            # Randomly select an idle action
            self.current_expression = random.choice([Blink()])
            self.blinking = True
            self.next_idle_action_time = (
                self.get_next_idle_time()
            )  # Reset timer
        return self.current_expression
