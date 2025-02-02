import queue


class AnimationQueue:
    def __init__(self):
        self.queue = queue.Queue()

    def queue_animation(
        self,
        expression,
        transition_duration,
        animation_duration,
        interpolation,
    ):
        """Adds animation with transition duration, animation duration, and interpolation style."""
        self.queue.put(
            (
                expression,
                transition_duration,
                animation_duration,
                interpolation,
            )
        )

    def get_next(self):
        """Retrieves next animation in queue."""
        return (
            self.queue.get()
            if not self.queue.empty()
            else (None, None, None, None)
        )
