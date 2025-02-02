import pygame
import numpy as np
import time
from .queue_manager import AnimationQueue
from .interpolation import linear_interpolation, ease_in_out_interpolation
from .expressions import Neutral
from .idle import IdlingState


class Engine:
    def __init__(self, width=1024, height=600):
        pygame.init()
        self.screen_width = width
        self.screen_height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.queue = AnimationQueue()
        self.running = True

        self.current_expression = Neutral()
        self.previous_vertices = (
            None  # Store previous vertices for smooth transition
        )
        self.start_time = time.perf_counter()
        self.duration = 1.0
        self.interpolation_func = linear_interpolation

        self.idling_state = IdlingState()
        self.fps = 120  # High FPS for smooth animations

    def queue_animation(
        self, expression, duration=1.0, interpolation="linear"
    ):
        """Queues an animation with a duration and interpolation style."""
        self.queue.queue_animation(expression, duration, interpolation)

    def run(self):
        """Main event loop for rendering expressions correctly (fixes teleporting issue)."""
        while self.running:
            self.screen.fill((30, 30, 30))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            current_time = time.perf_counter()
            elapsed_time = current_time - self.start_time

            if elapsed_time > self.duration:
                # Transition to next animation
                next_expression, next_duration, interpolation_style = (
                    self.queue.get_next()
                )
                if next_expression:
                    self.previous_vertices = self.current_expression.render(
                        1.0,
                        self.interpolation_func,
                        self.screen_width,
                        self.screen_height,
                    )
                    self.current_expression = next_expression
                    self.start_time = time.perf_counter()
                    self.duration = next_duration or 1.0
                    self.interpolation_func = (
                        linear_interpolation
                        if interpolation_style == "linear"
                        else ease_in_out_interpolation
                    )
                else:
                    # No animation in queue? Use idling state
                    self.previous_vertices = self.current_expression.render(
                        1.0,
                        self.interpolation_func,
                        self.screen_width,
                        self.screen_height,
                    )
                    self.current_expression = (
                        self.idling_state.get_idle_expression()
                    )
                    self.start_time = time.perf_counter()

            # 🔹 FIX: Normalize `t` correctly to interpolate between expressions
            t = min(1.0, elapsed_time / self.duration)

            # 🔹 FIX: Interpolate between previous and current expression
            current_vertices = self.current_expression.render(
                1.0,
                self.interpolation_func,
                self.screen_width,
                self.screen_height,
            )
            if self.previous_vertices is not None:
                interpolated_vertices = [
                    (
                        self.previous_vertices[i][0] * (1 - t)
                        + current_vertices[i][0] * t,
                        self.previous_vertices[i][1] * (1 - t)
                        + current_vertices[i][1] * t,
                    )
                    for i in range(len(self.previous_vertices))
                ]
            else:
                interpolated_vertices = current_vertices

            # 🔹 FIX: Render left & right eyes separately!
            left_eye = interpolated_vertices[:12]
            right_eye = interpolated_vertices[12:]

            pygame.draw.polygon(
                self.screen, (255, 255, 255), left_eye
            )  # Render left eye
            pygame.draw.polygon(
                self.screen, (255, 255, 255), right_eye
            )  # Render right eye

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
