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

        # Animation state
        self.current_expression = Neutral()
        self.target_expression = None
        self.previous_vertices = None
        self.start_time = time.perf_counter()
        self.transition_duration = 1.0
        self.animation_duration = 1.0
        self.interpolation_func = linear_interpolation
        self.is_transitioning = False

        self.idling_state = IdlingState()
        self.fps = 120

    def queue_animation(
        self,
        expression,
        transition_duration=0.2,
        animation_duration=1.0,
        interpolation="linear",
    ):
        """Queues an animation with a transition duration, animation duration, and interpolation style."""
        self.queue.queue_animation(
            expression, transition_duration, animation_duration, interpolation
        )

    def run(self):
        while self.running:
            self.screen.fill((30, 30, 30))
            current_time = time.perf_counter()
            elapsed_time = current_time - self.start_time

            # Determine current vertices to display
            if not self.is_transitioning:
                # When not transitioning, just render current expression
                interpolated_vertices = self.current_expression.render(
                    1.0,
                    self.interpolation_func,
                    self.screen_width,
                    self.screen_height,
                )

                # Check if we need to start a new transition
                if elapsed_time > self.animation_duration:
                    next_expr, next_trans_dur, next_anim_dur, interp_style = (
                        self.queue.get_next()
                    )

                    if next_expr:
                        # Store current state before transition
                        self.previous_vertices = interpolated_vertices
                        self.target_expression = next_expr
                        self.transition_duration = next_trans_dur or 1.0
                        self.animation_duration = next_anim_dur or 1.0
                        self.interpolation_func = (
                            linear_interpolation
                            if interp_style == "linear"
                            else ease_in_out_interpolation
                        )
                        self.is_transitioning = True
                        self.start_time = current_time
                    else:
                        # Check for idle state changes
                        next_idle = self.idling_state.get_idle_expression()
                        if next_idle != self.current_expression:
                            self.previous_vertices = interpolated_vertices
                            self.target_expression = next_idle
                            self.is_transitioning = True
                            self.start_time = current_time

            else:  # Handle transition state
                t = min(1.0, elapsed_time / self.transition_duration)

                if t >= 1.0:
                    # Transition complete
                    self.current_expression = self.target_expression
                    self.target_expression = None
                    self.is_transitioning = False
                    self.start_time = current_time
                    interpolated_vertices = self.current_expression.render(
                        1.0,
                        self.interpolation_func,
                        self.screen_width,
                        self.screen_height,
                    )
                else:
                    # Interpolate between previous and target vertices
                    target_vertices = self.target_expression.render(
                        1.0,
                        self.interpolation_func,
                        self.screen_width,
                        self.screen_height,
                    )
                    interpolated_vertices = [
                        (
                            self.previous_vertices[i][0] * (1 - t)
                            + target_vertices[i][0] * t,
                            self.previous_vertices[i][1] * (1 - t)
                            + target_vertices[i][1] * t,
                        )
                        for i in range(len(self.previous_vertices))
                    ]

            # Render eyes
            left_eye = interpolated_vertices[:12]
            right_eye = interpolated_vertices[12:]
            pygame.draw.polygon(self.screen, (255, 255, 255), left_eye)
            pygame.draw.polygon(self.screen, (255, 255, 255), right_eye)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
