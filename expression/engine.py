import pygame
import numpy as np
import time
import asyncio

from .interpolation import INTERPOLATION
from .expressions import Neutral
from .idle import IdlingState


class Engine:
    def __init__(self, width=1024, height=600, fullscreen=False):
        pygame.init()
        self.screen_width = width
        self.screen_height = height
        if fullscreen:
            self.screen = pygame.display.set_mode(
                (width, height), pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.running = True

        # Async Queue for animations
        self.expression_queue = asyncio.Queue()

        # Animation state
        self.current_expression = Neutral(sticky=True)
        self.target_expression = None
        self.previous_vertices = None
        self.start_time = time.perf_counter()
        self.transition_duration = 1.0
        self.animation_duration = 1.0
        self.interpolation_func = INTERPOLATION["linear"]
        self.is_transitioning = False

        self.idling_state = IdlingState()
        self.fps = 120

    async def queue_animation(
        self,
        expression,
    ):
        await self.expression_queue.put(
            expression,
        )

    async def handle_queue(self):
        """Asynchronously handle incoming animations from the queue."""
        while self.running:
            if not self.is_transitioning and not self.expression_queue.empty():
                next_expr = await self.expression_queue.get()

                if next_expr:
                    self.previous_vertices = self.current_expression.render(
                        1.0,
                        self.interpolation_func,
                        self.screen_width,
                        self.screen_height,
                    )
                    self.target_expression = next_expr
                    self.transition_duration = (
                        next_expr.transition_duration or 1.0
                    )
                    self.animation_duration = next_expr.duration or 1.0
                    self.interpolation_func = INTERPOLATION.get(
                        next_expr.interpolation, INTERPOLATION["linear"]
                    )
                    self.is_transitioning = True
                    self.start_time = time.perf_counter()

            await asyncio.sleep(0.01)  # Avoid excessive CPU usage

    async def run(self):
        """Main event loop for rendering expressions with smooth transitions."""
        asyncio.create_task(self.handle_queue())  # Start queue handling

        while self.running:
            self.screen.fill((30, 30, 30))
            current_time = time.perf_counter()
            elapsed_time = current_time - self.start_time

            # Determine current vertices to display
            if not self.is_transitioning:
                interpolated_vertices = self.current_expression.render(
                    1.0,
                    self.interpolation_func,
                    self.screen_width,
                    self.screen_height,
                )

                if elapsed_time > self.animation_duration:
                    next_idle = self.idling_state.get_idle_expression()
                    if next_idle != self.current_expression:
                        self.previous_vertices = interpolated_vertices
                        self.target_expression = next_idle
                        self.is_transitioning = True
                        self.start_time = current_time
                        self.transition_duration = (
                            next_idle.transition_duration
                        )
                        self.animation_duration = next_idle.duration
                        self.interpolation_func = INTERPOLATION.get(
                            next_idle.interpolation,
                            INTERPOLATION["linear"],
                        )

            else:  # Handle transition state
                t = min(1.0, elapsed_time / self.transition_duration)

                if t >= 1.0:
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

            left_eye = interpolated_vertices[:12]
            right_eye = interpolated_vertices[12:]
            pygame.draw.polygon(self.screen, (255, 255, 255), left_eye)
            pygame.draw.polygon(self.screen, (255, 255, 255), right_eye)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            pygame.display.flip()
            self.clock.tick(self.fps)
            await asyncio.sleep(0)  # Allow other tasks to run

        pygame.quit()
