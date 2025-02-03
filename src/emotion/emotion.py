import time
import pygame
import asyncio
from .idle import IdlingState
from .expressions import Neutral
from .interpolation import INTERPOLATION
from pyee.asyncio import AsyncIOEventEmitter


class Emotion(AsyncIOEventEmitter):
    """Manages facial expressions and animations for the robot.

    Controls the display and rendering of facial expressions, handles transitions
    between expressions, and manages the animation queue. Uses pygame for rendering
    and asyncio for animation scheduling.

    Attributes:
        screen_width (int): Width of display window in pixels
        screen_height (int): Height of display window in pixels
        screen (pygame.Surface): Pygame display surface
        running (bool): Flag indicating if animation loop is running
        expression_queue (asyncio.Queue): Queue of pending expressions to display
        current_expression (Expression): Currently displayed expression
        target_expression (Expression): Expression being transitioned to
        previous_vertices (np.ndarray): Previous frame's vertex positions
        is_transitioning (bool): Flag indicating if in transition
        fps (int): Target frames per second
    """

    def __init__(self, width=1024, height=600, fullscreen=False):
        """Initialize the emotion display system.

        Args:
            width (int, optional): Display width in pixels. Defaults to 1024.
            height (int, optional): Display height in pixels. Defaults to 600.
            fullscreen (bool, optional): Whether to use fullscreen. Defaults to False.

        Initializes pygame display, animation state variables, and expression queue.
        Sets up initial neutral expression and idle animation state.
        """

        super().__init__()
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

        self.expression_queue = asyncio.Queue()

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
        """Queue a new expression for display.

        Args:
            expression (Expression): Expression object to be displayed.
                Must implement render() method.

        The expression will be added to the queue and displayed when previous
        animations complete. Transitions between expressions are handled automatically.
        """

        await self.expression_queue.put(
            expression,
        )
        self.emit(
            "expression_started",
            {
                "id": expression.id,
                "label": expression.label,
                "duration": expression.duration,
                "transition_duration": expression.transition_duration,
                "interpolation": expression.interpolation,
            },
        )

    async def handle_queue(self):
        """Process queued expressions and handle transitions.

        Continuously monitors the expression queue and initiates transitions
        to new expressions when they become available. Manages the transition
        state and timing between expressions.

        This method runs as a background task and continues while self.running
        is True. It handles:
        - Checking for new expressions in queue
        - Starting transitions between expressions
        - Updating transition state and timing
        """

        while self.running:
            if not self.is_transitioning and not self.expression_queue.empty():
                next_expr = await self.expression_queue.get()

                if next_expr:
                    if self.current_expression.sticky:
                        self.emit("idle_ended")

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

            await asyncio.sleep(0.01)

    async def run(self):
        """Main animation loop for rendering expressions.

        Continuously renders the current expression and handles transitions.
        Manages the display window, animation timing, and idle animations.
        Coordinates with handle_queue() for processing new expressions.

        The loop:
        - Clears screen and maintains consistent frame timing
        - Renders current expression or transition state
        - Handles idle animations when expression completes
        - Maintains synchronized state with expression queue

        This method runs until self.running is set to False.
        Screen updates occur at the rate specified by self.fps.
        """

        asyncio.create_task(self.handle_queue())

        while self.running:
            self.screen.fill((30, 30, 30))
            current_time = time.perf_counter()
            elapsed_time = current_time - self.start_time

            if not self.is_transitioning:
                interpolated_vertices = self.current_expression.render(
                    1.0,
                    self.interpolation_func,
                    self.screen_width,
                    self.screen_height,
                )

                if elapsed_time > self.animation_duration:
                    if not self.current_expression.sticky:
                        self.emit(
                            "expression_completed",
                            {
                                "id": self.current_expression.id,
                                "label": self.current_expression.label,
                                "duration": self.current_expression.duration,
                            },
                        )

                    next_idle = self.idling_state.get_idle_expression()
                    if next_idle != self.current_expression:
                        if next_idle.sticky:
                            self.emit("idle_started")

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

            else:
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
            await asyncio.sleep(0)

        pygame.quit()
