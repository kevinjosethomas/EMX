import time
import pygame
import asyncio
from .expressions import Neutral
from .idle import IdleAnimationManager
from .interpolation import INTERPOLATION
from pyee.asyncio import AsyncIOEventEmitter


class Emotion(AsyncIOEventEmitter):
    """Manages facial expressions and animations for the robot.

    Controls the display and rendering of facial expressions, handles transitions
    between expressions, and manages the animation queue. Uses pygame for rendering
    and asyncio for animation scheduling.

    Events:
        expression_started: Emitted when a new expression begins playing with data:
            - id: Unique identifier for the expression
            - label: Name of the expression
            - duration: Duration of the expression in seconds
            - transition_duration: Duration of transition in seconds
            - interpolation: Type of interpolation used
        expression_completed: Emitted when an expression finishes with data:
            - id: Unique identifier for the expression
            - label: Name of the expression
            - duration: Duration of the expression in seconds

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
        Sets up initial neutral expression and idle manger
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

        self.idle_manager = IdleAnimationManager(self)
        self.fps = 120

    async def queue_animation(
        self,
        expression,
    ):
        """Add an expression to the animation queue.

        Args:
            expression (Expression): Expression object to be displayed.
                Must implement render() method.

        The expression is added to an async queue and will be displayed when
        previous expressions complete. No return value or direct animation control.
        """

        await self.expression_queue.put(
            expression,
        )

    async def handle_queue(self):
        """Process queued expressions and manage transitions.

        Background task that:
        - Monitors the expression queue for new expressions
        - Emits completion event for current expression
        - Sets up transition parameters for next expression
        - Emits started event for new expression
        - Updates animation state and timing

        Runs continuously while self.running is True.
        No parameters or return values.
        """

        while self.running:
            if not self.is_transitioning and not self.expression_queue.empty():
                next_expr = await self.expression_queue.get()

                if next_expr:
                    # First emit completion for current expression
                    self.emit("expression_completed", self.current_expression)

                    # Set up transition
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

                    # Now emit started for new expression
                    self.emit("expression_started", next_expr)

            await asyncio.sleep(0.01)

    async def run(self):
        """Main animation loop for expression rendering.

        Manages:
        - Background tasks for queue handling and idle animations
        - Expression transitions and interpolation
        - Frame rendering and timing
        - Neutral expression queueing for non-sticky expressions
        - Pygame event handling and window management

        Runs continuously while self.running is True.
        No parameters or return values.
        """

        asyncio.create_task(self.handle_queue())
        asyncio.create_task(self.idle_manager.run_idle_loop())

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

                # Only queue neutral if non-sticky expression completes
                if (
                    elapsed_time > self.animation_duration
                    and not self.current_expression.sticky
                    and self.expression_queue.empty()
                ):
                    await self.queue_animation(
                        Neutral(
                            duration=1.0,
                            transition_duration=0.2,
                            interpolation="linear",
                            sticky=True,
                        )
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

            # Render the eyes
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
