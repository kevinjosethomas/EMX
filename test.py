# test_all_expressions_with_pause.py
import pygame
import time
import inspect

from engine import expressions
from engine.face_manager import FaceManager
from engine.procedural_face import ProceduralFace


def main():
    pygame.init()
    screen_width, screen_height = 1024, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("All Expressions Test with Pause")

    clock = pygame.time.Clock()

    # 1) Gather all expression subclasses from expressions.py
    expression_classes = []
    for name, obj in inspect.getmembers(expressions):
        if (
            inspect.isclass(obj)
            and issubclass(obj, ProceduralFace)
            and obj is not ProceduralFace
        ):
            expression_classes.append(obj)

    # 2) Build a queue of (ExpressionInstance, interpolation_duration)
    expression_queue = []
    for cls in expression_classes:
        expression_queue.append((cls(), 0.5))  # 1.5s to interpolate

    # 3) Create FaceManager
    fm = FaceManager()

    # Index to track which expression is currently active
    expr_index = 0

    # If we have at least one expression, start with it
    if expression_queue:
        first_expr, first_duration = expression_queue[expr_index]
        fm.set_next_expression(first_expr, duration=first_duration)

    # We'll track the time to compute 'elapsed_time' each frame
    previous_time = time.time()
    running = True

    # Font for text rendering
    font = pygame.font.SysFont(None, 36)

    # We'll also track a "pause_timer" to implement a pause
    # after the expression finishes.
    pause_timer = 0.0  # in milliseconds
    PAUSE_AFTER_EXPRESSION = 1000.0  # 1 second of pause

    while running:
        # -- Event Handling --
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # -- Compute elapsed time in ms --
        current_time = time.time()
        elapsed_time = (current_time - previous_time) * 1000.0
        previous_time = current_time

        # By default, get the next face from FaceManager
        face = fm.get_next_frame(elapsed_time)

        # If the interpolation finished, FaceManager sets next_expression=None
        # We then pause for a bit, then move to the next expression.
        if fm.next_expression is None:
            # Are we already in a pause?
            if pause_timer <= 0:
                # Start the pause
                pause_timer = PAUSE_AFTER_EXPRESSION
            else:
                # If we are in the middle of a pause:
                pause_timer -= elapsed_time
                if pause_timer <= 0:
                    # Done pausing; move to the next expression
                    expr_index += 1
                    if expr_index >= len(expression_queue):
                        expr_index = 0  # loop back to start
                    new_expr, new_dur = expression_queue[expr_index]
                    fm.set_next_expression(new_expr, duration=new_dur)
                # Meanwhile, re-use the last expression's final face
                # so it stays on screen during the pause.
                face = fm.current_expression

        # -- Clear the screen to black --
        screen.fill((0, 0, 0))

        # -- Render the face --
        if face:
            face.render(screen)

        # -- Get the expression name from the queue, not the face object --
        current_expr = expression_queue[expr_index][0]
        expr_name = current_expr.__class__.__name__
        text_surface = font.render(expr_name, True, (255, 255, 255))
        screen.blit(text_surface, (20, 20))

        # -- Flip (update) the display --
        pygame.display.flip()

        # -- Cap to ~60 FPS --
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
