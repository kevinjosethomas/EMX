import pygame
import time
import inspect

# Adjust these imports to match your project structure:
from engine.face_manager import FaceManager
from engine import expressions  # The module containing all expression classes.
from engine.procedural_face import ProceduralFace


def main():
    pygame.init()
    screen_width, screen_height = 1024, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Procedural Face Test - All Expressions")

    clock = pygame.time.Clock()

    # 1) Dynamically gather all expression classes (subclasses of ProceduralFace)
    expression_classes = []
    for name, obj in inspect.getmembers(expressions):
        # We skip the base ProceduralFace and only grab the actual expressions
        if (
            inspect.isclass(obj)
            and issubclass(obj, ProceduralFace)
            and obj is not ProceduralFace
        ):
            expression_classes.append(obj)

    # 2) Build a queue of (ExpressionInstance, duration) for each expression class
    expression_queue = []
    for cls in expression_classes:
        # For example, 1.5 second duration each
        expression_queue.append((cls(), 1.5))

    # 3) Create a FaceManager
    fm = FaceManager()

    # Index to track which expression in the queue we’re on
    expr_index = 0

    # Initialize with the first expression
    if expression_queue:
        first_expr, first_duration = expression_queue[expr_index]
        fm.set_next_expression(first_expr, duration=first_duration)

    previous_time = time.time()
    running = True

    while running:
        # -- Handle events --
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # -- Compute elapsed time (ms) --
        current_time = time.time()
        elapsed_time = (current_time - previous_time) * 1000.0
        previous_time = current_time

        # -- Get the next interpolated face from FaceManager --
        face = fm.get_next_frame(elapsed_time)

        # If we've finished the current expression, move on to the next
        if fm.next_expression is None and expression_queue:
            expr_index += 1
            if expr_index >= len(expression_queue):
                # Loop back to the start or exit
                expr_index = 0
                # If you prefer to stop after the last:
                # running = False
                # break

            next_expr, next_duration = expression_queue[expr_index]
            fm.set_next_expression(next_expr, duration=next_duration)

        # -- Clear screen and draw the face --
        screen.fill((0, 0, 0))
        face.render(screen)
        pygame.display.flip()

        # -- Limit to 60 FPS --
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
