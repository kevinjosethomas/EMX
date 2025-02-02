import asyncio
import pygame
from engine import expressions as exp
from engine.face_manager import FaceManager

# Mapping of keys to expression classes
KEY_TO_EXPRESSION = {
    pygame.K_1: "Neutral",
    pygame.K_2: "Anger",
    pygame.K_3: "Sadness",
    pygame.K_4: "Happiness",
    pygame.K_5: "Surprise",
    pygame.K_6: "Disgust",
    pygame.K_7: "Fear",
    # Add more mappings as needed
}


async def main():
    # Gather all expression subclasses
    expression_classes = exp.get_all_expression_classes()
    expression_map = {cls.__name__: cls for cls in expression_classes}

    # Create FaceManager
    fm = FaceManager(debug=True)

    # Start the rendering loop
    await fm.start_rendering()

    # Function to handle keyboard input and queue expressions
    async def handle_input():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    expression_name = KEY_TO_EXPRESSION.get(event.key)
                    if expression_name:
                        expression_cls = expression_map.get(expression_name)
                        if expression_cls:
                            await fm.cue_expression(expression_cls(), 0.5)
            await asyncio.sleep(0)  # Yield control to the event loop

    # Start handling input in a separate task
    input_task = asyncio.create_task(handle_input())

    # Wait for the input task to complete (it won't in this example)
    await input_task


if __name__ == "__main__":
    asyncio.run(main())
