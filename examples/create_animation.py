import pygame
import json
import pyperclip

# Constants
WIDTH, HEIGHT = 1024, 600
BG_COLOR = (30, 30, 30)
POINT_COLOR_LEFT, POINT_COLOR_RIGHT = (255, 100, 100), (100, 100, 255)
EYE_FILL_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)
FPS, POINT_RADIUS, MAX_POINTS = 60, 6, 24  # 12 points per eye

# Expected point order (consistent with `engine.py`)
POINT_LABELS = [
    "Outer Top-Left",
    "Outer Top",
    "Outer Top-Right",
    "Outer Right",
    "Outer Bottom-Right",
    "Outer Bottom",
    "Outer Bottom-Left",
    "Outer Left",
    "Inner Left",
    "Inner Top",
    "Inner Bottom",
    "Inner Right",
]


class AnimationCreator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Animation Creator")
        self.clock, self.points, self.frames = pygame.time.Clock(), [], []
        self.selected_point, self.running = None, True

    def draw_points(self):
        """Draws filled eyes and numbered landmarks."""
        if len(self.points) == MAX_POINTS:
            pygame.draw.polygon(
                self.screen, EYE_FILL_COLOR, self.points[:12]
            )  # Left eye
            pygame.draw.polygon(
                self.screen, EYE_FILL_COLOR, self.points[12:]
            )  # Right eye

        font = pygame.font.Font(None, 24)
        for i, (x, y) in enumerate(self.points):
            color = POINT_COLOR_LEFT if i < 12 else POINT_COLOR_RIGHT
            pygame.draw.circle(self.screen, color, (x, y), POINT_RADIUS)
            label = f"{i+1}: {POINT_LABELS[i % 12]}"  # Show correct placement label
            self.screen.blit(
                font.render(label, True, TEXT_COLOR), (x + 10, y - 10)
            )

    def handle_mouse_input(self, pos, action):
        """Handles mouse clicks and dragging."""
        if action == "click":
            for i, (px, py) in enumerate(self.points):
                if (px - pos[0]) ** 2 + (
                    py - pos[1]
                ) ** 2 < POINT_RADIUS**2 * 4:
                    self.selected_point = i
                    return
            if len(self.points) < MAX_POINTS:
                self.points.append(pos)
        elif action == "drag" and self.selected_point is not None:
            self.points[self.selected_point] = pos

    def reset_points(self):
        self.points, self.selected_point = [], None

    def save_frame(self):
        """Saves the current frame into the animation sequence."""
        if len(self.points) == MAX_POINTS:
            self.frames.append(self.points.copy())
            print(f"Frame {len(self.frames)} saved!")

    def export_animation(self):
        """Copies animation code to clipboard and exits."""
        if not self.frames:
            print("No frames to export!")
            return

        normalized_frames = [
            [[round(x / WIDTH, 3), round(y / HEIGHT, 3)] for x, y in frame]
            for frame in self.frames
        ]
        animation_code = f"class CustomExpression(BaseExpression):\n    def define_keyframes(self):\n        return {json.dumps(normalized_frames, indent=4)}, {{}}"

        pyperclip.copy(animation_code)
        print(
            "\n✅ Animation copied to clipboard! Paste it into `expression.py` 🎉"
        )
        pygame.quit()
        exit()

    def run(self):
        """Main event loop."""
        while self.running:
            self.screen.fill(BG_COLOR)
            self.draw_points()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_input(event.pos, "click")
                elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                    self.handle_mouse_input(event.pos, "drag")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_points()
                    elif event.key == pygame.K_SPACE:
                        self.save_frame()
                    elif event.key == pygame.K_RETURN:
                        self.export_animation()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    AnimationCreator().run()
