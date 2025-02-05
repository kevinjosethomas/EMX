import numpy as np
from .interpolation import INTERPOLATION


class BaseExpression:
    """Base class for facial expressions and animations.

    Provides core functionality for rendering facial expressions and animations:
    - Keyframe-based animation system
    - Normalized coordinate system (0-1) for screen independence
    - Configurable interpolation between keyframes
    - Position and scale transformation support
    - Sticky expressions that persist until changed

    Attributes:
        id (str): Unique identifier for the expression
        label (str): Human-readable name of the expression
        keyframes (list): List of vertex arrays defining expression frames
        duration (float): Total animation duration in seconds
        transition_duration (float): Transition time between expressions
        interpolation (str): Name of interpolation method to use
        sticky (bool): Whether expression persists after animation
        position (tuple): (x,y) offset for expression, normalized 0-1
        scale (float): Scale factor for expression size
        current_frame (int): Index of current animation frame
    """

    base_defaults = {
        "id": "base",
        "label": "Base",
        "duration": 1.0,
        "transition_duration": 0.2,
        "interpolation": "linear",
        "sticky": False,
        "position": (0.0, 0.0),
        "scale": 1.0,
    }

    def __init__(self, **kwargs):
        """Initialize a new expression.

        Args:
            id (str): Unique identifier for the expression
            label (str): Human-readable name
            keyframes (list): List of vertex arrays defining frames
            duration (float, optional): Animation duration. Defaults to 1.0.
            transition_duration (float, optional): Transition time. Defaults to 0.2.
            interpolation (callable, optional): Interpolation function. Defaults to linear.
            sticky (bool, optional): Whether expression persists. Defaults to False.
            position (tuple, optional): (x,y) offset. Defaults to (0,0).
            scale (float, optional): Size scaling. Defaults to 1.0.
        """

        settings = self.base_defaults.copy()
        if hasattr(self.__class__, "defaults"):
            settings.update(self.__class__.defaults)

        settings.update(kwargs)

        for key, value in settings.items():
            if key == "interpolation":
                value = INTERPOLATION[value]
            setattr(self, key, value)

        self.keyframes = self.__class__.keyframes

    def define_interpolation_methods(self):
        """Define interpolation functions for keyframe transitions.

        Returns:
            dict: Mapping of keyframe indices to interpolation functions.
                Default is linear interpolation for all transitions.

        Can be overridden by subclasses to customize interpolation per keyframe.
        """

        return {i: self.interpolation for i in range(len(self.keyframes))}

    def get_interpolated_vertices(
        self, t, interpolation_func, screen_width, screen_height
    ):
        """Define interpolation methods for keyframe transitions.

        Returns:
            dict: Mapping of keyframe indices to interpolation functions.
                Default is linear interpolation for all transitions.

        Can be overridden by subclasses to customize interpolation per keyframe.
        """
        num_frames = len(self.keyframes)

        if num_frames == 1:
            # Single-frame expressions (no interpolation needed)
            return self.scale_vertices(
                self.apply_position_offset(self.keyframes[0]),
                screen_width,
                screen_height,
            )

        # Determine which two frames we are interpolating between
        frame_idx = min(int(t * (num_frames - 1)), num_frames - 2)

        v_start = np.array(self.keyframes[frame_idx])
        v_end = np.array(self.keyframes[frame_idx + 1])

        # Get interpolation function for this transition
        interp_func = self.interpolation_methods.get(
            frame_idx, interpolation_func
        )

        # Normalize `t` for local frame transition
        t_local = (t * (num_frames - 1)) % 1

        # Compute interpolated vertices
        interpolated_vertices = interp_func(v_start, v_end, t_local)

        return self.scale_vertices(
            self.apply_position_offset(interpolated_vertices),
            screen_width,
            screen_height,
        )

    def scale_vertices(self, vertices, screen_width, screen_height):
        """Scale vertices while keeping shape within screen bounds.

        Args:
            vertices (list): List of (x,y) vertex coordinates
            screen_width (int): Width of display in pixels
            screen_height (int): Height of display in pixels

        Returns:
            list: Scaled vertices that keep shape within screen
        """
        # First apply scaling to get intended size
        scaled = [
            (x * screen_width * self.scale, y * screen_height * self.scale)
            for x, y in vertices
        ]

        # Find current bounds
        min_x = min(x for x, _ in scaled)
        max_x = max(x for x, _ in scaled)
        min_y = min(y for _, y in scaled)
        max_y = max(y for _, y in scaled)

        # Calculate scale adjustment if needed
        width = max_x - min_x
        height = max_y - min_y

        x_scale = 1.0
        if width > screen_width:
            x_scale = screen_width / width

        y_scale = 1.0
        if height > screen_height:
            y_scale = screen_height / height

        # Use the more constraining scale
        adjust_scale = min(x_scale, y_scale)

        # Apply final scaling
        final_scale = self.scale * adjust_scale
        return [
            (x * screen_width * final_scale, y * screen_height * final_scale)
            for x, y in vertices
        ]

    def apply_position_offset(self, vertices):
        """Apply position offset while keeping entire shape within bounds.

        Args:
            vertices (list): List of (x,y) vertex coordinates

        Returns:
            list: Position-adjusted vertices that keep shape within screen
        """

        offset_x, offset_y = self.position

        # Apply initial offset to get intended positions
        adjusted = [(x + offset_x, y + offset_y) for x, y in vertices]

        # Find the bounding box of the expression
        min_x = min(x for x, _ in adjusted)
        max_x = max(x for x, _ in adjusted)
        min_y = min(y for _, y in adjusted)
        max_y = max(y for _, y in adjusted)

        # Calculate how much we need to shift to keep expression in bounds
        x_shift = 0
        if min_x < 0:
            x_shift = -min_x
        elif max_x > 1:
            x_shift = 1 - max_x

        y_shift = 0
        if min_y < 0:
            y_shift = -min_y
        elif max_y > 1:
            y_shift = 1 - max_y

        # Apply the final adjustment to keep in bounds
        return [(x + x_shift, y + y_shift) for x, y in adjusted]

    def render(self, t, interpolation_func, screen_width, screen_height):
        """Render interpolated expression for current animation frame.

        Args:
            t (float): Animation progress from 0-1
            interpolation_func (callable): Function to interpolate between keyframes
            screen_width (int): Display width in pixels
            screen_height (int): Display height in pixels

        Returns:
            list: Interpolated vertex coordinates for current frame
        """

        return self.get_interpolated_vertices(
            t, interpolation_func, screen_width, screen_height
        )


class Neutral(BaseExpression):

    defaults = {
        "id": "neutral",
        "label": "Neutral",
        "sticky": True,
    }

    keyframes = [
        [
            [0.29, 0.272],
            [0.357, 0.302],
            [0.406, 0.385],
            [0.424, 0.499],
            [0.406, 0.628],
            [0.357, 0.696],
            [0.29, 0.727],
            [0.223, 0.696],
            [0.174, 0.613],
            [0.156, 0.499],
            [0.174, 0.385],
            [0.223, 0.302],
            [0.71, 0.272],
            [0.777, 0.302],
            [0.826, 0.385],
            [0.844, 0.499],
            [0.826, 0.613],
            [0.777, 0.696],
            [0.71, 0.727],
            [0.643, 0.696],
            [0.594, 0.613],
            [0.576, 0.499],
            [0.594, 0.385],
            [0.643, 0.302],
        ]
    ]


class Happy(BaseExpression):

    defaults = {
        "id": "happy",
        "label": "Happy",
    }

    keyframes = [
        [
            [0.29, 0.272],
            [0.357, 0.302],
            [0.406, 0.385],
            [0.424, 0.499],
            [0.406, 0.613],
            [0.357, 0.559],
            [0.29, 0.534],
            [0.223, 0.559],
            [0.174, 0.613],
            [0.156, 0.499],
            [0.174, 0.385],
            [0.223, 0.302],
            [0.71, 0.272],
            [0.777, 0.302],
            [0.826, 0.385],
            [0.844, 0.499],
            [0.826, 0.613],
            [0.777, 0.559],
            [0.71, 0.534],
            [0.643, 0.559],
            [0.594, 0.613],
            [0.576, 0.499],
            [0.594, 0.385],
            [0.643, 0.302],
        ]
    ]


class Love(BaseExpression):

    defaults = {
        "id": "love",
        "label": "Love",
    }

    keyframes = [
        [
            [0.283, 0.353],
            [0.35, 0.315],
            [0.389, 0.385],
            [0.404, 0.499],
            [0.37, 0.605],
            [0.332, 0.682],
            [0.283, 0.727],
            [0.237, 0.682],
            [0.194, 0.605],
            [0.16, 0.499],
            [0.175, 0.385],
            [0.216, 0.315],
            [0.718, 0.353],
            [0.785, 0.315],
            [0.824, 0.385],
            [0.839, 0.499],
            [0.805, 0.605],
            [0.767, 0.682],
            [0.718, 0.727],
            [0.672, 0.682],
            [0.629, 0.605],
            [0.595, 0.499],
            [0.61, 0.385],
            [0.651, 0.315],
        ]
    ]


class Scared(BaseExpression):

    defaults = {
        "id": "scared",
        "label": "Scared",
    }

    keyframes = [
        [
            [0.29, 0.339],
            [0.357, 0.279],
            [0.406, 0.385],
            [0.424, 0.499],
            [0.406, 0.613],
            [0.357, 0.588],
            [0.29, 0.588],
            [0.223, 0.593],
            [0.174, 0.613],
            [0.156, 0.499],
            [0.174, 0.385],
            [0.223, 0.368],
            [0.777, 0.368],
            [0.826, 0.385],
            [0.844, 0.499],
            [0.826, 0.613],
            [0.777, 0.593],
            [0.71, 0.588],
            [0.643, 0.588],
            [0.594, 0.613],
            [0.576, 0.499],
            [0.594, 0.385],
            [0.643, 0.279],
            [0.71, 0.339],
        ]
    ]


class Sad(BaseExpression):

    defaults = {
        "id": "sad",
        "label": "Sad",
    }

    keyframes = [
        [
            [0.301, 0.398],
            [0.357, 0.334],
            [0.406, 0.385],
            [0.424, 0.499],
            [0.406, 0.613],
            [0.357, 0.696],
            [0.29, 0.727],
            [0.223, 0.696],
            [0.174, 0.613],
            [0.156, 0.499],
            [0.174, 0.458],
            [0.234, 0.451],
            [0.766, 0.451],
            [0.826, 0.458],
            [0.844, 0.499],
            [0.826, 0.613],
            [0.777, 0.696],
            [0.71, 0.727],
            [0.643, 0.696],
            [0.594, 0.613],
            [0.576, 0.499],
            [0.595, 0.385],
            [0.643, 0.334],
            [0.699, 0.398],
        ]
    ]


class Angry(BaseExpression):

    defaults = {
        "id": "angry",
        "label": "Angry",
    }

    keyframes = [
        [
            [0.357, 0.429],
            [0.406, 0.482],
            [0.424, 0.545],
            [0.406, 0.613],
            [0.357, 0.696],
            [0.29, 0.727],
            [0.223, 0.696],
            [0.174, 0.613],
            [0.156, 0.499],
            [0.174, 0.385],
            [0.223, 0.337],
            [0.29, 0.39],
            [0.71, 0.39],
            [0.777, 0.337],
            [0.826, 0.385],
            [0.844, 0.499],
            [0.826, 0.613],
            [0.777, 0.696],
            [0.71, 0.727],
            [0.643, 0.696],
            [0.594, 0.613],
            [0.576, 0.545],
            [0.594, 0.482],
            [0.643, 0.429],
        ]
    ]


class Blink(BaseExpression):
    defaults = {
        "id": "blink",
        "label": "Blink",
        "duration": 0.1,
        "transition_duration": 0.05,
        "interpolation": "ease_in_out",
    }

    keyframes = [
        [
            [0.225, 0.467],
            [0.225, 0.467],
            [0.225, 0.467],
            [0.476, 0.467],
            [0.476, 0.467],
            [0.476, 0.467],
            [0.476, 0.535],
            [0.476, 0.535],
            [0.476, 0.535],
            [0.227, 0.535],
            [0.227, 0.535],
            [0.227, 0.535],
            [0.524, 0.467],
            [0.524, 0.467],
            [0.524, 0.467],
            [0.773, 0.467],
            [0.773, 0.467],
            [0.773, 0.467],
            [0.773, 0.535],
            [0.773, 0.535],
            [0.773, 0.535],
            [0.524, 0.535],
            [0.524, 0.535],
            [0.524, 0.535],
        ]
    ]


class Squint(BaseExpression):
    defaults = {
        "id": "squint",
        "label": "Squint",
        "duration": 2,
        "transition_duration": 0.2,
        "interpolation": "ease_in_out",
    }

    keyframes = [
        [
            [0.227, 0.318],
            [0.241, 0.302],
            [0.251, 0.29],
            [0.451, 0.29],
            [0.461, 0.302],
            [0.476, 0.318],
            [0.476, 0.682],
            [0.464, 0.695],
            [0.451, 0.71],
            [0.251, 0.71],
            [0.237, 0.695],
            [0.227, 0.682],
            [0.524, 0.318],
            [0.539, 0.302],
            [0.549, 0.29],
            [0.749, 0.29],
            [0.759, 0.302],
            [0.773, 0.318],
            [0.773, 0.682],
            [0.762, 0.695],
            [0.749, 0.71],
            [0.549, 0.71],
            [0.535, 0.695],
            [0.524, 0.682],
        ]
    ]
