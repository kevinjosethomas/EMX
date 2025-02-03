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
        """Scale normalized vertices to screen coordinates with scaling factor.

        Args:
            vertices (list): List of (x,y) vertex coordinates in 0-1 range
            screen_width (int): Width of display in pixels
            screen_height (int): Height of display in pixels

        Returns:
            list: Scaled vertex coordinates in screen space
        """

        return [
            (x * screen_width * self.scale, y * screen_height * self.scale)
            for x, y in vertices
        ]

    def apply_position_offset(self, vertices):
        """Apply position offset to vertices while keeping within bounds.

        Args:
            vertices (list): List of (x,y) vertex coordinates

        Returns:
            list: Adjusted vertex coordinates clamped to screen bounds (0-1)

        Shifts vertices by position offset and ensures coordinates stay within
        valid screen space by clamping to 0-1 range.
        """

        offset_x, offset_y = self.position
        adjusted_vertices = []

        for x, y in vertices:
            new_x = x + offset_x
            new_y = y + offset_y

            # Clamp the new coordinates to stay within the screen boundaries (0 to 1)
            new_x = max(0.0, min(new_x, 1.0))
            new_y = max(0.0, min(new_y, 1.0))

            adjusted_vertices.append((new_x, new_y))

        return adjusted_vertices

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
        "duration": 1.0,
        "transition_duration": 0.2,
        "interpolation": "linear",
        "sticky": True,
    }

    keyframes = [
        [
            [0.227, 0.227],
            [0.241, 0.203],
            [0.251, 0.185],
            [0.451, 0.185],
            [0.461, 0.203],
            [0.476, 0.227],
            [0.476, 0.775],
            [0.464, 0.794],
            [0.451, 0.817],
            [0.251, 0.817],
            [0.237, 0.794],
            [0.227, 0.775],
            [0.524, 0.227],
            [0.539, 0.203],
            [0.549, 0.185],
            [0.749, 0.185],
            [0.759, 0.203],
            [0.773, 0.227],
            [0.773, 0.775],
            [0.762, 0.794],
            [0.749, 0.817],
            [0.549, 0.817],
            [0.535, 0.794],
            [0.524, 0.775],
        ]
    ]


class Happy(BaseExpression):
    defaults = {
        "id": "happy",
        "label": "Happy",
        "duration": 1.0,
        "transition_duration": 0.2,
        "interpolation": "linear",
    }

    keyframes = [
        [
            [0.227, 0.227],
            [0.241, 0.203],
            [0.251, 0.185],
            [0.451, 0.185],
            [0.461, 0.203],
            [0.476, 0.227],
            [0.476, 0.775],
            [0.464, 0.794],
            [0.451, 0.817],
            [0.251, 0.817],
            [0.237, 0.794],
            [0.227, 0.775],
            [0.524, 0.227],
            [0.539, 0.203],
            [0.549, 0.185],
            [0.749, 0.185],
            [0.759, 0.203],
            [0.773, 0.227],
            [0.773, 0.775],
            [0.762, 0.794],
            [0.749, 0.817],
            [0.549, 0.817],
            [0.535, 0.794],
            [0.524, 0.775],
        ]
    ]


class Sad(BaseExpression):
    defaults = {
        "id": "sad",
        "label": "Sad",
        "duration": 1.0,
        "transition_duration": 0.2,
        "interpolation": "linear",
    }

    keyframes = [
        [
            [0.227, 0.227],
            [0.241, 0.203],
            [0.251, 0.185],
            [0.451, 0.185],
            [0.461, 0.203],
            [0.476, 0.227],
            [0.476, 0.775],
            [0.464, 0.794],
            [0.451, 0.817],
            [0.251, 0.817],
            [0.237, 0.794],
            [0.227, 0.775],
            [0.524, 0.227],
            [0.539, 0.203],
            [0.549, 0.185],
            [0.749, 0.185],
            [0.759, 0.203],
            [0.773, 0.227],
            [0.773, 0.775],
            [0.762, 0.794],
            [0.749, 0.817],
            [0.549, 0.817],
            [0.535, 0.794],
            [0.524, 0.775],
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
