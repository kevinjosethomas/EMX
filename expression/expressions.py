import numpy as np
from .interpolation import ease_in_out_interpolation, linear_interpolation


class BaseExpression:
    """
    Base class for expressions.
    - Supports single-frame and multi-frame animations.
    - Uses normalized coordinates for screen independence.
    - Allows per-keyframe interpolation control.
    """

    def __init__(self, duration=1.0, interpolation="linear"):
        self.duration = duration
        self.keyframes, self.interpolation_methods = self.define_keyframes()
        self.current_frame = 0
        self.default_interpolation = (
            interpolation  # Default interpolation style
        )

    def define_keyframes(self):
        """Subclasses must override this to define keyframes and interpolation methods."""
        raise NotImplementedError()

    def get_interpolated_vertices(
        self, t, interpolation_func, screen_width, screen_height
    ):
        """Interpolates smoothly between animation keyframes based on time `t`."""
        num_frames = len(self.keyframes)

        if num_frames == 1:
            # Single-frame expressions (no interpolation needed)
            return self.scale_vertices(
                self.keyframes[0], screen_width, screen_height
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
            interpolated_vertices, screen_width, screen_height
        )

    def scale_vertices(self, vertices, screen_width, screen_height):
        """Scales normalized coordinates (0-1) to the actual screen size."""
        return [(x * screen_width, y * screen_height) for x, y in vertices]

    def render(self, t, interpolation_func, screen_width, screen_height):
        """Renders the interpolated expression based on animation progress `t`."""
        return self.get_interpolated_vertices(
            t, interpolation_func, screen_width, screen_height
        )


class Neutral(BaseExpression):
    def define_keyframes(self):
        return [
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
        ], {}


class Happy(BaseExpression):
    def define_keyframes(self):
        return [
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
        ], {}


class Sad(BaseExpression):
    def define_keyframes(self):
        return [
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
        ], {}


class Blink(BaseExpression):
    def define_keyframes(self):
        return [
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
        ], {}
