import numpy as np
from .interpolation import ease_in_out_interpolation, linear_interpolation


import numpy as np


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
                [0.184, 0.14],
                [0.196, 0.133],
                [0.213, 0.133],
                [0.396, 0.128],
                [0.41, 0.137],
                [0.42, 0.163],
                [0.442, 0.793],
                [0.428, 0.808],
                [0.409, 0.825],
                [0.186, 0.83],
                [0.162, 0.83],
                [0.137, 0.803],
                [0.621, 0.162],
                [0.637, 0.135],
                [0.661, 0.123],
                [0.83, 0.1],
                [0.864, 0.115],
                [0.89, 0.133],
                [0.915, 0.797],
                [0.907, 0.818],
                [0.896, 0.855],
                [0.63, 0.847],
                [0.614, 0.845],
                [0.595, 0.825],
            ]
        ], {}


class Happy(BaseExpression):
    def define_keyframes(self):
        return [
            [
                [0.132, 0.152],
                [0.159, 0.195],
                [0.175, 0.218],
                [0.362, 0.235],
                [0.388, 0.203],
                [0.412, 0.162],
                [0.442, 0.647],
                [0.446, 0.695],
                [0.421, 0.698],
                [0.149, 0.707],
                [0.13, 0.707],
                [0.128, 0.663],
                [0.611, 0.135],
                [0.634, 0.172],
                [0.653, 0.207],
                [0.845, 0.215],
                [0.858, 0.172],
                [0.887, 0.117],
                [0.896, 0.663],
                [0.9, 0.707],
                [0.868, 0.712],
                [0.629, 0.715],
                [0.597, 0.715],
                [0.594, 0.642],
            ]
        ], {}


class Sad(BaseExpression):
    def define_keyframes(self):
        return [
            [
                [0.102, 0.213],
                [0.102, 0.143],
                [0.171, 0.143],
                [0.388, 0.147],
                [0.44, 0.147],
                [0.441, 0.228],
                [0.459, 0.758],
                [0.459, 0.887],
                [0.386, 0.767],
                [0.187, 0.757],
                [0.115, 0.862],
                [0.104, 0.725],
                [0.583, 0.242],
                [0.582, 0.133],
                [0.656, 0.133],
                [0.851, 0.133],
                [0.914, 0.152],
                [0.914, 0.252],
                [0.906, 0.74],
                [0.906, 0.917],
                [0.829, 0.775],
                [0.639, 0.773],
                [0.6, 0.878],
                [0.584, 0.723],
            ]
        ], {}


class Blink(BaseExpression):
    def define_keyframes(self):
        return [
            [
                [0.133, 0.395],
                [0.154, 0.395],
                [0.184, 0.395],
                [0.387, 0.397],
                [0.424, 0.397],
                [0.459, 0.397],
                [0.472, 0.508],
                [0.429, 0.508],
                [0.385, 0.508],
                [0.194, 0.52],
                [0.16, 0.52],
                [0.119, 0.52],
                [0.614, 0.38],
                [0.656, 0.38],
                [0.703, 0.38],
                [0.878, 0.372],
                [0.902, 0.372],
                [0.947, 0.372],
                [0.963, 0.527],
                [0.911, 0.527],
                [0.877, 0.527],
                [0.726, 0.518],
                [0.654, 0.518],
                [0.627, 0.518],
            ]
        ], {}
