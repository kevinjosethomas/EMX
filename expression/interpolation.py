import numpy as np


def linear_interpolation(start, end, t):
    """Linear interpolation between two points."""
    return (1 - t) * np.array(start) + t * np.array(end)


def ease_in_out_interpolation(start, end, t):
    """Smooth easing function."""
    t = t * t * (3 - 2 * t)
    return (1 - t) * np.array(start) + t * np.array(end)
