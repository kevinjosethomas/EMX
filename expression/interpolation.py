import numpy as np


def linear_interpolation(start, end, t):
    """Linear interpolation between two points."""
    return (1 - t) * np.array(start) + t * np.array(end)


def ease_in_out_interpolation(start, end, t):
    """Smooth easing function."""
    t = t * t * (3 - 2 * t)
    return (1 - t) * np.array(start) + t * np.array(end)


def ease_in_interpolation(start, end, t):
    """Ease in interpolation."""
    t = t * t
    return (1 - t) * np.array(start) + t * np.array(end)


def ease_out_interpolation(start, end, t):
    """Ease out interpolation."""
    t = 1 - (1 - t) * (1 - t)
    return (1 - t) * np.array(start) + t * np.array(end)


def cubic_bezier_interpolation(start, end, t, p1, p2):
    """Cubic Bezier interpolation."""
    u = 1 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t

    p = uuu * np.array(start)  # (1-t)^3 * start
    p += 3 * uu * t * np.array(p1)  # 3(1-t)^2 * t * p1
    p += 3 * u * tt * np.array(p2)  # 3(1-t) * t^2 * p2
    p += ttt * np.array(end)  # t^3 * end

    return p


INTERPOLATION = {
    "linear": linear_interpolation,
    "ease_in_out": ease_in_out_interpolation,
    "ease_in": ease_in_interpolation,
    "ease_out": ease_out_interpolation,
    "cubic_bezier": cubic_bezier_interpolation,
}
