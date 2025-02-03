import numpy as np


def linear_interpolation(start, end, t):
    """Linear interpolation between two points.

    Args:
        start (list): Starting point coordinates
        end (list): Ending point coordinates
        t (float): Interpolation parameter between 0 and 1

    Returns:
        numpy.ndarray: Interpolated point coordinates
    """

    return (1 - t) * np.array(start) + t * np.array(end)


def ease_in_out_interpolation(start, end, t):
    """Smooth easing function with acceleration and deceleration.

    Args:
        start (list): Starting point coordinates
        end (list): Ending point coordinates
        t (float): Interpolation parameter between 0 and 1

    Returns:
        numpy.ndarray: Interpolated point coordinates
    """

    t = t * t * (3 - 2 * t)
    return (1 - t) * np.array(start) + t * np.array(end)


def ease_in_interpolation(start, end, t):
    """Ease in interpolation with gradual acceleration.

    Args:
        start (list): Starting point coordinates
        end (list): Ending point coordinates
        t (float): Interpolation parameter between 0 and 1

    Returns:
        numpy.ndarray: Interpolated point coordinates
    """

    t = t * t
    return (1 - t) * np.array(start) + t * np.array(end)


def ease_out_interpolation(start, end, t):
    """Ease out interpolation with gradual deceleration.

    Args:
        start (list): Starting point coordinates
        end (list): Ending point coordinates
        t (float): Interpolation parameter between 0 and 1

    Returns:
        numpy.ndarray: Interpolated point coordinates
    """

    t = 1 - (1 - t) * (1 - t)
    return (1 - t) * np.array(start) + t * np.array(end)


def cubic_bezier_interpolation(start, end, t, p1, p2):
    """Cubic Bezier interpolation using control points.

    Args:
        start (list): Starting point coordinates
        end (list): Ending point coordinates
        t (float): Interpolation parameter between 0 and 1
        p1 (list): First control point coordinates
        p2 (list): Second control point coordinates

    Returns:
        numpy.ndarray: Interpolated point coordinates
    """

    u = 1 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t

    p = uuu * np.array(start)
    p += 3 * uu * t * np.array(p1)
    p += 3 * u * tt * np.array(p2)


INTERPOLATION = {
    "linear": linear_interpolation,
    "ease_in_out": ease_in_out_interpolation,
    "ease_in": ease_in_interpolation,
    "ease_out": ease_out_interpolation,
    "cubic_bezier": cubic_bezier_interpolation,
}
