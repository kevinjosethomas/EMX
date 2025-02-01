# procedural_face.py (Pygame version)

import pygame
import numpy as np
import math
from functools import lru_cache
from typing import Optional, List, Generator

########################
# Constants and Config #
########################

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

DEFAULT_EYE_WIDTH = 280
DEFAULT_EYE_HEIGHT = 450

X_FACTOR = 0.55
Y_FACTOR = 0.25

#######################################
# Helper Methods for Filled Arcs/Chords
#######################################


def draw_filled_arc(
    surface: pygame.Surface,
    color: tuple,
    rect: pygame.Rect,
    start_angle_deg: float,
    end_angle_deg: float,
    segments: int = 60,
):
    """Draw a filled arc (equivalent to a pieslice) on the surface."""
    # We'll approximate the arc by constructing a polygon.
    x, y, w, h = rect
    cx = x + w / 2.0
    cy = y + h / 2.0

    points = [(cx, cy)]
    step = (end_angle_deg - start_angle_deg) / segments
    for i in range(segments + 1):
        ang_deg = start_angle_deg + i * step
        rad = math.radians(ang_deg)
        px = cx + (w / 2.0) * math.cos(rad)
        py = cy + (h / 2.0) * math.sin(rad)
        points.append((px, py))

    pygame.draw.polygon(surface, color, points)


def draw_filled_chord(
    surface: pygame.Surface,
    color: tuple,
    rect: pygame.Rect,
    start_angle_deg: float,
    end_angle_deg: float,
    segments: int = 60,
):
    """Draw a filled chord from start_angle to end_angle."""
    x, y, w, h = rect
    cx = x + w / 2.0
    cy = y + h / 2.0

    points = []
    step = (end_angle_deg - start_angle_deg) / segments
    for i in range(segments + 1):
        ang_deg = start_angle_deg + i * step
        rad = math.radians(ang_deg)
        px = cx + (w / 2.0) * math.cos(rad)
        py = cy + (h / 2.0) * math.sin(rad)
        points.append((px, py))

    # Connect endpoints (start & end of the arc)
    start_rad = math.radians(start_angle_deg)
    start_px = cx + (w / 2.0) * math.cos(start_rad)
    start_py = cy + (w / 2.0) * math.sin(start_rad)

    end_rad = math.radians(end_angle_deg)
    end_px = cx + (w / 2.0) * math.cos(end_rad)
    end_py = cy + (w / 2.0) * math.sin(end_rad)

    points.append((start_px, start_py))
    pygame.draw.polygon(surface, color, points)


###################################
# Procedural Face Base Components #
###################################


class ProceduralBase:
    __slots__ = (
        "params",
        "offset",
        "width",
        "height",
        "eye_width",
        "eye_height",
        "half_eye_width",
        "half_eye_height",
        "scale_factor_lid_height",
        "scale_factor_lid_bend",
    )

    def __init__(
        self, params: List[float], offset: int, width: int, height: int
    ):
        self.params = params
        self.offset = offset
        self.width = width
        self.height = height
        self.eye_width = width * (DEFAULT_EYE_WIDTH / DEFAULT_WIDTH)
        self.eye_height = height * (DEFAULT_EYE_HEIGHT / DEFAULT_HEIGHT)
        self.half_eye_width = self.eye_width / 2
        self.half_eye_height = self.eye_height / 2
        self.scale_factor_lid_height = 1.2 * self.eye_width
        self.scale_factor_lid_bend = 1.2 * self.half_eye_width


class ProceduralLid(ProceduralBase):
    __slots__ = (
        "y_offset",
        "angle_offset",
    )

    def __init__(
        self,
        params: List[float],
        offset: int,
        y_offset: float,
        angle_offset: float,
        width: int,
        height: int,
    ):
        super().__init__(params, offset, width, height)
        self.y_offset = float(y_offset)
        self.angle_offset = float(angle_offset)

    @property
    def y(self) -> float:
        return self.params[self.offset + 0]

    @y.setter
    def y(self, value: float) -> None:
        self.params[self.offset + 0] = value

    @property
    def angle(self) -> float:
        return self.params[self.offset + 1]

    @angle.setter
    def angle(self, value: float) -> None:
        self.params[self.offset + 1] = value

    @property
    def bend(self) -> float:
        return self.params[self.offset + 2]

    @bend.setter
    def bend(self, value: float) -> None:
        self.params[self.offset + 2] = value

    def render(self, surface: pygame.Surface) -> None:
        # 1) Create a large surface to draw the lid
        lid_surf = pygame.Surface(
            (self.width * 2, self.height * 2), pygame.SRCALPHA
        )

        # 2) Draw rectangle for the lid
        lid_height = int(self.eye_height * self.y)
        x1 = self.width - self.scale_factor_lid_height
        y1 = self.height - 1 - self.half_eye_height
        rect_width = self.scale_factor_lid_height * 2
        rect_height = lid_height

        pygame.draw.rect(
            lid_surf,
            (0, 0, 0, 255),
            (x1, y1, rect_width, rect_height),
        )

        # 3) Draw chord for the bend
        bend_height = int(self.eye_height * (1.0 - self.y) * self.bend)
        x3 = self.width - self.scale_factor_lid_bend
        y3 = self.height - 1 + lid_height - bend_height
        chord_width = self.scale_factor_lid_bend * 2
        chord_height = bend_height * 2
        chord_rect = pygame.Rect(x3, y3, chord_width, chord_height)
        draw_filled_chord(lid_surf, (0, 0, 0, 255), chord_rect, 0, 180)

        # 4) Rotate the lid surface
        angle_total = self.angle + self.angle_offset
        rotated_lid = pygame.transform.rotate(lid_surf, angle_total)

        # 5) Blit onto parent surface
        lid_x = (surface.get_width() - rotated_lid.get_width()) // 2
        lid_y = (surface.get_height() - rotated_lid.get_height()) // 2 + int(
            self.y_offset
        )
        surface.blit(rotated_lid, (lid_x, lid_y))


class ProceduralEye(ProceduralBase):
    __slots__ = (
        "corner_radius",
        "x_offset",
        "lids",
    )

    def __init__(
        self,
        params: List[float],
        offset: int,
        x_offset: float = 0.0,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ):
        super().__init__(params, offset, width, height)
        self.x_offset = float(x_offset)
        self.corner_radius = self.width / 20 + self.height / 10
        self.lids = (
            ProceduralLid(
                params,
                offset + 13,
                -self.half_eye_height,
                0.0,
                self.width,
                self.height,
            ),
            ProceduralLid(
                params,
                offset + 13 + 3,
                self.half_eye_height + 1,
                180.0,
                self.width,
                self.height,
            ),
        )

    # Eye-related parameters
    @property
    def center_x(self) -> float:
        return self.params[self.offset + 0]

    @center_x.setter
    def center_x(self, value: float) -> None:
        self.params[self.offset + 0] = value

    @property
    def center_y(self) -> float:
        return self.params[self.offset + 1]

    @center_y.setter
    def center_y(self, value: float) -> None:
        self.params[self.offset + 1] = value

    @property
    def scale_x(self) -> float:
        return self.params[self.offset + 2]

    @scale_x.setter
    def scale_x(self, value: float) -> None:
        self.params[self.offset + 2] = value

    @property
    def scale_y(self) -> float:
        return self.params[self.offset + 3]

    @scale_y.setter
    def scale_y(self, value: float) -> None:
        self.params[self.offset + 3] = value

    @property
    def angle(self) -> float:
        return self.params[self.offset + 4]

    @angle.setter
    def angle(self, value: float) -> None:
        self.params[self.offset + 4] = value

    # Arc radius parameters
    @property
    def lower_inner_radius_x(self) -> float:
        return self.params[self.offset + 5]

    @lower_inner_radius_x.setter
    def lower_inner_radius_x(self, value: float) -> None:
        self.params[self.offset + 5] = value

    @property
    def lower_inner_radius_y(self) -> float:
        return self.params[self.offset + 6]

    @lower_inner_radius_y.setter
    def lower_inner_radius_y(self, value: float) -> None:
        self.params[self.offset + 6] = value

    @property
    def lower_outer_radius_x(self) -> float:
        return self.params[self.offset + 7]

    @lower_outer_radius_x.setter
    def lower_outer_radius_x(self, value: float) -> None:
        self.params[self.offset + 7] = value

    @property
    def lower_outer_radius_y(self) -> float:
        return self.params[self.offset + 8]

    @lower_outer_radius_y.setter
    def lower_outer_radius_y(self, value: float) -> None:
        self.params[self.offset + 8] = value

    @property
    def upper_inner_radius_x(self) -> float:
        return self.params[self.offset + 9]

    @upper_inner_radius_x.setter
    def upper_inner_radius_x(self, value: float) -> None:
        self.params[self.offset + 9] = value

    @property
    def upper_inner_radius_y(self) -> float:
        return self.params[self.offset + 10]

    @upper_inner_radius_y.setter
    def upper_inner_radius_y(self, value: float) -> None:
        self.params[self.offset + 10] = value

    @property
    def upper_outer_radius_x(self) -> float:
        return self.params[self.offset + 11]

    @upper_outer_radius_x.setter
    def upper_outer_radius_x(self, value: float) -> None:
        self.params[self.offset + 11] = value

    @property
    def upper_outer_radius_y(self) -> float:
        return self.params[self.offset + 12]

    @upper_outer_radius_y.setter
    def upper_outer_radius_y(self, value: float) -> None:
        self.params[self.offset + 12] = value

    # Sub-rendering helper methods to replicate Pillow's rectangle & pieslice logic
    def _render_inner_rect(
        self, surface: pygame.Surface, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        rect = pygame.Rect(
            x2
            - int(
                self.corner_radius
                * max(self.upper_inner_radius_x, self.lower_inner_radius_x)
            ),
            y1 + int(self.corner_radius * self.upper_inner_radius_y),
            int(
                self.corner_radius
                * max(self.upper_inner_radius_x, self.lower_inner_radius_x)
            ),
            (y2 - int(self.corner_radius * self.lower_inner_radius_y))
            - (y1 + int(self.corner_radius * self.upper_inner_radius_y)),
        )
        pygame.draw.rect(surface, (255, 255, 255, 255), rect)

    def _render_upper_rect(
        self, surface: pygame.Surface, x1: int, y1: int, x2: int
    ) -> None:
        rect = pygame.Rect(
            x1 + int(self.corner_radius * self.upper_outer_radius_x),
            y1,
            (x2 - int(self.corner_radius * self.upper_inner_radius_x))
            - (x1 + int(self.corner_radius * self.upper_outer_radius_x)),
            int(
                self.corner_radius
                * max(self.upper_outer_radius_y, self.upper_inner_radius_y)
            ),
        )
        pygame.draw.rect(surface, (255, 255, 255, 255), rect)

    def _render_outer_rect(
        self, surface: pygame.Surface, x1: int, y1: int, y2: int
    ) -> None:
        rect = pygame.Rect(
            x1,
            y1 + int(self.corner_radius * self.upper_outer_radius_y),
            int(
                self.corner_radius
                * max(self.upper_outer_radius_x, self.lower_outer_radius_x)
            ),
            (y2 - int(self.corner_radius * self.lower_outer_radius_y))
            - (y1 + int(self.corner_radius * self.upper_outer_radius_y)),
        )
        pygame.draw.rect(surface, (255, 255, 255, 255), rect)

    def _render_lower_rect(
        self, surface: pygame.Surface, x1: int, x2: int, y2: int
    ) -> None:
        rect = pygame.Rect(
            x1 + int(self.corner_radius * self.lower_outer_radius_x),
            y2
            - int(
                self.corner_radius
                * max(self.lower_outer_radius_y, self.lower_inner_radius_y)
            ),
            (x2 - int(self.corner_radius * self.lower_inner_radius_x))
            - (x1 + int(self.corner_radius * self.lower_outer_radius_x)),
            int(
                self.corner_radius
                * max(self.lower_outer_radius_y, self.lower_inner_radius_y)
            ),
        )
        pygame.draw.rect(surface, (255, 255, 255, 255), rect)

    def _render_center_rect(
        self, surface: pygame.Surface, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        rx1 = (
            x1
            + int(
                self.corner_radius
                * max(self.upper_outer_radius_x, self.lower_outer_radius_x)
            )
            - 2
        )
        ry1 = (
            y1
            + int(
                self.corner_radius
                * max(self.upper_outer_radius_y, self.upper_inner_radius_y)
            )
            - 1
        )
        rx2 = (
            x2
            - int(
                self.corner_radius
                * max(self.upper_inner_radius_x, self.lower_inner_radius_x)
            )
            + 2
        )
        ry2 = (
            y2
            - int(
                self.corner_radius
                * max(self.lower_outer_radius_y, self.lower_inner_radius_y)
            )
            + 1
        )
        rect = pygame.Rect(rx1, ry1, rx2 - rx1, ry2 - ry1)
        pygame.draw.rect(surface, (255, 255, 255, 255), rect)

    def _render_lower_inner_pie(
        self, surface: pygame.Surface, x2: int, y2: int
    ) -> None:
        w = int(self.corner_radius * self.lower_inner_radius_x) * 2
        h = int(self.corner_radius * self.lower_inner_radius_y) * 2
        x = x2 - w
        y = y2 - h
        draw_filled_arc(
            surface,
            (255, 255, 255, 255),
            pygame.Rect(x, y, w, h),
            0,
            90,
        )

    def _render_upper_inner_pie(
        self, surface: pygame.Surface, y1: int, x2: int
    ) -> None:
        w = int(self.corner_radius * self.upper_inner_radius_x) * 2
        h = int(self.corner_radius * self.upper_inner_radius_y) * 2
        x = x2 - w
        y = y1
        draw_filled_arc(
            surface,
            (255, 255, 255, 255),
            pygame.Rect(x, y, w, h),
            270,
            360,
        )

    def _render_upper_outer_pie(
        self, surface: pygame.Surface, x1: int, y1: int
    ) -> None:
        w = int(self.corner_radius * self.upper_outer_radius_x) * 2
        h = int(self.corner_radius * self.upper_outer_radius_y) * 2
        x = x1
        y = y1
        draw_filled_arc(
            surface,
            (255, 255, 255, 255),
            pygame.Rect(x, y, w, h),
            180,
            270,
        )

    def _render_lower_outer_pie(
        self, surface: pygame.Surface, x1: int, y2: int
    ) -> None:
        w = int(self.corner_radius * self.lower_outer_radius_x) * 2
        h = int(self.corner_radius * self.lower_outer_radius_y) * 2
        x = x1
        y = y2 - h
        draw_filled_arc(
            surface,
            (255, 255, 255, 255),
            pygame.Rect(x, y, w, h),
            90,
            180,
        )

    def render(self, parent_surface: pygame.Surface) -> None:
        # Eye surface
        eye_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        eye_surf.fill((0, 0, 0, 0))  # transparent

        # geometry
        x1 = self.width // 2 - self.half_eye_width
        y1 = self.height // 2 - self.half_eye_height
        x2 = self.width // 2 + self.half_eye_width
        y2 = self.height // 2 + self.half_eye_height

        # sub-renders
        self._render_inner_rect(eye_surf, x1, y1, x2, y2)
        self._render_upper_rect(eye_surf, x1, y1, x2)
        self._render_outer_rect(eye_surf, x1, y1, y2)
        self._render_lower_rect(eye_surf, x1, x2, y2)
        self._render_center_rect(eye_surf, x1, y1, x2, y2)
        self._render_lower_inner_pie(eye_surf, x2, y2)
        self._render_upper_inner_pie(eye_surf, y1, x2)
        self._render_upper_outer_pie(eye_surf, x1, y1)
        self._render_lower_outer_pie(eye_surf, x1, y2)

        # lids
        for lid in self.lids:
            lid.render(eye_surf)

        # rotate
        rotated_eye = pygame.transform.rotate(eye_surf, self.angle)

        # scale
        new_w = int(rotated_eye.get_width() * self.scale_x)
        new_h = int(rotated_eye.get_height() * self.scale_y)
        if new_w <= 0 or new_h <= 0:
            return

        scaled_eye = pygame.transform.smoothscale(rotated_eye, (new_w, new_h))

        # blit
        x_pos = int(
            (parent_surface.get_width() - scaled_eye.get_width()) / 2
            + self.center_x * X_FACTOR
            + self.x_offset
        )
        y_pos = int(
            (parent_surface.get_height() - scaled_eye.get_height()) / 2
            + self.center_y * Y_FACTOR
        )
        parent_surface.blit(scaled_eye, (x_pos, y_pos))


############################
# The Full Procedural Face #
############################


class ProceduralFace(ProceduralBase):
    __slots__ = ("eyes",)

    def __init__(
        self,
        params: Optional[List[float]] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ):
        if params is None:
            params = [
                0.0,  # face center_x
                0.0,  # face center_y
                1.0,  # face scale_x
                1.0,  # face scale_y
                0.0,  # face angle
                # left eye (19 params)
                0.0,  # center_x
                0.0,  # center_y
                1.0,  # scale_x
                1.0,  # scale_y
                0.0,  # angle
                0.5,  # lower_inner_radius_x
                0.5,  # lower_inner_radius_y
                0.5,  # lower_outer_radius_x
                0.5,  # lower_outer_radius_y
                0.5,  # upper_inner_radius_x
                0.5,  # upper_inner_radius_y
                0.5,  # upper_outer_radius_x
                0.5,  # upper_outer_radius_y
                0.0,  # lid top y
                0.0,  # lid top angle
                0.0,  # lid top bend
                0.0,  # lid bottom y
                0.0,  # lid bottom angle
                0.0,  # lid bottom bend
                # right eye (19 params)
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        if not isinstance(params, list) or len(params) < 43:
            raise ValueError(
                "Procedural face parameters must be a list of 43 floats."
            )
        super().__init__(params, 0, width, height)

        eye_offset = int(self.width / 5)
        # left eye at offset=5, right eye at offset=24
        self.eyes = (
            ProceduralEye(params, 5, -eye_offset, self.width, self.height),
            ProceduralEye(params, 5 + 19, eye_offset, self.width, self.height),
        )

    @property
    def center_x(self) -> float:
        return self.params[self.offset + 0]

    @center_x.setter
    def center_x(self, value: float) -> None:
        self.params[self.offset + 0] = value

    @property
    def center_y(self) -> float:
        return self.params[self.offset + 1]

    @center_y.setter
    def center_y(self, value: float) -> None:
        self.params[self.offset + 1] = value

    @property
    def scale_x(self) -> float:
        return self.params[self.offset + 2]

    @scale_x.setter
    def scale_x(self, value: float) -> None:
        self.params[self.offset + 2] = value

    @property
    def scale_y(self) -> float:
        return self.params[self.offset + 3]

    @scale_y.setter
    def scale_y(self, value: float) -> None:
        self.params[self.offset + 3] = value

    @property
    def angle(self) -> float:
        return self.params[self.offset + 4]

    @angle.setter
    def angle(self, value: float) -> None:
        self.params[self.offset + 4] = value

    def render(self, parent_surface: pygame.Surface) -> pygame.Surface:
        face_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        face_surf.fill((0, 0, 0, 0))

        for eye in self.eyes:
            eye.render(face_surf)

        rotated_face = pygame.transform.rotate(face_surf, self.angle)
        new_w = int(rotated_face.get_width() * self.scale_x)
        new_h = int(rotated_face.get_height() * self.scale_y)
        if new_w <= 0 or new_h <= 0:
            return parent_surface
        scaled_face = pygame.transform.smoothscale(
            rotated_face, (new_w, new_h)
        )

        x_pos = int(
            (parent_surface.get_width() - scaled_face.get_width()) / 2
            + self.center_x * X_FACTOR
        )
        y_pos = int(
            (parent_surface.get_height() - scaled_face.get_height()) / 2
            + self.center_y * Y_FACTOR
        )
        parent_surface.blit(scaled_face, (x_pos, y_pos))
        return parent_surface


######################
# Interpolation Logic
######################


def interpolate(
    from_face: ProceduralFace, to_face: ProceduralFace, steps: int
) -> Generator[ProceduralFace, None, None]:
    """
    Given two ProceduralFace objects, generate interpolated
    ProceduralFace objects over a specified number of steps.
    """
    if steps < 2:
        raise ValueError("At least 2 steps needed for interpolation.")

    for step in range(steps):
        t = step / (steps - 1)
        # Interpolate each param individually
        new_params = []
        for i in range(len(from_face.params)):
            start_val = from_face.params[i]
            end_val = to_face.params[i]
            interp_val = start_val + (end_val - start_val) * t
            new_params.append(interp_val)

        # Create a new ProceduralFace from these interpolated parameters
        face = ProceduralFace(new_params, from_face.width, from_face.height)
        yield face
