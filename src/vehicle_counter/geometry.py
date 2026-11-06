"""Finite line geometry in original-frame pixel coordinates."""

from dataclasses import dataclass
from math import hypot

from vehicle_counter.contracts import CountingLineConfig, Point, _integer


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Include touches at segment endpoints; never extend either segment."""
    tolerance = 1e-7

    def sign(value: float) -> int:
        return 1 if value > tolerance else -1 if value < -tolerance else 0

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (
            min(p[0], r[0]) - tolerance <= q[0] <= max(p[0], r[0]) + tolerance
            and min(p[1], r[1]) - tolerance <= q[1] <= max(p[1], r[1]) + tolerance
        )

    s1, s2 = sign(_cross(a, b, c)), sign(_cross(a, b, d))
    s3, s4 = sign(_cross(c, d, a)), sign(_cross(c, d, b))
    if s1 * s2 < 0 and s3 * s4 < 0:
        return True
    return (
        (s1 == 0 and on_segment(a, c, b))
        or (s2 == 0 and on_segment(a, d, b))
        or (s3 == 0 and on_segment(c, a, d))
        or (s4 == 0 and on_segment(c, b, d))
    )


@dataclass(frozen=True, slots=True)
class PixelCountingLine:
    p1: Point
    p2: Point
    epsilon_px: float

    @classmethod
    def from_config(
        cls, config: CountingLineConfig, *, width: int, height: int
    ) -> "PixelCountingLine":
        _integer(width, "width", 2)
        _integer(height, "height", 2)
        return cls(
            p1=(config.p1[0] * (width - 1), config.p1[1] * (height - 1)),
            p2=(config.p2[0] * (width - 1), config.p2[1] * (height - 1)),
            epsilon_px=config.epsilon_height_fraction * height,
        )

    def signed_distance(self, point: Point) -> float:
        return _cross(self.p1, self.p2, point) / hypot(
            self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        )

    def side(self, point: Point) -> str | None:
        distance = self.signed_distance(point)
        if distance < -self.epsilon_px:
            return "A"
        if distance > self.epsilon_px:
            return "B"
        return None

    def intersects_path(self, previous: Point, current: Point) -> bool:
        return segments_intersect(previous, current, self.p1, self.p2)
