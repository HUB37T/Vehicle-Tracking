"""Normalized simple polygons, independent of OpenCV and image resolution."""

from math import isfinite
from numbers import Real

_EDGE_TOLERANCE = 1e-12


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_edge(a, b, p):
    return (
        abs(_cross(a, b, p)) < _EDGE_TOLERANCE
        and min(a[0], b[0]) - _EDGE_TOLERANCE <= p[0] <= max(a[0], b[0]) + _EDGE_TOLERANCE
        and min(a[1], b[1]) - _EDGE_TOLERANCE <= p[1] <= max(a[1], b[1]) + _EDGE_TOLERANCE
    )


def polygon_contains(polygon, point) -> bool:
    """Test a point against a validated polygon, including its boundary."""
    x, y = point
    inside = False
    for a, b in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        if _on_edge(a, b, point):
            return True
        ax, ay = a
        bx, by = b
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def validated_polygon(value):
    points = tuple(tuple(p) for p in value)
    if not 3 <= len(points) <= 12 or any(len(p) != 2 for p in points):
        raise ValueError("Poligon memerlukan 3–12 titik dengan dua koordinat.")
    if any(
        isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or not 0 <= v <= 1
        for p in points
        for v in p
    ):
        raise ValueError("Titik poligon harus angka berhingga pada rentang 0–1.")
    if len(set(points)) != len(points):
        raise ValueError("Titik poligon tidak boleh berulang.")
    edges = list(zip(points, points[1:] + points[:1], strict=True))
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)) / 2
    if area < 0.0001:
        raise ValueError("Luas poligon terlalu kecil.")

    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i or j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if (
                _cross(a, b, c) * _cross(a, b, d) < 0
                and _cross(c, d, a) * _cross(c, d, b) < 0
                or any((_on_edge(a, b, c), _on_edge(a, b, d), _on_edge(c, d, a), _on_edge(c, d, b)))
            ):
                raise ValueError("Sisi poligon tidak boleh saling berpotongan.")
    return points
