"""Normalized simple polygons, independent of OpenCV and image resolution."""

from math import isfinite
from numbers import Real


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

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on(a, b, p):
        return (
            abs(cross(a, b, p)) < 1e-12
            and min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12
        )

    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i or j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if (
                cross(a, b, c) * cross(a, b, d) < 0
                and cross(c, d, a) * cross(c, d, b) < 0
                or any((on(a, b, c), on(a, b, d), on(c, d, a), on(c, d, b)))
            ):
                raise ValueError("Sisi poligon tidak boleh saling berpotongan.")
    return points
