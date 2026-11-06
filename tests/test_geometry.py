import pytest

from vehicle_counter.contracts import CountingLineConfig
from vehicle_counter.geometry import PixelCountingLine, segments_intersect


def test_normalized_coordinates_scale_to_original_image():
    config = CountingLineConfig(p1=(0.1, 0.6), p2=(0.9, 0.6))
    line = PixelCountingLine.from_config(config, width=1001, height=501)
    assert line.p1 == (100, 300)
    assert line.p2 == (900, 300)
    assert line.epsilon_px == pytest.approx(5.01)
    assert line.side((500, 290)) == "A"
    assert line.side((500, 310)) == "B"
    assert line.side((500, 300)) is None


def test_diagonal_line_distance_is_normalized():
    line = PixelCountingLine((0, 0), (100, 100), 1)
    assert line.signed_distance((10, 20)) == pytest.approx(10 / 2**0.5)
    assert line.side((10, 20)) == "B"
    assert line.side((20, 10)) == "A"


@pytest.mark.parametrize("dimensions", [(0, 100), (100, 1), (True, 100)])
def test_invalid_canvas_dimensions(dimensions):
    with pytest.raises(ValueError):
        PixelCountingLine.from_config(
            CountingLineConfig(), width=dimensions[0], height=dimensions[1]
        )


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((50, 10), (50, 90), True),
        ((20, 10), (20, 90), True),
        ((10, 10), (10, 90), False),
        ((30, 10), (70, 10), False),
        ((10, 50), (30, 50), True),
        ((0, 50), (10, 50), False),
        ((50, 50), (50, 50), True),
        ((10, 40), (10, 40), False),
    ],
)
def test_finite_segment_intersection(a, b, expected):
    assert segments_intersect(a, b, (20, 50), (80, 50)) is expected
