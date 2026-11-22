import numpy as np
import pytest

from vehicle_counter.detection_filter import distinct_vehicle_indices


def test_conflicting_box_keeps_stronger_class_and_original_order():
    boxes = np.array([[10, 20, 80, 100], [11, 20, 81, 100], [120, 20, 190, 100]])
    kept = distinct_vehicle_indices(boxes, np.array([0.29, 0.40, 0.9]), ["car", "truck", "car"])
    assert kept == [1, 2]


@pytest.mark.parametrize("classes", [["car", "car"], ["truck", "truck"], ["motorcycle", "truck"]])
def test_same_class_and_motorcycle_are_not_discarded(classes):
    boxes = np.array([[10, 20, 80, 100], [10, 20, 80, 100]])
    assert distinct_vehicle_indices(boxes, np.array([0.29, 0.40]), classes) == [0, 1]


def test_nearby_distinct_vehicles_survive_even_with_overlap():
    boxes = np.array([[10, 20, 80, 100], [35, 20, 105, 100]])
    assert distinct_vehicle_indices(boxes, np.array([0.29, 0.40]), ["car", "truck"]) == [0, 1]


def test_empty_detections_and_deterministic_tie():
    assert distinct_vehicle_indices(np.empty((0, 4)), np.array([]), []) == []
    boxes = np.array([[10, 20, 80, 100], [10, 20, 80, 100]])
    assert distinct_vehicle_indices(boxes, np.array([0.4, 0.4]), ["car", "truck"]) == [0]
