from vehicle_counter.contracts import CountEvent, TrackObservation
from vehicle_counter.crossing_audit import audit_crossings
from vehicle_counter.tracking import VehicleDetection


def test_overlap_flag_is_review_evidence_and_keeps_event_class_and_position():
    observation = TrackObservation(10, 1.0, 7, "truck", 0.8, (10, 10, 30, 40))
    event = CountEvent("run", "main", 7, 10, 1.0, "car", "A_TO_B")
    own = VehicleDetection("truck", 0.8, observation.bbox_xyxy)
    other = VehicleDetection("car", 0.9, (15, 20, 40, 60))
    rows = audit_crossings([event], [observation], [own, other])
    assert len(rows) == 1
    assert rows[0]["class_name"] == "car"  # final domain class, after optional review
    assert (rows[0]["anchor_x"], rows[0]["anchor_y"]) == (20, 40)
    assert rows[0]["overlapping_boxes"] == 1
    assert rows[0]["review_reason"] == "anchor_inside_other_detection"
    assert audit_crossings([event], [observation], [own])[0]["review_reason"] == ""
    assert (
        audit_crossings([event], [observation], None)[0]["review_reason"]
        == "detections_unavailable"
    )
