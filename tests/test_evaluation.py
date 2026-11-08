from vehicle_counter.evaluation import Event, evaluate_events


def test_matching_maximizes_pairs_before_choosing_closest_times():
    manual = [Event("a", 10, 1.0, "car", "A_TO_B"), Event("b", 14, 1.4, "car", "A_TO_B")]
    predicted = [
        Event("1", 6, 0.6, "car", "A_TO_B"),
        Event("2", 11, 1.1, "car", "A_TO_B"),
        Event("3", 14, 1.4, "car", "B_TO_A"),
    ]
    result = evaluate_events(manual, predicted)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 1, 0)
    assert [row["predicted"]["event_id"] for row in result["matches"]] == ["1", "2"]
    assert result["extra_events"][0]["event_id"] == "3"


def test_matching_uses_closest_time_when_pair_count_ties():
    manual = [Event("a", 10, 1.0, "truck", "B_TO_A")]
    predicted = [Event("1", 6, 0.6, "truck", "B_TO_A"), Event("2", 11, 1.1, "truck", "B_TO_A")]
    result = evaluate_events(manual, predicted)
    assert result["matches"][0]["predicted"]["event_id"] == "2"


def test_zero_event_clip_has_no_defined_precision_recall_or_count_error():
    result = evaluate_events([], [])
    assert result["precision"] is result["recall"] is result["f1"] is None
    assert result["count_error_percent"] is None
    assert len(result["buckets"]) == 8
