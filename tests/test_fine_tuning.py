import json

import numpy as np
import pytest
import yaml

cv2 = pytest.importorskip("cv2")

from vehicle_counter.fine_tuning import prepare_dataset, remap_dataset  # noqa: E402


@pytest.fixture
def label_manifest(tmp_path):
    records = []
    for index, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"image-{index}.png"
        cv2.imwrite(str(image), np.full((40, 80, 3), index * 60, dtype=np.uint8))
        records.append(
            {
                "image": str(image),
                "split": split,
                "source_group": f"{index + 1:064x}",
                "reviewed": True,
                "reviewer": "fixture-agent",
                "boxes": [{"class_name": "car", "bbox": [8, 4, 40, 30]}],
            }
        )
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path, records


def test_dataset_labels_and_provenance_are_consistent(label_manifest, tmp_path):
    manifest, records = label_manifest
    records[2]["boxes"] = []
    manifest.write_text(json.dumps(records), encoding="utf-8")
    data = prepare_dataset(manifest, tmp_path / "dataset")
    config = yaml.safe_load(data.read_text())
    assert list(config["names"].values()) == ["car", "motorcycle", "bus", "truck"]
    label = next((data.parent / "labels" / "train").glob("*.txt")).read_text().split()
    assert int(label[0]) == 0
    assert list(map(float, label[1:])) == [0.3, 0.425, 0.4, 0.65]
    assert next((data.parent / "labels" / "test").glob("*.txt")).read_text() == ""
    provenance = json.loads((data.parent / "provenance.json").read_text())
    assert provenance["frames_per_split"] == {"train": 1, "val": 1, "test": 1}
    assert provenance["human_reviewed"] is False
    assert "bus" in provenance["missing_train_classes"]
    with pytest.raises(ValueError, match="tidak ditimpa"):
        prepare_dataset(manifest, data.parent)


@pytest.mark.parametrize("failure", ["unreviewed", "source_leak", "duplicate_image", "bad_box"])
def test_invalid_labels_fail_before_dataset_is_written(label_manifest, tmp_path, failure):
    manifest, records = label_manifest
    if failure == "unreviewed":
        records[0]["reviewed"] = False
    elif failure == "source_leak":
        records[1]["source_group"] = records[0]["source_group"]
    elif failure == "duplicate_image":
        records[1]["image"] = records[0]["image"]
    else:
        records[0]["boxes"][0]["bbox"][2] = 800
    manifest.write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare_dataset(manifest, output)
    assert not output.exists()


def test_retained_head_maps_by_class_names(label_manifest, tmp_path):
    manifest, records = label_manifest
    records[0]["boxes"] += [
        {"class_name": "truck", "bbox": [40, 10, 70, 35]},
        {"class_name": "motorcycle", "bbox": [2, 10, 7, 30]},
    ]
    manifest.write_text(json.dumps(records), encoding="utf-8")
    source = prepare_dataset(manifest, tmp_path / "dataset")
    names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    mapped = remap_dataset(source, names, tmp_path / "mapped")
    label = next((mapped.parent / "labels" / "train").glob("*.txt")).read_text()
    assert [int(line.split()[0]) for line in label.splitlines()] == [2, 7, 3]
    assert yaml.safe_load(mapped.read_text())["names"] == names
    original = next((source.parent / "labels" / "train").glob("*.txt")).read_text()
    assert [int(line.split()[0]) for line in original.splitlines()] == [0, 3, 1]
