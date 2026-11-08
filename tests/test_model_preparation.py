import hashlib
import json

import pytest

from vehicle_counter.tracking import prepare_model


def fake_responses(monkeypatch, *, checksum_matches=True):
    requests = pytest.importorskip("requests")
    payload = b"test fixture, not real weights" * 50000
    checksum = hashlib.sha256(payload).hexdigest()
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "sha": "revision-from-official-api",
                "siblings": [
                    {
                        "rfilename": "yolo11n.pt",
                        "lfs": {
                            "sha256": checksum if checksum_matches else "0" * 64,
                        },
                    }
                ],
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, chunk_size):
            yield payload

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(requests, "get", get)
    return payload, calls


def test_official_download_verifies_checksum_and_pins_revision(monkeypatch, tmp_path):
    payload, calls = fake_responses(monkeypatch)
    path = prepare_model(tmp_path)
    assert path.read_bytes() == payload
    assert "/resolve/revision-from-official-api/" in calls[1]
    provenance = json.loads(path.with_suffix(".pt.source.json").read_text())
    assert provenance["sha256"] == provenance["expected_sha256"]
    assert prepare_model(tmp_path) == path
    assert len(calls) == 2


def test_bad_checksum_never_promotes_download_to_model_file(monkeypatch, tmp_path):
    fake_responses(monkeypatch, checksum_matches=False)
    with pytest.raises(RuntimeError, match="Checksum"):
        prepare_model(tmp_path)
    assert not (tmp_path / "yolo11n.pt").exists()


def test_invalid_source_and_truncated_cached_file_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Sumber"):
        prepare_model(tmp_path, source="random-mirror")
    (tmp_path / "yolo11n.pt").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="terlalu kecil"):
        prepare_model(tmp_path)
