"""Optional DeepSORT association with original matched detections for counting."""

from contextlib import nullcontext
from importlib.resources import files

import numpy as np

from vehicle_counter.exporting import file_sha256


class DeepSortAssociation:
    def __init__(self, settings: dict) -> None:
        import torch

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as error:
            raise ImportError("Pasang DeepSORT dengan pip install -e '.[deepsort]'.") from error

        self._torch = torch
        self._portable_cpu = torch.backends.cpu.get_cpu_capability() == "DEFAULT"
        with self._cpu_context(), torch.inference_mode():
            self._tracker = DeepSort(
                **{key: value for key, value in settings.items() if key != "tracker_type"},
                embedder="mobilenet",
                embedder_gpu=False,
                half=False,
            )
        weights = files("deep_sort_realtime").joinpath(
            "embedder/weights/mobilenetv2_bottleneck_wts.pt"
        )
        self.metadata = {
            "model": "mobilenetv2_imagenet_bottleneck",
            "weights_sha256": file_sha256(weights),
            "device": "cpu",
            "vehicle_reid_finetuned": False,
        }

    def _cpu_context(self):
        if self._portable_cpu:
            return self._torch.backends.mkldnn.flags(enabled=False)
        return nullcontext()

    def update(self, detections, *, img=None):
        raw = [
            ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(score), int(class_id))
            for (x1, y1, x2, y2), score, class_id in zip(
                detections.xyxy, detections.conf, detections.cls, strict=True
            )
        ]
        with self._cpu_context(), self._torch.inference_mode():
            tracks = self._tracker.update_tracks(raw, frame=img, others=list(range(len(raw))))
        rows = []
        for track in tracks:
            box = track.to_ltrb(orig=True, orig_strict=True)
            if not track.is_confirmed() or track.time_since_update != 0 or box is None:
                continue
            index = track.get_det_supplementary()
            if index is None or not 0 <= index < len(raw):
                raise ValueError("DeepSORT mengembalikan indeks deteksi yang tidak valid.")
            rows.append(
                [
                    *box,
                    int(track.track_id),
                    float(detections.conf[index]),
                    int(detections.cls[index]),
                    index,
                ]
            )
        return np.asarray(rows, dtype=np.float64).reshape(-1, 8)
