"""Explainable preflight using grayscale, histogram, Laplacian, FFT and ORB."""

from pathlib import Path

import cv2
import numpy as np

from vehicle_counter.execution import check_cancel
from vehicle_counter.exporting import write_json
from vehicle_counter.video import VideoReader


def frame_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / hist.sum()
    nonzero = probabilities[probabilities > 0]
    reduced = cv2.resize(gray, (128, 128)).astype(np.float32)
    reduced -= reduced.mean()
    # OpenCV's float32 DFT also works on the demo VM's generic x86 CPU.
    frequency = cv2.dft(reduced, flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude = cv2.magnitude(frequency[:, :, 0], frequency[:, :, 1])
    spectrum = np.roll(magnitude, (64, 64), axis=(0, 1))
    power = spectrum.astype(np.float64) ** 2
    yy, xx = np.ogrid[-64:64, -64:64]
    high = (xx**2 + yy**2) > 32**2
    return (
        {
            "brightness_mean": float(gray.mean()),
            "contrast_std": float(gray.std()),
            "dark_fraction": float((gray < 35).mean()),
            "clipped_bright_fraction": float((gray > 245).mean()),
            "histogram_entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "high_frequency_energy_fraction": float(power[high].sum() / power.sum())
            if power.sum()
            else 0.0,
        },
        gray,
        hist,
        spectrum,
    )


def inspect_quality(source: Path, config=None, *, should_cancel=None):
    samples, frames = [], []
    with VideoReader(source) as reader:
        meta = reader.metadata
        last = min((meta.frame_count or round(meta.source_fps * 3)) - 1, round(meta.source_fps * 3))
        targets = set(map(int, np.linspace(0, max(0, last), 6)))
        for index, frame in reader:
            check_cancel(should_cancel)
            if index in targets:
                scale = min(1, 640 / max(frame.shape[:2]))
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                metrics, *_ = frame_metrics(small)
                samples.append(
                    {"frame_index": index, "timestamp_seconds": index / meta.source_fps, **metrics}
                )
                frames.append(small)
            if index >= last:
                break
    motions = []
    evidence_matches = None
    orb = cv2.ORB_create(nfeatures=1000)
    for index, (first, second) in enumerate(zip(frames, frames[1:], strict=False)):
        check_cancel(should_cancel)
        a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(a)
        mask[: round(a.shape[0] * 0.45), :] = 255
        mask[:, : round(a.shape[1] * 0.12)] = 255
        mask[:, -max(1, round(a.shape[1] * 0.12)) :] = 255
        if config is not None:
            if config.detection_roi is not None:
                x1, y1, x2, y2 = config.detection_roi
                mask[
                    round(y1 * a.shape[0]) : round(y2 * a.shape[0]),
                    round(x1 * a.shape[1]) : round(x2 * a.shape[1]),
                ] = 0
            for zone in config.traffic_zones:
                points = np.rint(np.array(zone.polygon) * (a.shape[1] - 1, a.shape[0] - 1)).astype(
                    np.int32
                )
                cv2.fillPoly(mask, [points], 0)
        ka, da = orb.detectAndCompute(a, mask)
        kb, db = orb.detectAndCompute(b, mask)
        if da is None or db is None or min(len(da), len(db)) < 2:
            continue
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        good = [
            pair[0]
            for pair in matches
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        if len(good) < 15:
            continue
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        affine, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2
        )
        if affine is None or int(inliers.sum()) < 15 or inliers.mean() < 0.65:
            continue
        dt = samples[index + 1]["timestamp_seconds"] - samples[index]["timestamp_seconds"]
        translation = float(np.linalg.norm(affine[:, 2]) / np.hypot(*a.shape) / dt)
        rotation = float(abs(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))) / dt)
        motions.append(
            {
                "from_frame": samples[index]["frame_index"],
                "to_frame": samples[index + 1]["frame_index"],
                "inliers": int(inliers.sum()),
                "inlier_fraction": float(inliers.mean()),
                "translation_diagonals_per_second": translation,
                "rotation_degrees_per_second": rotation,
            }
        )
        if evidence_matches is None:
            evidence_matches = cv2.drawMatches(
                first,
                ka,
                second,
                kb,
                [m for m, keep in zip(good, inliers.ravel(), strict=True) if keep][:40],
                None,
                flags=2,
            )
    med = {
        key: float(np.median([s[key] for s in samples]))
        for key in samples[0]
        if key not in ("frame_index", "timestamp_seconds")
    }
    warnings = []
    if med["brightness_mean"] < 65 or med["dark_fraction"] > 0.6:
        warnings.append(
            "Bagian awal video cenderung gelap. Coba bandingkan CLAHE/gamma dengan masukan asli."
        )
    if med["contrast_std"] < 20:
        warnings.append("Kontras awal rendah. Periksa histogram dan bandingkan CLAHE.")
    if med["laplacian_variance"] < 30:
        warnings.append(
            "Detail frekuensi tinggi rendah: kandidat blur atau area minim "
            "tekstur. Periksa frame sumber."
        )
    moving = sum(
        m["translation_diagonals_per_second"] > 0.005 or m["rotation_degrees_per_second"] > 0.3
        for m in motions
    )
    if moving >= 2:
        warnings.append(
            "Beberapa pasangan fitur latar bergerak bersama. Periksa "
            "goyangan/panning kamera sebelum memakai indikator macet."
        )
    if not motions:
        motion_status = "insufficient_features"
    else:
        motion_status = "candidate_camera_motion" if moving >= 2 else "no_motion_flag_in_sample"
    metrics, gray, hist, spectrum = frame_metrics(frames[0])
    histogram = np.full((180, 256, 3), 20, np.uint8)
    normalized = hist / max(1, hist.max()) * 160
    for x, value in enumerate(normalized):
        cv2.line(histogram, (x, 179), (x, 179 - round(value)), (120, 220, 255), 1)
    fft = np.log1p(spectrum)
    fft = cv2.normalize(fft, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    from vehicle_counter.pcd_lab import _contact_sheet

    panel = _contact_sheet(
        [
            ("Frame sumber", frames[0]),
            ("Grayscale", gray),
            ("Histogram grayscale", histogram),
            ("Laplacian | detail", lap),
            ("FFT | log magnitude", fft),
            (
                "ORB + Hamming + RANSAC",
                evidence_matches if evidence_matches is not None else np.zeros_like(frames[0]),
            ),
        ]
    )
    return {
        "samples": samples,
        "median_metrics": med,
        "background_motion": {"status": motion_status, "pairs": motions},
        "warnings": warnings,
        "scope": "Awal video sampai 3 detik; ukuran pengukuran maksimal 640 piksel.",
        "thresholds_are_heuristic": True,
        "materials": [
            "Grayscale",
            "Histogram",
            "Laplacian",
            "FFT",
            "ORB",
            "Hamming matching",
            "RANSAC",
        ],
    }, panel


def export_quality(output: Path, report, panel):
    path = output / "quality.json"
    write_json(path, report)
    image = output / "quality.png"
    ok, encoded = cv2.imencode(".png", panel)
    if not ok:
        raise RuntimeError("Gagal mengodekan panel kualitas.")
    image.write_bytes(encoded.tobytes())
    return {"quality_json": str(path), "quality_png": str(image)}
