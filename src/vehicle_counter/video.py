"""Local CFR video input and streaming H.264 output without keeping all frames."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import VideoMetadata


class VideoReader:
    """Validate the first frame and yield every frame once, in source order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file() or self.path.suffix.lower() != ".mp4":
            raise ValueError("Masukan harus berupa berkas video MP4 lokal.")
        self._capture = cv2.VideoCapture(str(self.path))
        self._iterated = False
        try:
            if not self._capture.isOpened():
                raise ValueError("Video tidak dapat dibuka oleh decoder.")
            ok, self._first_frame = self._capture.read()
            if not ok or self._first_frame is None:
                raise ValueError("Video tidak memiliki frame yang dapat dibaca.")
            height, width = self._first_frame.shape[:2]
            reported_count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_count = (
                round(reported_count)
                if math.isfinite(reported_count) and reported_count >= 1
                else None
            )
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            self.metadata = VideoMetadata(
                source_name=self.path.name,
                width=width,
                height=height,
                source_fps=fps,
                frame_count=frame_count,
                duration_seconds=(frame_count / fps if frame_count and fps > 0 else None),
            )
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._capture.release()

    def __iter__(self) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        if self._iterated:
            raise RuntimeError("VideoReader hanya dapat diiterasi sekali; buka reader baru.")
        self._iterated = True
        frame = self._first_frame
        index = 0
        while frame is not None:
            if frame.shape != (self.metadata.height, self.metadata.width, 3):
                raise ValueError(f"Dimensi frame {index} berubah dari metadata awal.")
            yield index, frame
            index += 1
            ok, frame = self._capture.read()
            if not ok:
                frame = None
        if self.metadata.frame_count is not None and index != self.metadata.frame_count:
            raise ValueError(
                f"Video berhenti pada {index} frame; metadata menyatakan "
                f"{self.metadata.frame_count}. Video mungkin terpotong/rusak."
            )


def inspect_video(path: str | Path) -> VideoMetadata:
    """Inspect metadata and one decodable frame; does not validate the whole video."""
    with VideoReader(path) as reader:
        return reader.metadata


class AnnotatedVideoWriter:
    """Stream original-size BGR frames to browser-compatible H.264/yuv420p MP4."""

    def __init__(self, path: Path, metadata: VideoMetadata) -> None:
        if metadata.width % 2 or metadata.height % 2:
            raise ValueError(
                "Ekspor H.264 membutuhkan lebar dan tinggi genap; konversi video dahulu."
            )
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
        self.path = path
        self.metadata = metadata
        self.frame_count = 0
        self._closed = False
        self._log_path = path.with_suffix(".encoder.log")
        self._log = self._log_path.open("wb")
        command = [
            executable,
            "-nostdin",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{metadata.width}x{metadata.height}",
            "-framerate",
            str(metadata.source_fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self._log.close()
            raise

    def __enter__(self) -> AnnotatedVideoWriter:
        return self

    def write(self, frame: NDArray[np.uint8]) -> None:
        if self._closed:
            raise RuntimeError("Video writer sudah ditutup.")
        if frame.dtype != np.uint8 or frame.shape != (self.metadata.height, self.metadata.width, 3):
            raise ValueError("Frame keluaran harus BGR uint8 dengan dimensi sumber.")
        try:
            self._process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("Encoder video berhenti; periksa annotated.encoder.log.") from error
        self.frame_count += 1

    def __exit__(self, exc_type: type | None, *_args: object) -> None:
        self.close(abort=exc_type is not None)

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.stdin:
                try:
                    self._process.stdin.close()
                except BrokenPipeError:
                    pass
            if abort:
                self._process.terminate()
            try:
                code = self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
                if not abort:
                    raise RuntimeError("Encoder video tidak selesai dalam batas waktu.") from None
                return
        finally:
            self._log.close()
        if not abort and (code != 0 or self.frame_count == 0):
            message = self._log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"Ekspor video gagal: {message}")


def verify_output_video(path: Path, metadata: VideoMetadata, expected_frames: int) -> None:
    """Decode the complete output before declaring a run complete."""
    with VideoReader(path) as reader:
        actual = reader.metadata
        if (actual.width, actual.height) != (metadata.width, metadata.height):
            raise RuntimeError("Dimensi video keluaran tidak sesuai sumber.")
        if not math.isclose(actual.source_fps, metadata.source_fps, rel_tol=0.001):
            raise RuntimeError("FPS video keluaran tidak sesuai sumber.")
        count = sum(1 for _ in reader)
        if count != expected_frames:
            raise RuntimeError(f"Video keluaran hanya memiliki {count}/{expected_frames} frame.")
