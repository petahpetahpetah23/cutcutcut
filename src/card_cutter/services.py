from __future__ import annotations

import csv
import os
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from card_cutter.io import (
    copy_original_atomic,
    discover_images,
    ensure_distinct_folders,
    natural_key,
    output_path_for,
    unique_path,
)
from card_cutter.models import ProcessingOptions, ProcessingStatus, ProcessResult
from card_cutter.processor import process_image

ProgressCallback = Callable[[int, int, ProcessResult], None]


def _review_destination(output_root: Path, relative_path: Path) -> Path:
    return output_root / "needs_review" / relative_path


def _copy_for_review(result: ProcessResult, destination: Path) -> ProcessResult:
    try:
        copy_original_atomic(result.input_path, destination)
        return ProcessResult(
            input_path=result.input_path,
            output_path=destination,
            status=result.status,
            crop_confidence=result.crop_confidence,
            rotation_degrees=result.rotation_degrees,
            rotation_confidence=result.rotation_confidence,
            message=result.message,
        )
    except OSError as exc:
        return ProcessResult(
            input_path=result.input_path,
            output_path=None,
            status=ProcessingStatus.ERROR,
            crop_confidence=result.crop_confidence,
            rotation_degrees=result.rotation_degrees,
            rotation_confidence=result.rotation_confidence,
            message=f"{result.message}; original could not be copied for review: {exc}",
        )


def write_report_atomic(results: Sequence[ProcessResult], output_root: Path) -> Path:
    report_path = output_root / "processing_report.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".processing-report-", suffix=".csv", dir=output_root
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "input_path",
                    "output_path",
                    "status",
                    "crop_confidence",
                    "rotation_degrees_clockwise",
                    "rotation_confidence",
                    "message",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        str(result.input_path),
                        str(result.output_path or ""),
                        result.status.value,
                        f"{result.crop_confidence:.4f}",
                        result.rotation_degrees,
                        f"{result.rotation_confidence:.4f}",
                        result.message,
                    ]
                )
        os.replace(temporary_path, report_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report_path


def _cancelled_result(input_path: Path) -> ProcessResult:
    return ProcessResult(
        input_path=input_path,
        output_path=None,
        status=ProcessingStatus.CANCELLED,
        message="Cancelled before processing",
    )


def process_folder(
    input_root: Path,
    output_root: Path,
    options: ProcessingOptions,
    *,
    cancel_event: threading.Event | None = None,
    progress: ProgressCallback | None = None,
) -> list[ProcessResult]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    ensure_distinct_folders(input_root, output_root)
    images = discover_images(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[ProcessResult] = []
    reserved: set[Path] = set()
    total = len(images)
    for index, input_path in enumerate(images, start=1):
        if cancel_event is not None and cancel_event.is_set():
            result = _cancelled_result(input_path)
        else:
            relative = input_path.relative_to(input_root)
            destination = unique_path(
                output_path_for(output_root, relative, source_path=input_path),
                reserved=reserved,
            )
            result = process_image(input_path, destination, options)
            if result.status in {ProcessingStatus.NEEDS_REVIEW, ProcessingStatus.ERROR}:
                review_path = unique_path(
                    _review_destination(output_root, relative), reserved=reserved
                )
                result = _copy_for_review(result, review_path)
        results.append(result)
        if progress is not None:
            progress(index, total, result)
    write_report_atomic(results, output_root)
    return results


def process_selected(
    inputs: Sequence[Path],
    output_root: Path,
    options: ProcessingOptions,
    *,
    cancel_event: threading.Event | None = None,
    progress: ProgressCallback | None = None,
) -> list[ProcessResult]:
    ordered_inputs = list(inputs)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[ProcessResult] = []
    reserved: set[Path] = set()
    total = len(ordered_inputs)
    for index, input_path in enumerate(ordered_inputs, start=1):
        input_path = input_path.resolve()
        if cancel_event is not None and cancel_event.is_set():
            result = _cancelled_result(input_path)
        else:
            destination = unique_path(
                output_path_for(
                    output_root,
                    Path(input_path.name),
                    source_path=input_path,
                ),
                reserved=reserved,
            )
            result = process_image(input_path, destination, options)
            if result.status in {ProcessingStatus.NEEDS_REVIEW, ProcessingStatus.ERROR}:
                review_path = unique_path(
                    _review_destination(output_root, Path(input_path.name)), reserved=reserved
                )
                result = _copy_for_review(result, review_path)
        results.append(result)
        if progress is not None:
            progress(index, total, result)
    write_report_atomic(results, output_root)
    return results


def sorted_selected_paths(paths: Sequence[Path]) -> list[Path]:
    return sorted(paths, key=natural_key)

