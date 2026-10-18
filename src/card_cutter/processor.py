from __future__ import annotations

from pathlib import Path

from card_cutter.config import CROP_CONFIDENCE_THRESHOLD
from card_cutter.io import ImageValidationError, load_image, save_image_atomic
from card_cutter.models import ProcessingOptions, ProcessingStatus, ProcessResult
from card_cutter.processing.detection import detect_card
from card_cutter.processing.geometry import crop_image
from card_cutter.processing.orientation import OCRScorer, choose_text_orientation, rotate_clockwise


def process_image(
    input_path: Path,
    output_path: Path,
    options: ProcessingOptions,
    *,
    ocr_scorer: OCRScorer | None = None,
) -> ProcessResult:
    """Process one image without depending on the desktop interface.

    A low-confidence crop or rotation returns NEEDS_REVIEW and deliberately writes no
    transformed output. The caller decides where to copy the untouched original.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    try:
        loaded = load_image(input_path)
        detection = detect_card(loaded.pixels)
        if detection.quad is None or detection.confidence < CROP_CONFIDENCE_THRESHOLD:
            return ProcessResult(
                input_path=input_path,
                output_path=None,
                status=ProcessingStatus.NEEDS_REVIEW,
                crop_confidence=detection.confidence,
                message=detection.message or "Crop confidence is below the safe threshold",
            )

        cropped = crop_image(
            loaded.pixels,
            detection.quad,
            options.margin_px,
            straighten=options.straighten,
        )
        rotation_degrees = 0
        rotation_confidence = 0.0
        if options.auto_rotate:
            decision = choose_text_orientation(cropped, scorer=ocr_scorer)
            if not decision.certain:
                return ProcessResult(
                    input_path=input_path,
                    output_path=None,
                    status=ProcessingStatus.NEEDS_REVIEW,
                    crop_confidence=detection.confidence,
                    rotation_confidence=decision.confidence,
                    message=decision.message,
                )
            rotation_degrees = decision.degrees_clockwise
            rotation_confidence = decision.confidence
            cropped = rotate_clockwise(cropped, rotation_degrees)

        save_image_atomic(cropped, output_path, loaded.icc_profile)
        return ProcessResult(
            input_path=input_path,
            output_path=output_path,
            status=ProcessingStatus.SUCCESS,
            crop_confidence=detection.confidence,
            rotation_degrees=rotation_degrees,
            rotation_confidence=rotation_confidence,
            message="Processed successfully",
        )
    except ImageValidationError as exc:
        return ProcessResult(
            input_path=input_path,
            output_path=None,
            status=ProcessingStatus.ERROR,
            message=str(exc),
        )
    except Exception as exc:
        return ProcessResult(
            input_path=input_path,
            output_path=None,
            status=ProcessingStatus.ERROR,
            message=f"Processing failed safely: {exc}",
        )

