from __future__ import annotations

import math
import os
import threading
import unicodedata
from typing import Any, Protocol

import cv2
import numpy as np

from card_cutter.config import OCR_MAX_SIDE, ROTATION_ADVANTAGE_THRESHOLD
from card_cutter.models import RotationDecision


class OCRScorer(Protocol):
    def __call__(self, pixels: np.ndarray) -> float: ...


_engine: Any | None = None
_engine_lock = threading.Lock()


def _get_engine() -> Any:
    global _engine
    with _engine_lock:
        if _engine is None:
            os.environ["ORT_DISABLE_TELEMETRY"] = "1"
            import onnxruntime as ort  # type: ignore[import-untyped]
            from rapidocr import RapidOCR

            ort.disable_telemetry_events()
            _engine = RapidOCR()
    return _engine


def _box_aspect(box: Any) -> float:
    try:
        points = np.asarray(box, dtype=np.float32).reshape(4, 2)
        width = (np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])) / 2
        height = (np.linalg.norm(points[3] - points[0]) + np.linalg.norm(points[2] - points[1])) / 2
        return float(width / max(height, 1e-6))
    except (TypeError, ValueError):
        return 1.0


def _parse_result(result: Any) -> tuple[list[str], list[float], list[float]]:
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if texts is not None and scores is not None:
        boxes = getattr(result, "boxes", ())
        aspects = [_box_aspect(box) for box in boxes]
        if len(aspects) != len(texts):
            aspects = [1.0] * len(texts)
        return [str(text) for text in texts], [float(score) for score in scores], aspects
    if isinstance(result, (list, tuple)):
        parsed_texts: list[str] = []
        parsed_scores: list[float] = []
        parsed_aspects: list[float] = []
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                parsed_texts.append(str(item[1]))
                parsed_scores.append(float(item[2]))
                parsed_aspects.append(_box_aspect(item[0]))
        return parsed_texts, parsed_scores, parsed_aspects
    return [], [], []


def rapidocr_score(pixels: np.ndarray) -> float:
    score, _texts = _rapidocr_evidence(pixels)
    return score


def _rapidocr_evidence(pixels: np.ndarray) -> tuple[float, list[str]]:
    height, width = pixels.shape[:2]
    largest = max(height, width)
    prepared = pixels
    if largest > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / largest
        prepared = cv2.resize(
            pixels,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    result = _get_engine()(prepared, use_cls=False)
    texts, confidences, aspects = _parse_result(result)
    score = 0.0
    for text, confidence, aspect in zip(texts, confidences, aspects, strict=False):
        useful_characters = sum(character.isalnum() for character in text)
        if useful_characters >= 2 and math.isfinite(confidence):
            horizontal_weight = min(2.0, max(0.08, aspect))
            score += (
                max(0.0, confidence) * min(useful_characters, 48) * horizontal_weight
            )
    if texts:
        score += min(len(texts), 20) * 0.2
    return score, texts


def rotate_clockwise(pixels: np.ndarray, degrees: int) -> np.ndarray:
    normalized = degrees % 360
    if normalized == 0:
        return pixels.copy()
    if normalized == 90:
        return cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(pixels, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(pixels, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Only right-angle rotations are supported")


def _normalized_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(character for character in decomposed if character.isalnum()).casefold()


def _pokemon_back_orientation(
    rotated_images: dict[int, np.ndarray],
    texts_by_rotation: dict[int, list[str]],
) -> tuple[int, float] | None:
    """Resolve the 180-degree symmetry of the standard blue Pokemon card back.

    The rule is deliberately narrow: OCR must find the Pokemon name, the image
    must have the back's strong blue/yellow palette, and the center Pokeball's
    red half must be substantially stronger above its horizontal midpoint.
    """
    text_matches = any(
        "pokemon" in _normalized_text(text)
        for texts in texts_by_rotation.values()
        for text in texts
    )
    if not text_matches:
        return None

    candidates: list[tuple[float, int]] = []
    for degrees, pixels in rotated_images.items():
        height, width = pixels.shape[:2]
        if height <= width:
            continue
        scale = min(1.0, 600 / height)
        prepared = cv2.resize(
            pixels,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(prepared, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        blue = (
            (hue >= 85)
            & (hue <= 135)
            & (saturation >= 70)
            & (value >= 40)
        )
        yellow = (
            (hue >= 18)
            & (hue <= 42)
            & (saturation >= 100)
            & (value >= 100)
        )
        if float(np.mean(blue)) < 0.20 or float(np.mean(yellow)) < 0.04:
            continue

        center = hsv[
            round(height * scale * 0.28) : round(height * scale * 0.72),
            round(width * scale * 0.22) : round(width * scale * 0.78),
        ]
        center_hue, center_saturation, center_value = cv2.split(center)
        red = (
            ((center_hue <= 18) | (center_hue >= 170))
            & (center_saturation >= 90)
            & (center_value >= 80)
        )
        if float(np.mean(red)) < 0.02:
            continue
        midpoint = red.shape[0] // 2
        upper_red = float(np.mean(red[:midpoint]))
        lower_red = float(np.mean(red[midpoint:]))
        red_advantage = (upper_red + 0.002) / (lower_red + 0.002)
        candidates.append((red_advantage, degrees))

    if not candidates:
        return None
    red_advantage, degrees = max(candidates)
    if red_advantage < 1.5:
        return None
    confidence = min(1.0, max(0.0, (red_advantage - 1.0) / 2.0))
    return degrees, confidence


def choose_text_orientation(
    pixels: np.ndarray,
    scorer: OCRScorer | None = None,
) -> RotationDecision:
    rotated_images = {
        degrees: rotate_clockwise(pixels, degrees) for degrees in (0, 90, 180, 270)
    }
    texts_by_rotation: dict[int, list[str]] = {}
    if scorer is None:
        evidence = {
            degrees: _rapidocr_evidence(rotated)
            for degrees, rotated in rotated_images.items()
        }
        scores = {degrees: item[0] for degrees, item in evidence.items()}
        texts_by_rotation = {degrees: item[1] for degrees, item in evidence.items()}
        pokemon_orientation = _pokemon_back_orientation(rotated_images, texts_by_rotation)
        if pokemon_orientation is not None:
            degrees, confidence = pokemon_orientation
            return RotationDecision(
                degrees_clockwise=degrees,
                confidence=confidence,
                scores=scores,
                certain=True,
                message=(
                    f"Selected {degrees}° clockwise from Pokemon-back color orientation"
                ),
            )
    else:
        scores = {
            degrees: scorer(rotated) for degrees, rotated in rotated_images.items()
        }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_degrees, best_score = ranked[0]
    runner_up = ranked[1][1]
    if best_score <= 0:
        return RotationDecision(
            degrees_clockwise=0,
            confidence=0.0,
            scores=scores,
            certain=False,
            message="No readable text was found in any orientation",
        )
    advantage = math.inf if runner_up <= 0 else best_score / runner_up
    certain = advantage >= ROTATION_ADVANTAGE_THRESHOLD
    confidence = 1.0 if math.isinf(advantage) else min(1.0, max(0.0, (advantage - 1.0) / 0.5))
    message = (
        f"Selected {best_degrees}° clockwise from OCR scores"
        if certain
        else "Text orientation scores were too similar"
    )
    return RotationDecision(
        degrees_clockwise=best_degrees if certain else 0,
        confidence=confidence,
        scores=scores,
        certain=certain,
        message=message,
    )

