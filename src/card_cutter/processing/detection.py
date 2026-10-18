from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from card_cutter.config import ANALYSIS_MAX_SIDE
from card_cutter.models import DetectionResult
from card_cutter.processing.geometry import order_quad, quad_dimensions


@dataclass(frozen=True, slots=True)
class _Candidate:
    quad: np.ndarray
    confidence: float


def _ray_clearance(
    points: np.ndarray,
    direction: np.ndarray,
    width: int,
    height: int,
) -> float:
    limits: list[float] = []
    for x, y in points:
        if direction[0] > 1e-6:
            limits.append((width - 1 - float(x)) / float(direction[0]))
        elif direction[0] < -1e-6:
            limits.append((0.0 - float(x)) / float(direction[0]))
        if direction[1] > 1e-6:
            limits.append((height - 1 - float(y)) / float(direction[1]))
        elif direction[1] < -1e-6:
            limits.append((0.0 - float(y)) / float(direction[1]))
    return max(0.0, min(limits, default=0.0))


def _restore_truncated_long_edge(quad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Extend a suspiciously squat object toward the available outer boundary.

    Transparent graded slabs can merge into a textured background. Their strong
    inner compartment then looks like the object boundary, most commonly dropping
    the grading-label end. Standard cards and complete slabs are taller than this
    trigger after perspective measurement, so only low-aspect candidates are
    adjusted.
    """
    ordered = order_quad(quad)
    width, height = quad_dimensions(ordered)
    short_side = min(width, height)
    long_side = max(width, height)
    aspect = long_side / max(short_side, 1.0)
    image_height, image_width = shape[:2]
    area_ratio = abs(float(cv2.contourArea(ordered))) / float(image_height * image_width)
    if not (0.22 <= area_ratio <= 0.60 and 1.10 <= aspect < 1.32):
        return ordered

    target_long_side = short_side * 1.62
    needed = target_long_side - long_side
    if needed <= 0:
        return ordered

    edge_choices = ((0, 1), (2, 3)) if height >= width else ((1, 2), (3, 0))

    choices: list[tuple[float, tuple[int, int], np.ndarray]] = []
    for first, second in edge_choices:
        vector = ordered[second] - ordered[first]
        normal = np.array([vector[1], -vector[0]], dtype=np.float32)
        normal /= max(float(np.linalg.norm(normal)), 1e-6)
        points = ordered[[first, second]]
        clearance = _ray_clearance(points, normal, image_width, image_height)
        choices.append((clearance, (first, second), normal))

    clearance, indices, normal = max(choices, key=lambda item: item[0])
    offset = min(needed, max(0.0, clearance - 1.0))
    if offset < needed * 0.65:
        return ordered
    restored = ordered.copy()
    restored[list(indices)] += normal * offset
    return order_quad(restored)


def _resize_for_analysis(pixels: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = pixels.shape[:2]
    largest = max(height, width)
    if largest <= ANALYSIS_MAX_SIDE:
        return pixels, 1.0
    scale = ANALYSIS_MAX_SIDE / largest
    resized = cv2.resize(
        pixels,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _make_masks(image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.55 * median))
    upper = int(min(255, max(lower + 1, 1.45 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        5,
    )
    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_width = max(3, min(image.shape[:2]) // 60)
    border_pixels = np.concatenate(
        [
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ]
    )
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    distance_range = float(np.ptp(distance))
    normalized = (
        np.zeros_like(distance, dtype=np.uint8)
        if distance_range <= 1e-6
        else ((distance - float(distance.min())) * (255.0 / distance_range)).astype(np.uint8)
    )
    _, foreground = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=2,
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(40, min(image.shape[:2]) // 8),
        minLineLength=min(image.shape[:2]) // 4,
        maxLineGap=min(image.shape[:2]) // 20,
    )
    line_evidence = min(1.0, 0.08 * (0 if lines is None else len(lines)))
    return [edges, adaptive, cv2.bitwise_not(adaptive), foreground], edges, line_evidence


def _edge_support(quad: np.ndarray, edges: np.ndarray) -> float:
    edge_band = np.zeros(edges.shape, dtype=np.uint8)
    cv2.polylines(edge_band, [np.round(quad).astype(np.int32)], True, 255, thickness=5)
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    band_pixels = int(np.count_nonzero(edge_band))
    if band_pixels == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(edge_band, dilated_edges)) / band_pixels)


def _score_candidate(
    quad: np.ndarray,
    contour_area: float,
    shape: tuple[int, ...],
    edges: np.ndarray,
    line_evidence: float,
) -> float:
    height, width = shape[:2]
    image_area = float(height * width)
    quad_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    if quad_area <= 0:
        return 0.0
    area_ratio = quad_area / image_area
    if not 0.12 <= area_ratio <= 0.998:
        return 0.0

    rectangularity = min(1.0, max(0.0, contour_area / quad_area))
    candidate_width, candidate_height = quad_dimensions(quad)
    if min(candidate_width, candidate_height) < min(width, height) * 0.2:
        return 0.0
    aspect = max(candidate_width, candidate_height) / max(
        1.0, min(candidate_width, candidate_height)
    )
    if 1.18 <= aspect <= 2.15:
        aspect_score = 1.0
    elif 1.0 <= aspect <= 2.5:
        aspect_score = 0.68
    else:
        aspect_score = max(0.0, 1.0 - abs(aspect - 1.65) / 2.0)

    center = quad.mean(axis=0)
    center_distance = np.linalg.norm(center - np.array([width / 2, height / 2]))
    maximum_distance = np.linalg.norm(np.array([width / 2, height / 2]))
    center_score = max(0.0, 1.0 - float(center_distance / maximum_distance))
    area_score = min(1.0, area_ratio / 0.55)
    if area_ratio > 0.96:
        area_score = 0.88
    edge_score = _edge_support(quad, edges)
    confidence = (
        0.25 * rectangularity
        + 0.22 * edge_score
        + 0.20 * area_score
        + 0.18 * aspect_score
        + 0.10 * center_score
        + 0.05 * line_evidence
    )

    # Edge masks often contain a strong contour around the photograph itself.
    # Under the supported-input assumption the card/slab is fully visible, so a
    # quadrilateral pinned to several image borders is a camera-frame artifact,
    # not the object. A penalty still permits close crops with one nearby corner.
    border_x = max(3.0, width * 0.018)
    border_y = max(3.0, height * 0.018)
    border_corners = int(
        np.count_nonzero(
            (quad[:, 0] <= border_x)
            | (quad[:, 0] >= width - 1 - border_x)
            | (quad[:, 1] <= border_y)
            | (quad[:, 1] >= height - 1 - border_y)
        )
    )
    if border_corners >= 2:
        confidence -= 0.20 + 0.025 * (border_corners - 2)
    return float(np.clip(confidence, 0.0, 1.0))


def _candidate_quads(mask: np.ndarray) -> list[tuple[np.ndarray, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(mask.shape[0] * mask.shape[1])
    output: list[tuple[np.ndarray, float]] = []
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        if not 0.10 * image_area <= area <= 0.999 * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            output.append((order_quad(approximation.reshape(4, 2)), area))
        hull = cv2.convexHull(contour)
        rectangle = cv2.boxPoints(cv2.minAreaRect(hull))
        output.append((order_quad(rectangle), area))
    return output


def detect_card(pixels: np.ndarray) -> DetectionResult:
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        return DetectionResult(None, 0.0, "Expected an RGB image")
    analysis, scale = _resize_for_analysis(pixels)
    if float(np.std(cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY))) < 3.0:
        return DetectionResult(None, 0.0, "The image has too little contrast to find an edge")
    masks, edges, line_evidence = _make_masks(analysis)
    if np.count_nonzero(edges) < edges.size * 0.001:
        return DetectionResult(None, 0.0, "The image has too few edges to find a card or slab")
    candidates: list[_Candidate] = []
    seen: set[tuple[int, ...]] = set()
    for mask in masks:
        for quad, contour_area in _candidate_quads(mask):
            signature = tuple(np.round(quad.reshape(-1) / 8).astype(int))
            if signature in seen:
                continue
            seen.add(signature)
            confidence = _score_candidate(
                quad,
                contour_area,
                analysis.shape,
                edges,
                line_evidence,
            )
            if confidence > 0:
                candidates.append(_Candidate(quad=quad, confidence=confidence))

    if not candidates:
        return DetectionResult(None, 0.0, "No card- or slab-shaped boundary was found")
    best = max(candidates, key=lambda item: item.confidence)
    restored_quad = _restore_truncated_long_edge(best.quad, analysis.shape)
    full_resolution_quad = restored_quad / scale
    return DetectionResult(
        quad=full_resolution_quad.astype(np.float32),
        confidence=best.confidence,
        message="Detected the strongest rectangular card/slab boundary",
    )

