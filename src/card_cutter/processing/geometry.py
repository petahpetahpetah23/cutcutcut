from __future__ import annotations

import cv2
import numpy as np


def order_quad(points: np.ndarray) -> np.ndarray:
    quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = quad.sum(axis=1)
    differences = np.diff(quad, axis=1).reshape(-1)
    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = quad[np.argmin(sums)]  # top-left
    ordered[2] = quad[np.argmax(sums)]  # bottom-right
    ordered[1] = quad[np.argmin(differences)]  # top-right
    ordered[3] = quad[np.argmax(differences)]  # bottom-left
    return ordered


def quad_dimensions(quad: np.ndarray) -> tuple[float, float]:
    top_left, top_right, bottom_right, bottom_left = order_quad(quad)
    width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    height = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    return float(width), float(height)


def expand_quad(quad: np.ndarray, margin_px: float, image_shape: tuple[int, ...]) -> np.ndarray:
    ordered = order_quad(quad)
    if margin_px <= 0:
        return ordered
    top_left, top_right, bottom_right, bottom_left = ordered
    horizontal = (top_right + bottom_right) / 2 - (top_left + bottom_left) / 2
    vertical = (bottom_left + bottom_right) / 2 - (top_left + top_right) / 2
    horizontal /= max(float(np.linalg.norm(horizontal)), 1e-6)
    vertical /= max(float(np.linalg.norm(vertical)), 1e-6)
    expanded = np.array(
        [
            top_left - horizontal * margin_px - vertical * margin_px,
            top_right + horizontal * margin_px - vertical * margin_px,
            bottom_right + horizontal * margin_px + vertical * margin_px,
            bottom_left - horizontal * margin_px + vertical * margin_px,
        ],
        dtype=np.float32,
    )
    height, width = image_shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded


def crop_image(
    pixels: np.ndarray,
    quad: np.ndarray,
    margin_px: int,
    *,
    straighten: bool,
) -> np.ndarray:
    ordered = order_quad(quad)
    height, width = pixels.shape[:2]
    if not straighten:
        x_min = max(0, int(np.floor(ordered[:, 0].min())) - margin_px)
        y_min = max(0, int(np.floor(ordered[:, 1].min())) - margin_px)
        x_max = min(width, int(np.ceil(ordered[:, 0].max())) + margin_px + 1)
        y_max = min(height, int(np.ceil(ordered[:, 1].max())) + margin_px + 1)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Detected crop has no usable area")
        return pixels[y_min:y_max, x_min:x_max].copy()

    expanded = expand_quad(ordered, margin_px, pixels.shape)
    output_width, output_height = quad_dimensions(expanded)
    target_width = max(1, int(round(output_width)))
    target_height = max(1, int(round(output_height)))
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(expanded, destination)
    return cv2.warpPerspective(
        pixels,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

