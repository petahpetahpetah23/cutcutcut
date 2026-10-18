from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from card_cutter.config import DEFAULT_MARGIN_PX, INPUT_MARGIN_MAX


class ProcessingStatus(StrEnum):
    SUCCESS = "success"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ProcessingOptions:
    margin_px: int = DEFAULT_MARGIN_PX
    straighten: bool = True
    auto_rotate: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.margin_px <= INPUT_MARGIN_MAX:
            raise ValueError(f"margin_px must be between 0 and {INPUT_MARGIN_MAX}")


@dataclass(frozen=True, slots=True)
class DetectionResult:
    quad: np.ndarray | None
    confidence: float
    message: str = ""


@dataclass(frozen=True, slots=True)
class RotationDecision:
    degrees_clockwise: int
    confidence: float
    scores: dict[int, float]
    certain: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class ProcessResult:
    input_path: Path
    output_path: Path | None
    status: ProcessingStatus
    crop_confidence: float = 0.0
    rotation_degrees: int = 0
    rotation_confidence: float = 0.0
    message: str = ""

