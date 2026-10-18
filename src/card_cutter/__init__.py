"""Card Cutter: local, cross-platform trading-card photo processing."""

from card_cutter.models import ProcessingOptions, ProcessingStatus, ProcessResult
from card_cutter.processor import process_image

__all__ = ["ProcessingOptions", "ProcessResult", "ProcessingStatus", "process_image"]
__version__ = "1.0.0"

