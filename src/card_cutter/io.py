from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from card_cutter.config import (
    JPEG_QUALITY,
    MAX_FILE_BYTES,
    MAX_IMAGE_PIXELS,
    SUPPORTED_EXTENSIONS,
)

register_heif_opener()
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageValidationError(ValueError):
    """Raised when an input cannot be processed safely as a supported image."""


@dataclass(frozen=True, slots=True)
class LoadedImage:
    pixels: np.ndarray
    icc_profile: bytes | None
    source_format: str


def is_supported_image(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and (path.suffix.lower() in SUPPORTED_EXTENSIONS or _sniff_image_suffix(path) is not None)
    )


def _sniff_image_suffix(path: Path) -> str | None:
    """Return a safe output suffix for supported image bytes.

    This intentionally recognizes only the formats the application already decodes.
    It lets a camera/exported image with a missing filename extension participate in
    batch processing without treating arbitrary files as images.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
    except OSError:
        return None
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brands = {header[8:12], *(header[index : index + 4] for index in range(16, len(header), 4))}
        if brands & {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return ".jpg"
    return None


def natural_key(path: Path | str) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"(\d+)", str(path).casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def discover_images(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise ImageValidationError(f"Input folder does not exist: {root}")
    images = [path for path in root.rglob("*") if is_supported_image(path)]
    return sorted(images, key=lambda item: natural_key(item.relative_to(root)))


def validate_input_path(path: Path) -> None:
    if path.is_symlink():
        raise ImageValidationError("Symbolic-link inputs are not processed")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS and _sniff_image_suffix(path) is None:
        raise ImageValidationError(f"Unsupported image type: {path.suffix or '(none)'}")
    if not path.is_file():
        raise ImageValidationError(f"Image does not exist: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ImageValidationError(f"Cannot inspect image: {exc}") from exc
    if size <= 0:
        raise ImageValidationError("Image file is empty")
    if size > MAX_FILE_BYTES:
        raise ImageValidationError(f"Image exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit")


def load_image(path: Path) -> LoadedImage:
    validate_input_path(path)
    try:
        with Image.open(path) as opened:
            opened.load()
            width, height = opened.size
            if width <= 0 or height <= 0:
                raise ImageValidationError("Image dimensions are invalid")
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit"
                )
            source_format = (opened.format or path.suffix.lstrip(".")).upper()
            icc = opened.info.get("icc_profile")
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
            pixels = np.asarray(normalized, dtype=np.uint8).copy()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("Image exceeds Pillow's decompression safety limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"Cannot decode image: {exc}") from exc
    return LoadedImage(pixels=pixels, icc_profile=icc, source_format=source_format)


def output_suffix_for(source: Path) -> str:
    if source.suffix.lower() == ".png":
        return ".png"
    if source.suffix.lower() in {".jpg", ".jpeg", ".heic"}:
        return ".jpg"
    sniffed = _sniff_image_suffix(source)
    if sniffed is None:
        raise ImageValidationError(f"Unsupported image type: {source.suffix or '(none)'}")
    return sniffed


def output_path_for(
    output_root: Path,
    relative_input: Path,
    *,
    source_path: Path | None = None,
) -> Path:
    suffix = output_suffix_for(source_path or relative_input)
    return output_root / relative_input.with_suffix(suffix)


def unique_path(path: Path, *, reserved: set[Path] | None = None) -> Path:
    reserved = reserved if reserved is not None else set()
    candidate = path
    counter = 2
    while candidate.exists() or candidate in reserved:
        candidate = path.with_name(f"{path.stem}_cropped_{counter}{path.suffix}")
        counter += 1
    reserved.add(candidate)
    return candidate


def ensure_distinct_folders(input_root: Path, output_root: Path) -> None:
    input_resolved = input_root.resolve()
    output_resolved = output_root.resolve()
    if input_resolved == output_resolved:
        raise ValueError("The output folder must be different from the input folder")
    if output_resolved.is_relative_to(input_resolved):
        raise ValueError(
            "The output folder cannot be inside the recursively processed input folder"
        )


def _atomic_destination(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent
    )
    return descriptor, Path(temporary)


def save_image_atomic(pixels: np.ndarray, path: Path, icc_profile: bytes | None = None) -> None:
    descriptor, temporary = _atomic_destination(path)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        image = Image.fromarray(pixels.astype(np.uint8), mode="RGB")
        save_kwargs: dict[str, Any] = {}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update(quality=JPEG_QUALITY, optimize=True, progressive=True)
            image.save(temporary_path, format="JPEG", **save_kwargs)
        elif path.suffix.lower() == ".png":
            image.save(temporary_path, format="PNG", optimize=True, **save_kwargs)
        else:
            raise ImageValidationError(f"Unsupported output type: {path.suffix}")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def copy_original_atomic(source: Path, destination: Path) -> None:
    descriptor, temporary = _atomic_destination(destination)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

