"""Replace machine-local roots in committed review artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def portable_review_value(
    value: Any,
    roots: Sequence[tuple[Path, str]],
) -> Any:
    """Recursively replace resolved roots with stable symbolic placeholders."""

    replacements = sorted(
        ((str(path.resolve()), placeholder) for path, placeholder in roots),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def convert(item: Any) -> Any:
        if isinstance(item, Path):
            item = str(item)
        if isinstance(item, str):
            for source, placeholder in replacements:
                item = item.replace(source, placeholder)
            return item
        if isinstance(item, Mapping):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(convert(child) for child in item)
        if isinstance(item, list):
            return [convert(child) for child in item]
        return item

    return convert(value)


def expand_portable_review_value(
    value: Any,
    roots: Sequence[tuple[Path, str]],
) -> Any:
    """Restore symbolic placeholders when validating a local review bundle."""

    replacements = sorted(
        ((placeholder, str(path.resolve())) for path, placeholder in roots),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def convert(item: Any) -> Any:
        if isinstance(item, str):
            for placeholder, target in replacements:
                item = item.replace(placeholder, target)
            return item
        if isinstance(item, Mapping):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(convert(child) for child in item)
        if isinstance(item, list):
            return [convert(child) for child in item]
        return item

    return convert(value)
