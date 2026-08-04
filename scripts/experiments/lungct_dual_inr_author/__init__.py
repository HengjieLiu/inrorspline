"""Author-faithful DUAL-INR-DIR experiment adapter.

This package deliberately imports the external implementation at runtime instead
of copying or modifying it.  See :mod:`external_api` for the provenance guard.
"""

from .adapter import (
    AUTHOR_EXTERNAL_COMMIT,
    build_author_config,
    load_transition_manifest,
    train_author,
    validate_transition_manifest,
)

__all__ = [
    "AUTHOR_EXTERNAL_COMMIT",
    "build_author_config",
    "load_transition_manifest",
    "train_author",
    "validate_transition_manifest",
]
