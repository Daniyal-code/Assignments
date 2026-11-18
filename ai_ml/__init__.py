"""Init module of ai_ml and corresponding package info."""

from importlib.metadata import PackageNotFoundError, metadata

__PACKAGE_NAME__ = "txn-classifier"

try:
    _metadata_info = metadata(__PACKAGE_NAME__)

    VERSION = _metadata_info["Version"]
    AUTHOR = _metadata_info.get("Author")
    AUTHOR_EMAIL = _metadata_info.get("Author-email")
    DESCRIPTION = _metadata_info.get("Summary")

except PackageNotFoundError:
    VERSION = "0.1.0"
    AUTHOR = None
    AUTHOR_EMAIL = None
    DESCRIPTION = "AI/ML assignments"