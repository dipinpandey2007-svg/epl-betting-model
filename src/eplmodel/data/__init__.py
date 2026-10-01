"""Data ingestion, cleaning, loading and validation."""

from eplmodel.data.load import load_matches
from eplmodel.data.validate import DataValidationError, validate_matches

__all__ = ["DataValidationError", "load_matches", "validate_matches"]
