"""Deterministic JSON evaluation dataset loading."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.evaluation.exceptions import EvaluationDatasetError
from app.evaluation.models import EvaluationSample


def load_dataset(path: str | Path) -> list[EvaluationSample]:
    """Load and validate a JSON list of evaluation samples."""
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationDatasetError("Unable to read the evaluation dataset.") from error
    if not isinstance(payload, list):
        raise EvaluationDatasetError("Evaluation dataset must be a JSON array.")
    try:
        samples = [EvaluationSample.model_validate(item) for item in payload]
    except (ValidationError, TypeError) as error:
        raise EvaluationDatasetError("Evaluation dataset contains invalid samples.") from error
    return sorted(samples, key=lambda sample: sample.id)