"""Errors raised by offline evaluation."""


class EvaluationError(Exception):
    """Base class for evaluation failures."""


class EvaluationDatasetError(EvaluationError):
    """Raised when an evaluation dataset is malformed or unreadable."""


class EvaluationMetricError(EvaluationError):
    """Raised when a metric cannot be evaluated."""