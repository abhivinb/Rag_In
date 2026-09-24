"""Pydantic models for controlled evaluation datasets and reports."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationSample(BaseModel):
    """One explicit, reviewable evaluation datum."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_answer: str
    expected_contexts: list[str] = Field(min_length=1)
    metadata: dict[str, Any] | None = None

    @field_validator("id", "question", "expected_answer")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty or whitespace-only")
        return value

    @field_validator("expected_contexts")
    @classmethod
    def contexts_must_be_meaningful(cls, value: list[str]) -> list[str]:
        if not value or any(not context.strip() for context in value):
            raise ValueError("expected_contexts must contain non-empty strings")
        return value


class EvaluationMetricResult(BaseModel):
    """One metric score and its threshold decision."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str | None = None


class EvaluationSampleResult(BaseModel):
    """Metrics for one executed evaluation sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    question: str
    answer: str
    actual_contexts: list[str]
    expected_contexts: list[str]
    metric_results: list[EvaluationMetricResult]


class MetricSummary(BaseModel):
    """Aggregate statistics for one metric."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    sample_count: int = Field(ge=0)
    average_score: float = Field(ge=0.0, le=1.0)
    minimum_score: float = Field(ge=0.0, le=1.0)
    maximum_score: float = Field(ge=0.0, le=1.0)
    pass_rate: float = Field(ge=0.0, le=1.0)


class EvaluationReport(BaseModel):
    """Machine-readable offline evaluation report."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_id: str
    evaluation_model: str
    total_samples: int = Field(ge=0)
    passed_samples: int = Field(ge=0)
    failed_samples: int = Field(ge=0)
    metric_summary: list[MetricSummary]
    sample_results: list[EvaluationSampleResult]


def evaluation_thresholds_from_settings(settings) -> dict[str, float]:
    """Extract the five evaluation thresholds without coupling production RAG."""
    return {
        "faithfulness": settings.eval_faithfulness_threshold,
        "answer_relevancy": settings.eval_answer_relevancy_threshold,
        "contextual_relevancy": settings.eval_context_relevancy_threshold,
        "contextual_precision": settings.eval_context_precision_threshold,
        "contextual_recall": settings.eval_context_recall_threshold,
    }