"""Metric protocol, deterministic test metric, and isolated DeepEval integration."""

from collections.abc import Sequence
from typing import Protocol

from app.evaluation.adapter import EvaluationCase
from app.evaluation.exceptions import EvaluationMetricError
from app.evaluation.models import EvaluationMetricResult


class EvaluationMetric(Protocol):
    name: str

    def evaluate(self, case: EvaluationCase) -> EvaluationMetricResult:
        """Evaluate one captured case."""


class DeepEvalMetricAdapter:
    """Adapt a current DeepEval metric's synchronous measure API."""

    def __init__(self, metric, name: str | None = None) -> None:
        self.metric = metric
        self.name = name or metric.__class__.__name__

    def evaluate(self, case: EvaluationCase) -> EvaluationMetricResult:
        try:
            from deepeval.test_case import LLMTestCase

            test_case = LLMTestCase(
                input=case.sample.question,
                actual_output=case.actual_answer,
                expected_output=case.sample.expected_answer,
                context=case.sample.expected_contexts,
                retrieval_context=case.actual_contexts,
            )
            self.metric.measure(test_case)
            score = float(self.metric.score)
            threshold = float(self.metric.threshold)
            return EvaluationMetricResult(
                metric_name=self.name,
                score=score,
                threshold=threshold,
                passed=score >= threshold,
                reason=getattr(self.metric, "reason", None),
            )
        except Exception as error:
            raise EvaluationMetricError(f"Metric {self.name} failed.") from error


def build_deepeval_metrics(thresholds: dict[str, float], model: str) -> list[DeepEvalMetricAdapter]:
    """Build the five supported DeepEval RAG metrics lazily."""
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        )
    except ImportError as error:
        raise EvaluationMetricError(
            "DeepEval is required only for real evaluation runs."
        ) from error
    definitions = (
        ("faithfulness", FaithfulnessMetric),
        ("answer_relevancy", AnswerRelevancyMetric),
        ("contextual_relevancy", ContextualRelevancyMetric),
        ("contextual_precision", ContextualPrecisionMetric),
        ("contextual_recall", ContextualRecallMetric),
    )
    return [
            DeepEvalMetricAdapter(
            metric_class(
                threshold=thresholds[name], model=model, include_reason=True
                ),
                name=name,
        )
        for name, metric_class in definitions
    ]


def aggregate_metric_results(
    results: Sequence[EvaluationMetricResult],
) -> list:
    """Aggregate metric results deterministically, returning empty for no samples."""
    from app.evaluation.models import MetricSummary

    names = sorted({result.metric_name for result in results})
    summaries = []
    for name in names:
        items = [result for result in results if result.metric_name == name]
        scores = [item.score for item in items]
        summaries.append(
            MetricSummary(
                metric_name=name,
                sample_count=len(items),
                average_score=sum(scores) / len(scores),
                minimum_score=min(scores),
                maximum_score=max(scores),
                pass_rate=sum(item.passed for item in items) / len(items),
            )
        )
    return summaries