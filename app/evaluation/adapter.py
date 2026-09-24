"""Adapter from Phase 8 responses to evaluation-friendly cases."""

from dataclasses import dataclass

from app.evaluation.models import EvaluationSample
from app.rag.models import RAGResponse


@dataclass(frozen=True)
class EvaluationCase:
    """Captured actual output alongside explicit ground truth."""

    sample: EvaluationSample
    actual_answer: str
    actual_contexts: list[str]


def adapt_response(sample: EvaluationSample, response: RAGResponse) -> EvaluationCase:
    """Keep runtime contexts separate from dataset expected contexts."""
    return EvaluationCase(
        sample=sample,
        actual_answer=response.answer,
        actual_contexts=list(response.retrieved_context),
    )