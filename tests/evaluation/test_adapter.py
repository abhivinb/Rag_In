"""Evaluation adapter tests."""

from app.evaluation.adapter import adapt_response
from app.evaluation.adapter import EvaluationCase
from app.evaluation.metrics import DeepEvalMetricAdapter
from app.evaluation.models import EvaluationSample
from app.evaluation.models import EvaluationMetricResult
from app.rag.models import RAGResponse, SourceReference


def test_adapter_preserves_actual_and_expected_contexts_separately() -> None:
    sample = EvaluationSample(
        id="sample-1",
        question="Question",
        expected_answer="Expected",
        expected_contexts=["Ground truth context"],
    )
    response = RAGResponse(
        answer="Actual",
        sources=[
            SourceReference(
                chunk_id="chunk-1",
                document_id="doc-1",
                metadata={},
                chunk_index=0,
                retrieval_score=0.8,
            )
        ],
        retrieved_context=["Actual retrieved context"],
    )

    case = adapt_response(sample, response)

    assert case.actual_answer == "Actual"
    assert case.actual_contexts == ["Actual retrieved context"]
    assert case.sample.expected_contexts == ["Ground truth context"]


def test_deepeval_adapter_builds_current_llm_test_case_without_network() -> None:
    sample = EvaluationSample(
        id="sample-1",
        question="Question",
        expected_answer="Expected",
        expected_contexts=["Expected context"],
    )
    captured = {}

    class FakeMetric:
        threshold = 0.7
        score = 0.9
        reason = "fake metric"

        def measure(self, test_case) -> None:
            captured["input"] = test_case.input
            captured["actual_output"] = test_case.actual_output
            captured["context"] = test_case.context
            captured["retrieval_context"] = test_case.retrieval_context

    result = DeepEvalMetricAdapter(FakeMetric(), name="fake").evaluate(
        EvaluationCase(sample, "Actual", ["Actual context"])
    )

    assert isinstance(result, EvaluationMetricResult)
    assert captured == {
        "input": "Question",
        "actual_output": "Actual",
        "context": ["Expected context"],
        "retrieval_context": ["Actual context"],
    }