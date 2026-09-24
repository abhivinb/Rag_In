"""Evaluation runner and aggregation tests with fake RAG and metrics."""

import json

import pytest

from app.evaluation.models import EvaluationMetricResult
from app.evaluation.runner import EvaluationRunner
from app.rag.models import RAGRequest, RAGResponse


class FakeRAG:
    async def answer(self, session, request: RAGRequest) -> RAGResponse:
        return RAGResponse(
            answer=f"Answer for {request.query}",
            sources=[],
            retrieved_context=[f"Context for {request.query}"],
        )


class FakeMetric:
    name = "fake_metric"

    def evaluate(self, case):
        return EvaluationMetricResult(
            metric_name=self.name,
            score=0.8,
            threshold=0.7,
            passed=True,
            reason="deterministic fake",
        )


@pytest.mark.asyncio
async def test_runner_executes_dataset_and_aggregates_metrics(tmp_path) -> None:
    dataset = [
        {
            "id": "sample-1",
            "question": "Question",
            "expected_answer": "Answer",
            "expected_contexts": ["Expected context"],
        }
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    report = await EvaluationRunner(FakeRAG(), [FakeMetric()], "fake-model").run(path)

    assert report.total_samples == 1
    assert report.passed_samples == 1
    assert report.failed_samples == 0
    assert report.metric_summary[0].average_score == 0.8
    assert report.sample_results[0].actual_contexts == ["Context for Question"]


@pytest.mark.asyncio
async def test_empty_dataset_returns_empty_report(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text("[]", encoding="utf-8")

    report = await EvaluationRunner(FakeRAG(), [FakeMetric()], "fake-model").run(path)

    assert report.total_samples == 0
    assert report.passed_samples == 0
    assert report.metric_summary == []