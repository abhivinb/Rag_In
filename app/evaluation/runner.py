"""Offline evaluation execution and report aggregation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from app.evaluation.adapter import EvaluationCase, adapt_response
from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import EvaluationMetric, aggregate_metric_results
from app.evaluation.models import EvaluationReport, EvaluationSampleResult
from app.rag.models import RAGRequest, RAGResponse


class RAGApplication(Protocol):
    async def answer(self, session, request: RAGRequest) -> RAGResponse:
        """Execute one offline evaluation request."""


class EvaluationRunner:
    """Run a controlled dataset against RAG and configured metric adapters."""

    def __init__(
        self,
        rag_application: RAGApplication,
        metrics: Sequence[EvaluationMetric],
        evaluation_model: str,
    ) -> None:
        self.rag_application = rag_application
        self.metrics = list(metrics)
        self.evaluation_model = evaluation_model

    async def run(
        self,
        dataset_path: str | Path,
        *,
        session=None,
        dataset_id: str | None = None,
    ) -> EvaluationReport:
        samples = load_dataset(dataset_path)
        sample_results: list[EvaluationSampleResult] = []
        all_metric_results = []
        for sample in samples:
            response = await self.rag_application.answer(session, RAGRequest(query=sample.question))
            case = adapt_response(sample, response)
            metric_results = [metric.evaluate(case) for metric in self.metrics]
            all_metric_results.extend(metric_results)
            sample_results.append(
                EvaluationSampleResult(
                    sample_id=sample.id,
                    question=sample.question,
                    answer=case.actual_answer,
                    actual_contexts=case.actual_contexts,
                    expected_contexts=sample.expected_contexts,
                    metric_results=metric_results,
                )
            )
        passed_samples = sum(
            bool(result.metric_results) and all(metric.passed for metric in result.metric_results)
            for result in sample_results
        )
        return EvaluationReport(
            dataset_id=dataset_id or Path(dataset_path).stem,
            evaluation_model=self.evaluation_model,
            total_samples=len(sample_results),
            passed_samples=passed_samples,
            failed_samples=len(sample_results) - passed_samples,
            metric_summary=aggregate_metric_results(all_metric_results),
            sample_results=sample_results,
        )


def write_report(report: EvaluationReport, output_path: str | Path) -> None:
    """Write a machine-readable report without secrets."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.model_dump_json(indent=2), encoding="utf-8")