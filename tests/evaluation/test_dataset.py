"""Evaluation dataset validation tests."""

import json

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.exceptions import EvaluationDatasetError
from app.evaluation.models import EvaluationSample


def valid_sample(identifier: str = "sample-1") -> dict:
    return {
        "id": identifier,
        "question": "What is the policy?",
        "expected_answer": "The policy is documented.",
        "expected_contexts": ["The policy is documented."],
    }


def test_dataset_loads_valid_samples_in_deterministic_order(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([valid_sample("b"), valid_sample("a")]), encoding="utf-8")

    samples = load_dataset(path)

    assert [sample.id for sample in samples] == ["a", "b"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {**valid_sample(), "question": ""},
        {**valid_sample(), "expected_answer": " "},
        {**valid_sample(), "expected_contexts": []},
        {**valid_sample(), "expected_contexts": [""]},
    ],
)
def test_malformed_sample_is_rejected(tmp_path, payload) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([payload]), encoding="utf-8")

    with pytest.raises(EvaluationDatasetError):
        load_dataset(path)


def test_dataset_requires_json_array(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"samples": []}), encoding="utf-8")

    with pytest.raises(EvaluationDatasetError):
        load_dataset(path)


def test_empty_dataset_is_valid(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text("[]", encoding="utf-8")

    assert load_dataset(path) == []