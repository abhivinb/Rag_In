"""Chunking validation tests."""

import pytest

from app.ingestion.chunking.models import ChunkingConfig
from app.ingestion.chunking.validators import (
    ChunkingConfigurationError,
    validate_config,
)


def test_valid_configuration() -> None:
    assert validate_config(
        ChunkingConfig(chunk_size=100, chunk_overlap=20, minimum_chunk_size=10)
    ).chunk_size == 100


@pytest.mark.parametrize(
    "config",
    [
        ChunkingConfig(chunk_size=0),
        ChunkingConfig(chunk_size=10, chunk_overlap=-1),
        ChunkingConfig(chunk_size=10, chunk_overlap=10),
        ChunkingConfig(chunk_size=10, minimum_chunk_size=-1),
        ChunkingConfig(chunk_size=10, minimum_chunk_size=11),
    ],
)
def test_invalid_configuration_is_rejected(config: ChunkingConfig) -> None:
    with pytest.raises(ChunkingConfigurationError):
        validate_config(config)