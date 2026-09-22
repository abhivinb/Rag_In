"""Application HTTP routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "healthy"}
