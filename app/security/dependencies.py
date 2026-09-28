"""Optional request authentication dependencies."""

import secrets

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    *,
    settings: Settings = Depends(get_settings),
) -> None:
    """Require the configured API key for a protected route.

    Authentication is opt-in: deployments without ``API_KEY`` retain the
    existing local development behavior. Comparing with ``compare_digest``
    avoids turning the check into a straightforward timing oracle.
    """
    configured_key = settings.api_key
    if configured_key is None or not configured_key.get_secret_value():
        return
    supplied_key = x_api_key or ""
    if not secrets.compare_digest(supplied_key, configured_key.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid API key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
