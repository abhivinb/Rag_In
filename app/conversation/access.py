"""Capability tokens for protecting persisted conversation access."""

import hashlib
import hmac

from app.core.config import Settings


def conversation_token(conversation_id: str, settings: Settings) -> str:
    """Return a deterministic capability token for one conversation."""
    secret = settings.conversation_signing_secret
    if secret is None or not secret.get_secret_value():
        raise ValueError("Conversation signing is not configured.")
    return hmac.new(
        secret.get_secret_value().encode("utf-8"),
        conversation_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def valid_conversation_token(
    conversation_id: str, supplied_token: str | None, settings: Settings
) -> bool:
    """Constant-time validate a conversation capability token."""
    if not supplied_token:
        return False
    expected = conversation_token(conversation_id, settings)
    return hmac.compare_digest(expected, supplied_token)
