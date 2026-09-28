"""Errors isolated to observability instrumentation."""


class ObservabilityError(Exception):
    """Raised only for optional tracing failures."""