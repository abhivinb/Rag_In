"""Errors raised by the database layer."""


class DatabaseError(Exception):
    """Base class for database failures."""


class DatabaseConfigurationError(DatabaseError):
    """Raised when database and embedding configuration cannot agree."""


class DatabaseConnectionError(DatabaseError):
    """Raised when a database connection or health check fails."""


class PersistenceError(DatabaseError):
    """Raised when a transaction cannot persist a document and its chunks."""