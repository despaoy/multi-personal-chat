"""Database-domain exceptions shared by storage adapters and API services."""


class RegistrationClosedError(RuntimeError):
    """Raised when bootstrap-only registration races with an existing user."""
