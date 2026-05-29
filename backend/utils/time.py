"""Time helpers."""
from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return naive UTC datetime for existing SQLAlchemy DateTime columns."""
    return datetime.now(UTC).replace(tzinfo=None)
