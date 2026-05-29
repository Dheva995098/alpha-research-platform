"""Quota helpers for BRAIN account orchestration."""
from dataclasses import dataclass
from datetime import datetime

from backend.models import Account
from backend.utils.time import utc_now


@dataclass(frozen=True)
class AccountQuota:
    """Current quota state for an account."""

    account_id: int
    daily_quota: int
    submissions_today: int
    remaining: int
    is_active: bool

    @property
    def has_capacity(self) -> bool:
        return self.is_active and self.remaining > 0


def reset_daily_quota_if_needed(account: Account, now: datetime | None = None) -> bool:
    """Reset an account's daily counter once the UTC day changes."""
    now = now or utc_now()
    last_reset = account.last_quota_reset

    if last_reset is None or last_reset.date() < now.date():
        account.submissions_today = 0
        account.last_quota_reset = now
        return True

    return False


def quota_for_account(account: Account) -> AccountQuota:
    """Return quota summary for one account."""
    daily_quota = account.daily_quota or 0
    submissions_today = account.submissions_today or 0
    return AccountQuota(
        account_id=account.id,
        daily_quota=daily_quota,
        submissions_today=submissions_today,
        remaining=max(daily_quota - submissions_today, 0),
        is_active=bool(account.is_active),
    )
