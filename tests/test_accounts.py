"""Tests for account lifecycle behavior."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import Account, Base
from backend.routes.accounts import AccountCreate, create_account, delete_account, list_accounts
from backend.security import decrypt_credential, encrypt_credential


def make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory()


def test_create_account_reactivates_soft_deleted_email_with_new_password():
    db = make_db()
    account = Account(
        brain_email="second@example.com",
        brain_password_encrypted=encrypt_credential("old-secret"),
        is_active=False,
        worker_enabled=False,
        cooldown_until=None,
        last_worker_error="BRAIN authentication failed",
        submissions_today=17,
    )
    db.add(account)
    db.commit()

    reactivated = create_account(
        AccountCreate(brain_email="second@example.com", brain_password="new-secret"),
        db=db,
    )

    assert reactivated.id == account.id
    assert reactivated.is_active is True
    assert reactivated.worker_enabled is True
    assert reactivated.last_worker_error is None
    assert reactivated.submissions_today == 0
    assert decrypt_credential(reactivated.brain_password_encrypted) == "new-secret"
    assert len(list_accounts(db=db)) == 1


def test_delete_account_soft_deactivates_worker_lane():
    db = make_db()
    account = create_account(AccountCreate(brain_email="second@example.com", brain_password="secret"), db=db)

    delete_account(account.id, db=db)
    db.refresh(account)

    assert account.is_active is False
    assert account.worker_enabled is False
    assert list_accounts(db=db) == []
