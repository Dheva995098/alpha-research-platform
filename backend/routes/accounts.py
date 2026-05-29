"""
Basic API routes for account management.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict

from ..core.expression_normalizer import clean_brain_error_message
from ..core.data_fields import get_data_fields
from ..core.dataset_catalog import get_dataset_profile, list_dataset_profiles
from ..core.field_intelligence import upsert_field_records
from ..config import settings
from ..models import Account, Simulation, get_db
from ..security import encrypt_credential, decrypt_credential

router = APIRouter()


class AccountCreate(BaseModel):
    """Account creation request."""
    brain_email: str
    brain_password: str


class AccountResponse(BaseModel):
    """Account response (no password)."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    brain_email: str
    daily_quota: int
    submissions_today: int
    is_active: bool
    worker_enabled: bool
    max_running: int
    max_pending: int
    cooldown_until: Optional[datetime] = None
    last_worker_error: Optional[str] = None


class AccountUpdate(BaseModel):
    """Account update request."""
    daily_quota: Optional[int] = None
    submissions_today: Optional[int] = None
    is_active: Optional[bool] = None
    worker_enabled: Optional[bool] = None
    max_running: Optional[int] = None
    max_pending: Optional[int] = None


@router.post("/", response_model=AccountResponse, status_code=status.HTTP_201_CREATED, tags=["accounts"])
def create_account(
    account_data: AccountCreate,
    db: Session = Depends(get_db)
) -> Account:
    """
    Create a new BRAIN account.
    Credentials are encrypted before storage.
    """
    # Check if account already exists
    existing = db.query(Account).filter(
        Account.brain_email == account_data.brain_email
    ).first()

    encrypted_password = encrypt_credential(account_data.brain_password)
    if settings.single_account_mode:
        for account in db.query(Account).all():
            account.is_active = False
            account.worker_enabled = False

    if existing and existing.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Account with email {account_data.brain_email} already exists"
        )

    if existing:
        existing.brain_password_encrypted = encrypted_password
        existing.is_active = True
        existing.worker_enabled = True
        existing.cooldown_until = None
        existing.last_worker_error = None
        existing.submissions_today = 0
        db.commit()
        db.refresh(existing)
        return existing

    # Create account
    account = Account(
        brain_email=account_data.brain_email,
        brain_password_encrypted=encrypted_password,
        worker_enabled=True,
    )
    
    db.add(account)
    db.commit()
    db.refresh(account)
    
    return account


@router.get("/", response_model=List[AccountResponse], tags=["accounts"])
def list_accounts(
    db: Session = Depends(get_db)
) -> List[Account]:
    """List active BRAIN accounts."""
    query = db.query(Account).filter(Account.is_active == True)
    if settings.single_account_mode:
        primary_id = settings.primary_account_id
        if primary_id and query.filter(Account.id == int(primary_id)).first():
            query = query.filter(Account.id == int(primary_id))
        else:
            first = query.order_by(Account.id.asc()).first()
            return [first] if first else []
    return query.order_by(Account.id.asc()).all()


@router.get("/{account_id}", response_model=AccountResponse, tags=["accounts"])
def get_account(
    account_id: int,
    db: Session = Depends(get_db)
) -> Account:
    """Get account details by ID."""
    account = db.query(Account).filter(Account.id == account_id).first()
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found"
        )
    return account


@router.put("/{account_id}", response_model=AccountResponse, tags=["accounts"])
def update_account(
    account_id: int,
    account_data: AccountUpdate,
    db: Session = Depends(get_db)
) -> Account:
    """Update account settings."""
    account = db.query(Account).filter(Account.id == account_id).first()
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found"
        )
    if settings.single_account_mode and settings.primary_account_id and account.id != int(settings.primary_account_id):
        account.is_active = False
        account.worker_enabled = False
        db.commit()
        db.refresh(account)
        return account
    
    if account_data.daily_quota is not None:
        account.daily_quota = account_data.daily_quota
    if account_data.submissions_today is not None:
        account.submissions_today = max(account_data.submissions_today, 0)
    if account_data.is_active is not None:
        account.is_active = account_data.is_active
        if account_data.is_active:
            account.cooldown_until = None
            account.last_worker_error = None
    if account_data.worker_enabled is not None:
        account.worker_enabled = account_data.worker_enabled
        if account_data.worker_enabled:
            account.cooldown_until = None
            account.last_worker_error = None
    if account_data.max_running is not None:
        account.max_running = max(1, min(account_data.max_running, 30))
    if account_data.max_pending is not None:
        account.max_pending = max(0, min(account_data.max_pending, 1000))
    
    db.commit()
    db.refresh(account)
    return account


@router.post("/{account_id}/quota/reset", response_model=AccountResponse, tags=["accounts"])
def reset_account_quota(
    account_id: int,
    db: Session = Depends(get_db)
) -> Account:
    """Reset today's local quota counter for an account."""
    account = db.query(Account).filter(Account.id == account_id).first()
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found"
        )
    if settings.single_account_mode and settings.primary_account_id and account.id != int(settings.primary_account_id):
        account.is_active = False
        account.worker_enabled = False
        db.commit()
        db.refresh(account)
        return account

    account.submissions_today = 0
    db.commit()
    db.refresh(account)
    return account


@router.delete("/{account_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["accounts"])
def delete_account(
    account_id: int,
    db: Session = Depends(get_db)
) -> None:
    """Deactivate account (soft delete)."""
    account = db.query(Account).filter(Account.id == account_id).first()
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found"
        )
    
    account.is_active = False
    account.worker_enabled = False
    account.cooldown_until = None
    account.last_worker_error = None
    (
        db.query(Simulation)
        .filter(Simulation.account_id == account.id)
        .filter(Simulation.status.in_(["pending", "submitting", "running"]))
        .update(
            {
                "status": "cancelled",
                "error_message": "Account removed from active worker lanes",
            },
            synchronize_session=False,
        )
    )
    db.commit()


@router.post("/{account_id}/test", tags=["accounts"])
def test_account(
    account_id: int,
    db: Session = Depends(get_db)
) -> dict:
    """
    Test BRAIN API connection for account.
    Returns connection status.
    """
    account = db.query(Account).filter(Account.id == account_id).first()
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found"
        )
    
    # Decrypt credentials
    try:
        decrypted_password = decrypt_credential(account.brain_password_encrypted)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decrypt credentials: {str(e)}"
        )
    
    # Test connection
    try:
        from ..core.brain_api import BRAINClient
        client = BRAINClient(account.brain_email, decrypted_password)
        client.close()
        account.cooldown_until = None
        account.last_worker_error = None
        account.worker_enabled = True
        db.commit()
        return {"status": "success", "message": "BRAIN API connection successful"}
    except Exception as e:
        message = clean_brain_error_message(str(e)) or "BRAIN API connection failed"
        account.last_worker_error = message
        db.commit()
        return {"status": "failed", "message": f"BRAIN API connection failed: {message}"}


@router.post("/{account_id}/sync-fields", tags=["accounts"])
def sync_account_fields(
    account_id: int,
    dataset_id: Optional[str] = Query(default=None),
    limit_per_dataset: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> dict:
    """Fetch live BRAIN fields for known datasets and add them to the local schema."""
    account = db.query(Account).filter(Account.id == account_id).first()
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found",
        )

    try:
        decrypted_password = decrypt_credential(account.brain_password_encrypted)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decrypt credentials: {str(e)}",
        )

    if dataset_id:
        profile = get_dataset_profile(dataset_id)
        if profile is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found in local catalog",
            )
        profiles = [profile]
    else:
        profiles = list_dataset_profiles()

    from ..core.brain_api import BRAINClient

    schema = get_data_fields()
    dataset_stats = []
    unique_fields = set()
    try:
        client = BRAINClient(account.brain_email, decrypted_password)
        try:
            for profile in profiles:
                fields = client.session.get_data_fields(
                    dataset_id=profile.id,
                    universe=profile.default_universe,
                    limit=limit_per_dataset,
                ) or []
                schema.add_fields_from_api(fields)
                import_stats = upsert_field_records(
                    db,
                    fields,
                    dataset_id=profile.id,
                    universe=profile.default_universe,
                )
                names = [
                    str(field.get("name") or field.get("id") or "").strip().lower()
                    for field in fields
                    if isinstance(field, dict) and (field.get("name") or field.get("id"))
                ]
                unique_fields.update(name for name in names if name)
                dataset_stats.append(
                    {
                        "dataset_id": profile.id,
                        "universe": profile.default_universe,
                        "fetched": len(names),
                        **import_stats,
                    }
                )
        finally:
            client.close()
    except Exception as e:
        return {"status": "failed", "message": f"BRAIN field sync failed: {str(e)}", "datasets": dataset_stats}

    return {
        "status": "success",
        "message": f"Synced {len(unique_fields)} live field(s)",
        "field_count": len(unique_fields),
        "dataset_count": len(dataset_stats),
        "total_known_fields": len(schema.fields),
        "datasets": dataset_stats,
    }
