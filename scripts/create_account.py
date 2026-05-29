"""Create or update a BRAIN account from the command line."""
import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from backend.models import Account, SessionLocal, init_db
from backend.security import encrypt_credential


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update a BRAIN account")
    parser.add_argument("--email", required=True, help="WorldQuant BRAIN email")
    parser.add_argument("--daily-quota", type=int, default=450, help="Daily submission quota")
    parser.add_argument("--password", default=None, help="BRAIN password. Omit to prompt securely.")
    parser.add_argument("--reactivate", action="store_true", help="Reactivate an existing soft-deleted account")
    args = parser.parse_args()

    password = args.password or getpass.getpass("BRAIN password: ")
    init_db()
    db = SessionLocal()
    try:
        account = db.query(Account).filter(Account.brain_email == args.email).first()
        if account is None:
            account = Account(
                brain_email=args.email,
                brain_password_encrypted=encrypt_credential(password),
                daily_quota=args.daily_quota,
                is_active=True,
            )
            db.add(account)
            action = "created"
        else:
            account.brain_password_encrypted = encrypt_credential(password)
            account.daily_quota = args.daily_quota
            if args.reactivate:
                account.is_active = True
            action = "updated"

        db.commit()
        db.refresh(account)
        print(f"Account {action}: id={account.id}, email={account.brain_email}, quota={account.daily_quota}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
