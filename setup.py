"""
Setup script: Generate AES key and initialize .env file.
Run this once during first setup.
"""
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.security import generate_aes_key
from backend.config import settings


def setup():
    """Initialize .env and generate AES key."""
    print("=" * 60)
    print("Alpha Research Platform - Initial Setup")
    print("=" * 60)
    
    env_file = Path(__file__).parent / ".env"
    env_example = Path(__file__).parent / ".env.example"
    
    # Check if .env exists
    if env_file.exists():
        print("\n✓ .env file already exists")
    else:
        print("\n✗ .env file not found. Creating from .env.example...")
        if env_example.exists():
            with open(env_example, "r") as f:
                content = f.read()
            with open(env_file, "w") as f:
                f.write(content)
            print(f"✓ Created .env from template")
        else:
            print("✗ .env.example not found!")
            return False
    
    # Generate AES key
    print("\nGenerating AES-256 encryption key...")
    aes_key = generate_aes_key()
    print(f"✓ Generated AES key")
    
    # Update .env with AES_KEY
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("AES_KEY="):
            lines[i] = f"AES_KEY={aes_key}\n"
            updated = True
            break
    
    if not updated:
        lines.append(f"AES_KEY={aes_key}\n")
    
    with open(env_file, "w") as f:
        f.writelines(lines)
    
    print(f"✓ Updated .env with AES_KEY")
    
    # Database
    print("\nInitializing database...")
    from backend.models import init_db, engine
    init_db()
    print("✓ Database initialized")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env and add your settings:")
    print("   - API_HOST, API_PORT")
    print("   - OPENAI_API_KEY or CLAUDE_API_KEY")
    print("   - SLACK_WEBHOOK_URL (optional)")
    print("   - SENDGRID_API_KEY (optional)")
    print("\n2. Run backend: python -m uvicorn backend.main:app --reload")
    print("3. Run frontend: cd frontend && npm install && npm start")
    print("4. Access dashboard: http://localhost:3000")
    
    return True


if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)
