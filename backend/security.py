"""
Security module: AES-256 encryption for storing BRAIN credentials.
"""
import base64
from typing import Optional
from cryptography.fernet import Fernet

from .config import settings

# Cache the cipher instance
_cipher_instance: Optional[Fernet] = None


def generate_aes_key() -> str:
    """
    Generate a new AES-256 key encoded in base64.
    Use this once and store in .env as AES_KEY.
    """
    key = Fernet.generate_key()
    return base64.b64encode(key).decode()


def get_cipher() -> Fernet:
    """Get Fernet cipher instance from AES_KEY environment variable (cached)."""
    global _cipher_instance
    
    if _cipher_instance is not None:
        return _cipher_instance
    
    if not settings.aes_key:
        # If no key provided, generate one (warning: not persisted, regenerated each run)
        key = Fernet.generate_key()
    else:
        try:
            key = base64.b64decode(settings.aes_key)
        except Exception:
            raise ValueError("Invalid AES_KEY: must be base64-encoded Fernet key")
    
    _cipher_instance = Fernet(key)
    return _cipher_instance


def encrypt_credential(plaintext: str) -> str:
    """
    Encrypt a credential (password, token) using AES-256.
    Returns base64-encoded ciphertext.
    """
    cipher = get_cipher()
    encrypted = cipher.encrypt(plaintext.encode())
    return base64.b64encode(encrypted).decode()


def decrypt_credential(ciphertext: str) -> str:
    """
    Decrypt a credential (password, token) using AES-256.
    Expects base64-encoded ciphertext.
    """
    cipher = get_cipher()
    try:
        encrypted = base64.b64decode(ciphertext)
        plaintext = cipher.decrypt(encrypted)
        return plaintext.decode()
    except Exception as e:
        raise ValueError(f"Failed to decrypt credential: {e}")


# One-time setup: generate key if needed
if __name__ == "__main__":
    new_key = generate_aes_key()
    print(f"Add this to .env as AES_KEY:\nAES_KEY={new_key}")
