import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("regression_studio.crypto")

_ENV_VAR = "INTEGRATION_ENCRYPTION_KEY"


def _get_fernet() -> Fernet:
    key = os.getenv(_ENV_VAR)
    if not key:
        raise RuntimeError(
            f"{_ENV_VAR} is not set. Generate one with:\n"
            '  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"\n'
            "and add it to backend/.env before connecting a third-party account (e.g. Kaggle)."
        )
    try:
        return Fernet(key.encode())
    except Exception as e:
        raise RuntimeError(f"{_ENV_VAR} is not a valid Fernet key: {e}")


def encrypt_secret(plaintext_json: str) -> str:
    """Encrypts a small JSON blob (e.g. {"username": "...", "key": "..."})
    for storage in a user_integrations document. Never call this with
    anything that isn't already destined for an encrypted credentials field."""
    return _get_fernet().encrypt(plaintext_json.encode()).decode()


def decrypt_secret(token: str) -> str:
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except InvalidToken:
        raise RuntimeError(
            f"Failed to decrypt a stored credential — {_ENV_VAR} may have changed since it was saved."
        )


def warn_if_unconfigured() -> None:
    """Call once at app startup — mirrors the warning style already used for
    other security-sensitive env vars (see auth/security.py)."""
    if not os.getenv(_ENV_VAR):
        logger.warning(
            f"{_ENV_VAR} is not set. Connecting a third-party account (e.g. Kaggle) will fail "
            "until it's configured — see backend/.env.example."
        )
