import json
import logging
from typing import Any, Dict, Optional

import pandas as pd

from app.database.mongodb import db_client
from app.utils.crypto import decrypt_secret, encrypt_secret

logger = logging.getLogger("regression_studio.integrations.store")

COLLECTION = "user_integrations"


async def save_connection(
    user_id: str,
    provider: str,
    credentials: Dict[str, Any],
    provider_user_id: Optional[str] = None,
) -> None:
    """Encrypts `credentials` (e.g. {"username": ..., "key": ...}) and
    upserts it against (user_id, provider). Enforced as a find-then-update
    upsert at the application layer — db_client has no native unique-index
    support — so one user can never end up with two connections for the
    same provider."""
    now = pd.Timestamp.now().isoformat()
    encrypted = encrypt_secret(json.dumps(credentials))

    existing = await db_client.find_one(COLLECTION, {"user_id": user_id, "provider": provider})
    if existing:
        await db_client.update_one(
            COLLECTION,
            {"_id": existing["_id"]},
            {"$set": {
                "provider_user_id": provider_user_id,
                "encrypted_credentials": encrypted,
                "updated_at": now,
            }},
        )
    else:
        await db_client.insert_one(COLLECTION, {
            "user_id": user_id,
            "provider": provider,
            "provider_user_id": provider_user_id,
            "encrypted_credentials": encrypted,
            "created_at": now,
            "updated_at": now,
        })


async def get_connection(user_id: str, provider: str) -> Optional[Dict[str, Any]]:
    """Returns {"credentials": {...decrypted...}, "provider_user_id": ...,
    "created_at": ..., "updated_at": ...} or None if not connected."""
    doc = await db_client.find_one(COLLECTION, {"user_id": user_id, "provider": provider})
    if not doc:
        return None
    try:
        credentials = json.loads(decrypt_secret(doc["encrypted_credentials"]))
    except Exception as e:
        logger.error(f"Failed to decrypt {provider} credentials for user {user_id}: {e}")
        return None
    return {
        "credentials": credentials,
        "provider_user_id": doc.get("provider_user_id"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


async def is_connected(user_id: str, provider: str) -> bool:
    doc = await db_client.find_one(COLLECTION, {"user_id": user_id, "provider": provider})
    return doc is not None


async def delete_connection(user_id: str, provider: str) -> bool:
    return await db_client.delete_one(COLLECTION, {"user_id": user_id, "provider": provider})
