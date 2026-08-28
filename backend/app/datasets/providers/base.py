from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

SourceType = Literal["local", "google_drive", "kaggle"]


@dataclass
class DatasetSource:
    """The common handoff object every provider produces: a file already
    staged on local disk (downloaded or uploaded), ready for pandas to read.
    The ML pipeline (ingest_source, and everything downstream of it) only
    ever touches this shape — it never needs to know which provider produced
    it, satisfying the "ML engine must not care where the dataset originated"
    requirement."""

    source_type: SourceType
    file_name: str
    file_path: str
    file_size: int
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetSourceProvider(ABC):
    """Base class for a dataset source. Providers are intentionally NOT
    forced into one polymorphic fetch() signature — local/Kaggle/Google
    Drive each retrieve data in a fundamentally different shape (wrap
    already-uploaded bytes vs. resolve-then-download-by-ref vs.
    OAuth-list-then-download-by-id). What they share is is_configured() and
    that they all ultimately hand a DatasetSource to ingest_source()."""

    provider_id: str

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider has everything it needs (API keys, OAuth
        client config, etc.) to be offered to users right now. Local file
        upload is always configured; Kaggle/Google Drive depend on env vars
        the deployer may not have set."""
        raise NotImplementedError
