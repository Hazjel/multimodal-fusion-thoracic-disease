"""Stage gates that prevent accidental official-test exposure."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.protocol.contracts import file_sha256, read_json
from src.protocol.stages import validate_c7_prerequisites


class OfficialTestAccessError(RuntimeError):
    pass


def assert_official_test_access(
    *,
    stage: str,
    protocol_path: Path,
    access_event_id: Optional[str] = None,
) -> None:
    """Authorize only the claimed, resumable C7 secondary-holdout event."""
    if stage != "C7":
        raise OfficialTestAccessError(
            f"Official NIH test is blocked during {stage}; it may only be opened by C7."
        )
    protocol_path = Path(protocol_path)
    protocol_dir = protocol_path.parent
    try:
        state = validate_c7_prerequisites(protocol_dir, require_refit=True)
    except Exception as exc:
        raise OfficialTestAccessError(f"Official test C7 preflight failed: {exc}") from exc
    protocol = state["protocol"]
    receipt_path = protocol_dir / "secondary_holdout" / "official_test_access_receipt.json"
    if not receipt_path.is_file():
        raise OfficialTestAccessError("Official test access has not been explicitly claimed")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "CLAIMED":
        raise OfficialTestAccessError("Official test access receipt is not in CLAIMED status")
    if not access_event_id or receipt.get("access_event_id") != access_event_id:
        raise OfficialTestAccessError("Official test access event ID mismatch")
    if receipt.get("protocol_hash") != protocol.get("protocol_hash"):
        raise OfficialTestAccessError("Official test access receipt belongs to another protocol")
    if receipt.get("refit_index_hash") != file_sha256(state["refit_index_path"]):
        raise OfficialTestAccessError("Official test access receipt does not match deployment refits")
    if (protocol_dir / "secondary_holdout" / "_SUCCESS").is_file():
        raise OfficialTestAccessError(
            "Secondary holdout is already complete; repeated official-test access is blocked"
        )
