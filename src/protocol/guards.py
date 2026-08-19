"""Stage gates that prevent accidental official-test exposure."""
from __future__ import annotations

from pathlib import Path

from src.protocol.contracts import read_json


class OfficialTestAccessError(RuntimeError):
    pass


def assert_official_test_access(*, stage: str, protocol_path: Path) -> None:
    if stage != "C7":
        raise OfficialTestAccessError(
            f"Official NIH test is blocked during {stage}; it may only be opened by C7."
        )
    protocol = read_json(protocol_path)
    if protocol.get("status") != "FROZEN":
        raise OfficialTestAccessError("Official test requires a FROZEN protocol artifact")
    if not protocol.get("protocol_hash"):
        raise OfficialTestAccessError("Frozen protocol artifact has no protocol_hash")
