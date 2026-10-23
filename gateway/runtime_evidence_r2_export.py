"""Explicit, read-only, budgeted export of R2 settle markers to a local JSONL snapshot.

Run by hand only; the Gateway never calls it. Each ListObjectsV2 page is reserved as
one Class A operation and each marker GET as one Class B operation before the request
is sent. Every exported line is a validated settle marker (metadata only); the
Evaluator transport report reads the snapshot offline to compute ``emitted``.

    python -m gateway.runtime_evidence_r2_export --bucket B --account-id A \
        --budget-path LEDGER --output SNAPSHOT.jsonl --max-markers N
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from gateway.runtime_evidence_r2_settle import SETTLE_KEY_PREFIX, validate_settle_marker
from gateway.runtime_evidence_r2_transport import R2TransportError

MAX_EXPORT_MARKERS = 100_000


class R2ExportError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def export_settle_markers(*, transport: Any, budget: Any, bucket: str, max_markers: int) -> list[dict]:
    """Return validated markers; fail closed on any invalid object or a truncated listing."""
    if type(max_markers) is not int or not 1 <= max_markers <= MAX_EXPORT_MARKERS:
        raise R2ExportError("export_limit_invalid")
    keys: list[str] = []
    token: str | None = None
    while True:
        page, token = budget.execute(
            "A", lambda t=token: transport.list_keys_once(bucket, SETTLE_KEY_PREFIX, t)
        )
        keys.extend(page)
        if len(keys) > max_markers:
            raise R2ExportError("export_limit_exceeded")
        if token is None:
            break
    markers = []
    for key in keys:
        body = budget.execute("B", lambda k=key: transport.get_once(bucket, k))
        if type(body) is not bytes:
            raise R2ExportError("export_object_missing")
        try:
            marker = json.loads(body.decode("utf-8"))
        except (UnicodeError, ValueError):
            raise R2ExportError("export_object_invalid") from None
        if validate_settle_marker(marker):
            raise R2ExportError("export_object_invalid")
        markers.append(marker)
    return markers


def write_snapshot(markers: list[dict], output: Path) -> None:
    """Create-only, mode 0600; never overwrite an existing snapshot."""
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for marker in sorted(markers, key=lambda m: m["event_ref"]):
            handle.write(json.dumps(marker, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--budget-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-markers", required=True, type=int)
    parser.add_argument("--keychain-label", default="R2Key")
    args = parser.parse_args(argv)

    from gateway.runtime_evidence_r2_budget import R2BudgetError, R2UsageBudget
    from gateway.runtime_evidence_r2_s3 import (
        Boto3R2ObjectTransport,
        R2S3ConfigurationError,
        create_r2_s3_client,
        load_macos_keychain_credentials,
    )

    try:
        transport = Boto3R2ObjectTransport(
            create_r2_s3_client(args.account_id, load_macos_keychain_credentials(args.keychain_label))
        )
        markers = export_settle_markers(
            transport=transport, budget=R2UsageBudget(Path(args.budget_path).expanduser()),
            bucket=args.bucket, max_markers=args.max_markers,
        )
        write_snapshot(markers, Path(args.output).expanduser())
    except (R2ExportError, R2TransportError, R2BudgetError, R2S3ConfigurationError) as exc:
        print(json.dumps({"status": "failed", "code": exc.code}))
        return 1
    except FileExistsError:
        print(json.dumps({"status": "failed", "code": "export_output_exists"}))
        return 1
    print(json.dumps({"status": "exported", "markers": len(markers)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
