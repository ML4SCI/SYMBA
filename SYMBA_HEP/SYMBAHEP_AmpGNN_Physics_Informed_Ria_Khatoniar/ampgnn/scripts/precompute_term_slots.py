#!/usr/bin/env python3
"""Build a versioned SQLite cache of structured term-slot targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _package_parent() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_package_parent()))

from ampgnn.expr_simplify import coupling_for_model  # noqa: E402
from ampgnn.term_slots import (  # noqa: E402
    TERM_SLOT_CACHE_VERSION,
    build_term_slot_target,
    term_slot_cache_key,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, object]]]:
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "target_tokens" not in record or "model" not in record:
                    raise ValueError(
                        f"{path}:{line_number} lacks model or target_tokens"
                    )
                yield path, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    inputs = [path.resolve() for path in args.inputs]
    for path in inputs:
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(output)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS term_slot_cache "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )

    rows = 0
    inserted = 0
    per_file: Dict[str, int] = {}
    for source, record in _records(inputs):
        rows += 1
        per_file[str(source)] = per_file.get(str(source), 0) + 1
        tokens = [str(token) for token in record["target_tokens"]]
        coupling = coupling_for_model(str(record["model"]))
        key = term_slot_cache_key(tokens, coupling)
        if connection.execute(
            "SELECT 1 FROM term_slot_cache WHERE key = ?", (key,)
        ).fetchone() is not None:
            continue
        target = build_term_slot_target(tokens, coupling, use_cache=False)
        connection.execute(
            "INSERT INTO term_slot_cache (key, value) VALUES (?, ?)",
            (key, json.dumps(target, sort_keys=True, separators=(",", ":"))),
        )
        inserted += 1
        if inserted % 100 == 0:
            connection.commit()
            print(f"[term-slot-cache] {inserted:,} unique targets", flush=True)
    connection.commit()
    unique = int(
        connection.execute("SELECT COUNT(*) FROM term_slot_cache").fetchone()[0]
    )
    connection.close()

    summary = {
        "cache_version": TERM_SLOT_CACHE_VERSION,
        "input_rows": rows,
        "inserted_this_run": inserted,
        "unique_cache_keys": unique,
        "inputs": {
            str(path): {"rows": per_file[str(path)], "sha256": _sha256(path)}
            for path in inputs
        },
        "cache": str(output),
        "cache_sha256": _sha256(output),
    }
    summary_path = args.summary or output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
