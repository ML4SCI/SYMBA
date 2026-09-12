#!/usr/bin/env python3
"""Simplify tokenized squared amplitudes into factor * numerator/denominator * g^n.

Single example:
  sed -n '1p' data/QCD/QCD-2-to-2-diag-TreeLevel-0.txt | \\
    python -m ampgnn.scripts.simplify_squared_amplitude \\
      --dataset-line-file /dev/stdin --mass-rewrite-mode leg_sets

Batch analysis on QCD/QED:
  python -m ampgnn.scripts.simplify_squared_amplitude \\
    --data-dir ../data --models QCD,QED --analyze

Compare sympy simplification against dataset targets (July compact export):
  python -m ampgnn.scripts.simplify_squared_amplitude \\
    --data-dir ../data_july2026 --models QCD --orders tree \\
    --ranks 2_to_2,2_to_3 --compact-header-stub --canonicalize-targets \\
    --compare --compare-out ../simplify_comparison_qcd_july.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_ROOT = _REPO_ROOT.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ampgnn.data_loading import (
    PAIR_FIELD_RE,
    header_to_stub_diagram,
    load_lines_from_dir,
)
from ampgnn.diagram_parser import parse_diagram
from ampgnn.expr_simplify import (
    analyze_lines,
    compare_lines,
    decompose_token_list,
    extract_factor_g_num_den,
    format_factorized_tokens,
    parse_tokens_to_sympy,
)
from ampgnn.mass_rewrite import canonicalize_masses
from ampgnn.tokenizer import tokenize_expr

import sympy as sp


def split_dataset_line(
    line: str,
    *,
    compact_header_stub: bool = False,
) -> Tuple[str, str]:
    line = line.strip()
    if line.startswith("Interaction:"):
        line = line[len("Interaction:") :].strip()

    parts = line.split(" : ")
    vertex_idxs = [i for i, p in enumerate(parts) if p.lstrip().startswith("Vertex ")]
    if vertex_idxs:
        diagram = parts[vertex_idxs[0]].strip()
        squared_amplitude = parts[-1].strip()
        if not diagram or not squared_amplitude:
            raise ValueError("dataset line is missing diagram or squared amplitude")
        return diagram, squared_amplitude

    pair_idx = next(
        (i for i, p in enumerate(parts) if PAIR_FIELD_RE.match(p.strip())),
        None,
    )
    if pair_idx is not None:
        squared_amplitude = parts[-1].strip()
        if not squared_amplitude:
            raise ValueError("compact dataset line is missing squared-amplitude term")
        header = parts[0].strip()
        if compact_header_stub:
            diagram = header_to_stub_diagram(header)
            if not diagram:
                raise ValueError("could not build header stub diagram for compact row")
            return diagram, squared_amplitude
        raise ValueError(
            "compact row has no Vertex fields; pass --compact-header-stub or use "
            "Diagram N: sidecar lines"
        )

    raise ValueError("could not find diagram section beginning with 'Vertex '")


def tokens_from_text(text: str) -> List[str]:
    text = text.strip()
    if not text:
        raise ValueError("empty expression")
    if text.startswith("["):
        values = json.loads(text)
        if not isinstance(values, list):
            raise ValueError("JSON input must be a list of tokens")
        return [str(v) for v in values]
    return text.split()


def tokens_from_jsonl(path: Path, index: int, field: str) -> List[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if index < 1 or index > len(rows):
        raise IndexError(f"--index must be between 1 and {len(rows)}")
    tokens = rows[index - 1][field]
    if not isinstance(tokens, list):
        raise ValueError(f"{field!r} is not a token list")
    return [str(tok) for tok in tokens]


def tokens_from_dataset_line(
    line: str,
    mass_rewrite_mode: str,
    *,
    compact_header_stub: bool = False,
) -> tuple[List[str], dict[str, str]]:
    diagram, expr = split_dataset_line(
        line, compact_header_stub=compact_header_stub
    )
    bd = parse_diagram(diagram)
    rewritten, slot_map = canonicalize_masses(expr, bd, mode=mass_rewrite_mode)
    return tokenize_expr(rewritten), slot_map


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_dataset_lines(
    data_dir: Path,
    models: List[str],
    orders: Optional[List[str]],
    ranks: Optional[List[str]],
    compact_header_stub: bool,
):
    lines = []
    for model in models:
        folder = str(data_dir / model)
        lines.extend(
            load_lines_from_dir(
                folder,
                orders=orders,
                ranks=ranks,
                compact_header_stub=compact_header_stub,
            )
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Spaced token expression. If omitted, stdin is used.")
    ap.add_argument("--jsonl", type=Path, help="Read tokens from a failure JSONL file.")
    ap.add_argument("--index", type=int, default=1, help="1-based JSONL row index.")
    ap.add_argument("--field", choices=["target_text", "pred_text"], default="target_text")
    ap.add_argument("--dataset-line", help="Raw dataset line containing Interaction/diagram/amplitude fields.")
    ap.add_argument("--dataset-line-file", type=Path, help="File containing one raw dataset line.")
    ap.add_argument("--mass-rewrite-mode", choices=["species", "leg_sets"], default="leg_sets")
    ap.add_argument("--show-tokens", action="store_true", help="Print mass-rewritten tokens before simplifying.")
    ap.add_argument("--show-slot-map", action="store_true", help="Print the diagram-derived mass slot map.")
    ap.add_argument("--g-symbol", default=None, help="Coupling symbol to peel off (default: g for QCD, e for QED).")
    ap.add_argument("--data-dir", type=Path, help="Root data directory containing QCD/, QED/, ...")
    ap.add_argument("--models", default="QCD,QED", help="Comma-separated model folders to load.")
    ap.add_argument("--orders", default=None, help="Optional comma-separated orders filter (e.g. tree).")
    ap.add_argument("--ranks", default=None, help="Optional comma-separated ranks filter (e.g. 2_to_2,2_to_3).")
    ap.add_argument(
        "--compact-header-stub",
        action="store_true",
        help="Load compact 5-field pair rows using process-header stub diagrams.",
    )
    ap.add_argument("--analyze", action="store_true", help="Analyze token-length reduction and unique parts.")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Compare sympy-simplified tokens against dataset targets; write JSONL.",
    )
    ap.add_argument(
        "--compare-out",
        type=Path,
        default=Path("simplify_comparison.jsonl"),
        help="Output JSONL path for --compare (default: simplify_comparison.jsonl).",
    )
    ap.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Process at most N dataset rows (for quick smoke tests).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print compare progress every N rows (0 to disable).",
    )
    ap.add_argument(
        "--simplify-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SymPy-simplify targets before compare/analyze (matches train --simplify_targets).",
    )
    ap.add_argument(
        "--canonicalize-targets",
        action="store_true",
        help="Apply commutative canonicalization before simplify/compare.",
    )
    args = ap.parse_args()

    if args.analyze or args.compare:
        if args.data_dir is None:
            raise SystemExit("--analyze/--compare requires --data-dir")
        models = _parse_csv(args.models)
        orders = _parse_csv(args.orders) if args.orders else None
        ranks = _parse_csv(args.ranks) if args.ranks else None
        lines = _load_dataset_lines(
            args.data_dir,
            models,
            orders,
            ranks,
            args.compact_header_stub,
        )
        if not lines:
            raise SystemExit(f"no lines loaded from {args.data_dir} for models={models}")
        if args.compare:
            print(f"[compare] processing up to {len(lines)} rows (sympy per row — slow on CPU)...", flush=True)
            rows, summary = compare_lines(
                lines,
                mass_rewrite_mode=args.mass_rewrite_mode,
                canonicalize_targets=args.canonicalize_targets,
                simplify_targets=args.simplify_targets,
                max_lines=args.max_lines,
                progress_every=args.progress_every,
            )
            args.compare_out.write_text(
                "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else "")
            )
            print(json.dumps(summary, indent=2))
            print(f"wrote {len(rows)} rows to {args.compare_out}")
            return
        report = analyze_lines(
            lines,
            mass_rewrite_mode=args.mass_rewrite_mode,
            canonicalize_targets=args.canonicalize_targets,
            simplify_targets=args.simplify_targets,
        )
        print(json.dumps(report, indent=2))
        return

    slot_map = None
    if args.dataset_line or args.dataset_line_file:
        if args.dataset_line and args.dataset_line_file:
            raise SystemExit("use only one of --dataset-line or --dataset-line-file")
        line = args.dataset_line if args.dataset_line else args.dataset_line_file.read_text().strip()
        tokens, slot_map = tokens_from_dataset_line(
            line,
            args.mass_rewrite_mode,
            compact_header_stub=args.compact_header_stub,
        )
    elif args.jsonl:
        tokens = tokens_from_jsonl(args.jsonl, args.index, args.field)
    else:
        text = args.text if args.text is not None else sys.stdin.read()
        tokens = tokens_from_text(text)

    g_symbol = args.g_symbol or "g"

    if args.show_slot_map and slot_map is not None:
        print("slot_map:")
        for key in sorted(slot_map):
            print(f"  {key} -> {slot_map[key]}")
        print()
    if args.show_tokens:
        print("tokens:")
        print(" ".join(tokens))
        print(f"len={len(tokens)}")
        print()

    expr = parse_tokens_to_sympy(tokens)
    simplified = sp.factor(sp.together(expr))
    factor, g_power, numerator, denominator = extract_factor_g_num_den(simplified, g_symbol)
    out_tokens = format_factorized_tokens(factor, g_power, numerator, denominator, g_symbol)
    fac_t, gp, num_t, den_t = decompose_token_list(tokens, g_symbol=g_symbol)

    print(f"factor      = {factor}")
    print(f"g_power     = {g_power}")
    print(f"numerator   = {numerator}")
    print(f"denominator = {denominator}")
    print()
    print("form:")
    print(f"({factor}) * {g_symbol}^{g_power} * ({numerator}) / ({denominator})")
    print()
    print("tokens_out:")
    print(" ".join(out_tokens))
    print(f"len_in={len(tokens)} len_out={len(out_tokens)}")
    print()
    print("parts_as_tokens:")
    print(f"  factor      ({len(fac_t)}): {' '.join(fac_t) or '1'}")
    print(f"  g_power     : {gp}")
    print(f"  numerator   ({len(num_t)}): {' '.join(num_t) or '1'}")
    print(f"  denominator ({len(den_t)}): {' '.join(den_t) or '1'}")


if __name__ == "__main__":
    main()
