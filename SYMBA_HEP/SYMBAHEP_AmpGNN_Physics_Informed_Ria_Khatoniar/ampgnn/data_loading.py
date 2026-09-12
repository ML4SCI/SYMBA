import os
import re
from typing import Dict, List, Optional, Tuple

RANK_RE = re.compile(r"^(\d+)_to_(\d+)$")
FNAME_RE = re.compile(
    r"^(?P<model>[A-Za-z]+)-(?P<a>\d+)-to-(?P<b>\d+)-diag-"
    r"(?P<order>TreeLevel|\d+Loop)-\d+(?:[A-Za-z_][A-Za-z0-9_\-]*)?\.txt$"
)

BAD_SNIPPETS = ("Error evaluating", "The string is too long", "Error on pair")

PAIR_FIELD_RE = re.compile(r"^pair\s+(\d+)-(\d+)$")
DIAGRAM_LINE_RE = re.compile(r"^Diagram\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
HEADER_EXT_RE = re.compile(
    r"(AntiPart\s+)?([A-Za-z]+)_\{[^}]+\}\(X\)(\^\(\*\))?"
)


def meta_from_path(path: str, model_root: str) -> Optional[Dict[str, object]]:
    rel = os.path.relpath(path, model_root)
    parts = rel.split(os.sep)
    model_name = os.path.basename(os.path.normpath(model_root))

    if len(parts) >= 3:
        order, rank = parts[-3], parts[-2]
        m = RANK_RE.match(rank)
        if m:
            return {
                "model": model_name,
                "order": order,
                "rank": rank,
                "n_in": int(m.group(1)),
                "n_out": int(m.group(2)),
            }

    fm = FNAME_RE.match(os.path.basename(path))
    if fm is None:
        return None
    order_raw = fm.group("order")
    if order_raw == "TreeLevel":
        order = "tree"
    else:
        lm = re.match(r"^(\d+)Loop$", order_raw)
        if not lm:
            return None
        order = f"{int(lm.group(1))}_loop"
    a, b = int(fm.group("a")), int(fm.group("b"))
    return {
        "model": fm.group("model"),
        "order": order,
        "rank": f"{a}_to_{b}",
        "n_in": a,
        "n_out": b,
    }


def header_to_stub_diagram(header: str) -> Optional[str]:
    """Build a single-vertex diagram string from a process header."""
    header = header.strip()
    if " to " not in header:
        return None
    in_part, out_part = header.split(" to ", 1)
    attachments: List[str] = []
    xid = 1
    for side_text in (in_part, out_part):
        for m in HEADER_EXT_RE.finditer(side_text):
            antip = "AntiPart " if m.group(1) else ""
            particle = m.group(2)
            attachments.append(f"{antip}{particle}(X_{xid})")
            xid += 1
    if not attachments:
        return None
    return "Vertex V_0:" + ", ".join(attachments)


def _diagram_sidecar_paths(path: str) -> List[str]:
    base, _ = os.path.splitext(path)
    dirname, stem = os.path.split(path)
    candidates = [
        f"{base}-diagrams.txt",
        os.path.join(dirname, "diagrams", f"{stem}.txt"),
    ]
    return [p for p in candidates if os.path.isfile(p)]


def _load_diagram_table(path: str, raw_lines: List[str]) -> Dict[int, str]:
    table: Dict[int, str] = {}
    for sidecar in _diagram_sidecar_paths(path):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                raw_lines.extend(f.readlines())
        except OSError:
            continue

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        m = DIAGRAM_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        diagram = m.group(2).strip()
        if diagram:
            table[idx] = diagram
    return table


def _pair_meta(meta: Dict[str, object], parts: List[str]) -> Dict[str, object]:
    for p in parts:
        m = PAIR_FIELD_RE.match(p.strip())
        if m:
            meta_line = dict(meta)
            meta_line["pair"] = (int(m.group(1)), int(m.group(2)))
            return meta_line
    return meta


def _parse_vertex_row(
    parts: List[str],
    meta: Dict[str, object],
) -> Optional[Tuple[Tuple[str, Optional[str]], str, Dict[str, object]]]:
    vertex_idxs = [i for i, p in enumerate(parts) if p.lstrip().startswith("Vertex ")]
    if not vertex_idxs:
        return None

    diagram_i = parts[vertex_idxs[0]].strip()
    diagram_j = parts[vertex_idxs[1]].strip() if len(vertex_idxs) >= 2 else None
    expr = parts[-1].strip()
    if not diagram_i or not expr:
        return None
    return ((diagram_i, diagram_j), expr, _pair_meta(meta, parts))


def _parse_compact_row(
    parts: List[str],
    meta: Dict[str, object],
    diagram_table: Dict[int, str],
    header_stub: Optional[str],
) -> Optional[Tuple[Tuple[str, Optional[str]], str, Dict[str, object]]]:
    pair_idx = next(
        (i for i, p in enumerate(parts) if PAIR_FIELD_RE.match(p.strip())),
        None,
    )
    if pair_idx is None or len(parts) < pair_idx + 2:
        return None

    pair_m = PAIR_FIELD_RE.match(parts[pair_idx].strip())
    if pair_m is None:
        return None
    i, j = int(pair_m.group(1)), int(pair_m.group(2))
    expr = parts[-1].strip()
    if not expr:
        return None

    header = parts[0].strip() if pair_idx > 0 else ""
    fallback = header_stub or (header_to_stub_diagram(header) if header else None)

    diagram_i = diagram_table.get(i) or fallback
    diagram_j = diagram_table.get(j) or fallback
    if not diagram_i or not diagram_j:
        return None
    if i == j:
        diagram_j = diagram_i

    return ((diagram_i, diagram_j), expr, _pair_meta(meta, parts))


def _parse_interaction_line(
    line: str,
    meta: Dict[str, object],
    diagram_table: Dict[int, str],
    header_stub: Optional[str],
) -> Optional[Tuple[Tuple[str, Optional[str]], str, Dict[str, object]]]:
    if line.startswith("Interaction:"):
        line = line[len("Interaction:") :].strip()
    if " : " not in line:
        return None

    parts = line.split(" : ")
    if any(p.lstrip().startswith("Vertex ") for p in parts):
        return _parse_vertex_row(parts, meta)

    pair_idx = next(
        (i for i, p in enumerate(parts) if PAIR_FIELD_RE.match(p.strip())),
        None,
    )
    if pair_idx is not None:
        return _parse_compact_row(parts, meta, diagram_table, header_stub)

    return None


def load_lines_from_dir(
    folder: str,
    orders: Optional[List[str]] = None,
    ranks: Optional[List[str]] = None,
    compact_header_stub: bool = False,
) -> List[Tuple[Tuple[str, Optional[str]], str, Dict[str, object]]]:
    items: List[Tuple[Tuple[str, Optional[str]], str, Dict[str, object]]] = []
    if not os.path.isdir(folder):
        return items

    for dirpath, _, filenames in os.walk(folder):
        for fn in filenames:
            if not fn.lower().endswith(".txt"):
                continue
            if fn.endswith("-diagrams.txt"):
                continue
            path = os.path.join(dirpath, fn)
            meta = meta_from_path(path, folder)
            if meta is None:
                continue
            if orders is not None and meta["order"] not in orders:
                continue
            if ranks is not None and meta["rank"] not in ranks:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_lines = f.readlines()

                diagram_table = _load_diagram_table(path, list(raw_lines))
                file_header_stub: Optional[str] = None

                for raw in raw_lines:
                    line = raw.strip()
                    if not line:
                        continue
                    if any(s in line for s in BAD_SNIPPETS):
                        continue
                    if DIAGRAM_LINE_RE.match(line):
                        continue

                    if compact_header_stub and file_header_stub is None and line.startswith("Interaction:"):
                        body = line[len("Interaction:") :].strip()
                        header = body.split(" : ", 1)[0].strip()
                        file_header_stub = header_to_stub_diagram(header)

                    row = _parse_interaction_line(
                        line,
                        meta,
                        diagram_table,
                        file_header_stub if compact_header_stub else None,
                    )
                    if row is not None:
                        items.append(row)
            except Exception:
                continue
    return items
