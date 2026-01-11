"""
Deskriptive Analyse für Ist-Graphen (Instanzen) und LBS-Soll-Template-Graphen (Schema).

Ziele (Goal 2 aus Proposal):
- Transparenter Überblick: "Was ist drin? Wie oft? In welcher Form?"
- Vergleich Ist vs Soll: "Wo/Wie unterscheiden sie sich?"
- Technischer Abgleich mit Feature-Encoding (graph_pipeline.py): "Was nutzt das Modell tatsächlich?"
- Qualitäts-/Vollständigkeitschecks: "Fehlt etwas? Ist etwas redundant/unbenutzt?"

Eingabeformat (JSONL):
Jede Zeile ist ein Graph als JSON-Dict mit mindestens:
{
  "graph_id": "...",
  "nodes": [{"id": "...", "type": "MaLo|MeLo|TR|NeLo", "attrs": {...}}, ...],
  "edges": [{"src": "...", "dst": "...", "rel": "MEMA|METR|MENE|MEME|..."}, ...],
  "graph_attrs": {...}
}
Templates können zusätzlich "label" haben.

Ausgabe:
- out_dir/report.md (Markdown-Report)
- out_dir/tables/*.csv (Tabellen)
- out_dir/plots/*.png (Plots, wenn matplotlib installiert ist)

Wichtige Definitionen (Quick Wins 1/2/4):
- Originale JSON-Kanten sind i.d.R. **gerichtet** (src -> dst).
- Viele Strukturmetriken werden zusätzlich in einer **undirected**-Sicht ausgewertet
  (Kanten als ungerichtete Nachbarschaft), weil das später für Message Passing im GNN
  häufig relevant ist.
- Alle Strukturmetriken (Komponenten, Degree, Dichte, Isolierte Knoten, etc.) werden
  ausschließlich auf **valide Kanten** berechnet (src/dst referenzieren existierende Nodes).
  Ungültige Kanten werden separat gezählt und als Qualitätsindikator ausgewiesen.

Aufruf (Beispiel):
python descriptive_analysis.py ^
  --ist data/ist_graphs_all.jsonl ^
  --soll data/lbs_soll_graphs_pro.jsonl ^
  --out_dir analysis/descriptive ^
  --plots

Hinweis:
- Das Script liest standardmäßig ALLE Zeilen (keine Stichprobe).
- Mit --max_graphs kannst du für schnelle Iterationen begrenzen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional Dependencies
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore

# Try to import graph_pipeline to stay consistent with your model encoding.
# Falls Import fehlschlägt (z.B. weil du das Script nicht aus dem Repo-Root startest),
# versucht das Script zusätzlich den Ordner der Script-Datei zu sys.path hinzuzufügen.
gp = None  # type: ignore
try:
    import graph_pipeline as gp  # type: ignore
except Exception:
    try:
        import sys

        this_dir = str(Path(__file__).resolve().parent)
        if this_dir not in sys.path:
            sys.path.insert(0, this_dir)
        import graph_pipeline as gp  # type: ignore
    except Exception:
        gp = None  # type: ignore


# -------------------------
# Helpers: IO + Reporting
# -------------------------

def _read_text_with_fallback(path: Path) -> str:
    """
    Read text with UTF-8 first, fallback to cp1252.
    Raises with a helpful message if both fail.
    """
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Kann Datei nicht als UTF-8 oder cp1252 dekodieren: {path}",
            ) from e


def load_jsonl(path: Path, max_graphs: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts. Reads ALL lines by default (no sampling)."""
    text = _read_text_with_fallback(path)
    graphs: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if max_graphs is not None and len(graphs) >= max_graphs:
            break
        line = line.strip()
        if not line:
            continue
        try:
            g = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error in {path} at line {i}: {e}") from e
        graphs.append(g)
    return graphs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_div(n: float, d: float) -> float:
    return n / d if d else float("nan")


def pct(n: float, d: float) -> float:
    return 100.0 * safe_div(n, d) if d else 0.0


def fmt_pct(n: float, d: float, digits: int = 1) -> str:
    return f"{pct(n,d):.{digits}f}%"


def fmt_int(n: Optional[int]) -> str:
    return "∞" if n is None else str(n)


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> Dict[str, float]:
    if not values:
        return {f"q{int(q*100)}": float("nan") for q in qs}
    s = sorted(values)
    out: Dict[str, float] = {}
    for q in qs:
        # nearest-rank style
        idx = int(round(q * (len(s) - 1)))
        out[f"q{int(q*100)}"] = float(s[idx])
    return out


@dataclass
class Table:
    name: str
    rows: List[Dict[str, Any]]


class ReportWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        ensure_dir(out_dir)
        ensure_dir(out_dir / "tables")
        ensure_dir(out_dir / "plots")
        self.lines: List[str] = []

    def h1(self, title: str) -> None:
        self.lines.append(f"# {title}\n")

    def h2(self, title: str) -> None:
        self.lines.append(f"## {title}\n")

    def h3(self, title: str) -> None:
        self.lines.append(f"### {title}\n")

    def p(self, text: str) -> None:
        self.lines.append(text.strip() + "\n")

    def bullet(self, items: Sequence[str]) -> None:
        for it in items:
            self.lines.append(f"- {it}")
        self.lines.append("")

    def codeblock(self, code: str, lang: str = "") -> None:
        self.lines.append(f"```{lang}")
        self.lines.append(code.rstrip())
        self.lines.append("```")
        self.lines.append("")

    def save_table_csv(self, table: Table) -> Path:
        out = self.out_dir / "tables" / f"{table.name}.csv"
        if not table.rows:
            out.write_text("", encoding="utf-8")
            return out
        # Determine column order: stable (sorted), but keep common keys first if present
        keys = set().union(*(r.keys() for r in table.rows))
        preferred = [
            "key",
            "count",
            "pct",
            "dataset",
            "kind",
            "type",
            "rel",
            "attr",
            "value",
            "graph_id",
            "lbs_code",
        ]
        cols = [c for c in preferred if c in keys] + sorted([c for c in keys if c not in preferred])
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in table.rows:
                w.writerow({c: r.get(c, "") for c in cols})
        return out

    def add_table_markdown(self, table: Table, max_rows: int = 40) -> None:
        """Add a markdown table to report. Also writes a CSV file to out_dir/tables."""
        csv_path = self.save_table_csv(table)
        self.p(f"Tabellen-Export: `{csv_path.as_posix()}`")
        rows = table.rows[:max_rows]
        if not rows:
            self.p("_Keine Daten._")
            return
        if pd is not None:
            df = pd.DataFrame(rows)
            self.lines.append(df.to_markdown(index=False))
            self.lines.append("")
        else:
            cols = list(rows[0].keys())
            self.lines.append("| " + " | ".join(cols) + " |")
            self.lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for r in rows:
                self.lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
            self.lines.append("")
        if len(table.rows) > max_rows:
            self.p(f"_Hinweis: Tabelle gekürzt auf {max_rows} Zeilen (insgesamt {len(table.rows)})._")

    def save_plot(self, fig, filename: str) -> Optional[Path]:
        if plt is None:
            return None
        out = self.out_dir / "plots" / filename
        fig.savefig(out, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return out

    def finalize(self) -> Path:
        out = self.out_dir / "report.md"
        out.write_text("\n".join(self.lines), encoding="utf-8")
        return out


# -------------------------
# Graph utilities
# -------------------------

def iter_nodes(g: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for n in g.get("nodes", []) or []:
        if isinstance(n, dict):
            yield n


def iter_edges(g: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for e in g.get("edges", []) or []:
        if isinstance(e, dict):
            yield e


def node_id_set(g: Dict[str, Any]) -> set:
    # NOTE: Unique IDs; structural stats are computed on unique node ids.
    return {str(n.get("id")) for n in iter_nodes(g) if n.get("id") is not None}


def _edge_rel_raw(e: Dict[str, Any]) -> Optional[str]:
    # be tolerant: some generators might use other keys
    rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
    if rel is None:
        return None
    s = str(rel).strip()
    return s if s else None


def edge_validity_summary(g: Dict[str, Any]) -> Dict[str, int]:
    """Counts edge validity aspects (useful for Quick Win 1)."""
    ids = node_id_set(g)

    total = 0
    with_srcdst = 0
    with_rel = 0

    valid_endpoints = 0
    valid_typed = 0

    missing_srcdst = 0
    missing_rel = 0
    missing_refs = 0
    self_loops_valid = 0

    for e in iter_edges(g):
        total += 1
        s = e.get("src")
        t = e.get("dst")
        rel = _edge_rel_raw(e)

        if s is None or t is None:
            missing_srcdst += 1
            continue
        with_srcdst += 1

        s = str(s)
        t = str(t)

        if rel is None:
            missing_rel += 1
        else:
            with_rel += 1

        if s not in ids or t not in ids:
            missing_refs += 1
            continue

        valid_endpoints += 1
        if s == t:
            self_loops_valid += 1
        if rel is not None:
            valid_typed += 1

    return {
        "edges_total": total,
        "edges_with_srcdst": with_srcdst,
        "edges_with_rel": with_rel,
        "edges_valid_endpoints": valid_endpoints,
        "edges_valid_typed": valid_typed,
        "edges_missing_srcdst": missing_srcdst,
        "edges_missing_rel": missing_rel,
        "edges_missing_refs": missing_refs,
        "edges_self_loops_valid": self_loops_valid,
    }


def iter_valid_struct_edges(g: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """Yields edges (src,dst) where src/dst exist as node ids."""
    ids = node_id_set(g)
    for e in iter_edges(g):
        s = e.get("src")
        t = e.get("dst")
        if s is None or t is None:
            continue
        s = str(s)
        t = str(t)
        if s not in ids or t not in ids:
            continue
        yield s, t


def iter_valid_typed_edges(g: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    """Yields edges (src,dst,rel) where src/dst exist and rel exists."""
    ids = node_id_set(g)
    for e in iter_edges(g):
        s = e.get("src")
        t = e.get("dst")
        if s is None or t is None:
            continue
        rel = _edge_rel_raw(e)
        if rel is None:
            continue
        s = str(s)
        t = str(t)
        if s not in ids or t not in ids:
            continue
        yield s, t, rel.strip().upper()


def build_undirected_neighbors(g: Dict[str, Any]) -> Dict[str, set]:
    """Undirected neighbor sets (unique neighbors, valid edges only). Self-loops ignored."""
    ids = node_id_set(g)
    nbrs: Dict[str, set] = {nid: set() for nid in ids}
    for s, t in iter_valid_struct_edges(g):
        if s == t:
            continue
        nbrs[s].add(t)
        nbrs[t].add(s)
    return nbrs


def build_directed_neighbors(g: Dict[str, Any]) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Directed neighbor sets (unique neighbors, valid edges only). Self-loops ignored."""
    ids = node_id_set(g)
    out_nbrs: Dict[str, set] = {nid: set() for nid in ids}
    in_nbrs: Dict[str, set] = {nid: set() for nid in ids}
    for s, t in iter_valid_struct_edges(g):
        if s == t:
            continue
        out_nbrs[s].add(t)
        in_nbrs[t].add(s)
    return out_nbrs, in_nbrs


def connected_components_sizes(g: Dict[str, Any]) -> List[int]:
    """Connected components on undirected view (valid edges only)."""
    nbrs = build_undirected_neighbors(g)
    seen = set()
    sizes: List[int] = []
    for start in nbrs.keys():
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in nbrs.get(u, set()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def strongly_connected_components_sizes(g: Dict[str, Any]) -> List[int]:
    """SCC sizes on directed view (Kosaraju), valid edges only."""
    out_nbrs, in_nbrs = build_directed_neighbors(g)

    visited = set()
    order: List[str] = []

    def dfs1(u: str) -> None:
        visited.add(u)
        for v in out_nbrs.get(u, set()):
            if v not in visited:
                dfs1(v)
        order.append(u)

    for u in out_nbrs.keys():
        if u not in visited:
            dfs1(u)

    visited2 = set()
    sizes: List[int] = []

    def dfs2(u: str) -> int:
        visited2.add(u)
        size = 1
        for v in in_nbrs.get(u, set()):
            if v not in visited2:
                size += dfs2(v)
        return size

    for u in reversed(order):
        if u not in visited2:
            sizes.append(dfs2(u))

    sizes.sort(reverse=True)
    return sizes


def graph_struct_metrics(g: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-graph structural metrics (Quick Wins 1/2/4)."""
    ids = node_id_set(g)
    n = len(ids)

    ev = edge_validity_summary(g)

    # undirected degrees (unique neighbors)
    nbrs = build_undirected_neighbors(g)
    degrees = [len(nbrs.get(nid, set())) for nid in ids]
    isolated = sum(1 for d in degrees if d == 0)

    # unique undirected edges (pairs), valid only, self-loops excluded
    und_pairs = set()
    for s, t in iter_valid_struct_edges(g):
        if s == t:
            continue
        a, b = (s, t) if s <= t else (t, s)
        und_pairs.add((a, b))
    m_und = len(und_pairs)

    density = (2.0 * m_und / (n * (n - 1))) if n > 1 else 0.0

    comps = connected_components_sizes(g)
    sccs = strongly_connected_components_sizes(g)

    out_nbrs, in_nbrs = build_directed_neighbors(g)
    out_degs = [len(out_nbrs.get(nid, set())) for nid in ids]
    in_degs = [len(in_nbrs.get(nid, set())) for nid in ids]

    deg_q = quantiles([float(d) for d in degrees])

    return {
        # quality counters
        **ev,
        # core sizes
        "node_count_unique": n,
        "edge_pairs_undirected_unique": m_und,
        # undirected structure
        "density_undirected_simple": density,
        "edges_per_node_valid_endpoints": safe_div(float(ev["edges_valid_endpoints"]), float(n)),
        "edges_per_node_valid_typed": safe_div(float(ev["edges_valid_typed"]), float(n)),
        "avg_degree_undirected": float(statistics.mean(degrees)) if degrees else float("nan"),
        "deg_min_undirected": int(min(degrees)) if degrees else None,
        "deg_med_undirected": int(deg_q["q50"]) if degrees else None,
        "deg_max_undirected": int(max(degrees)) if degrees else None,
        "isolated_nodes_undirected": int(isolated),
        "isolated_pct_undirected": pct(float(isolated), float(n)),
        "cc_count_undirected": int(len(comps)),
        "cc_largest_undirected": int(comps[0]) if comps else 0,
        # directed summary (unique neighbors)
        "avg_out_degree_directed": float(statistics.mean(out_degs)) if out_degs else float("nan"),
        "avg_in_degree_directed": float(statistics.mean(in_degs)) if in_degs else float("nan"),
        "out_degree_max_directed": int(max(out_degs)) if out_degs else 0,
        "in_degree_max_directed": int(max(in_degs)) if in_degs else 0,
        "scc_count_directed": int(len(sccs)),
        "scc_largest_directed": int(sccs[0]) if sccs else 0,
    }


def degrees_by_type(g: Dict[str, Any]) -> Dict[str, List[int]]:
    """Undirected unique-neighbor degree sequence, grouped by node type."""
    nbrs = build_undirected_neighbors(g)
    id_to_type: Dict[str, str] = {}
    for n in iter_nodes(g):
        nid = n.get("id")
        if nid is None:
            continue
        id_to_type[str(nid)] = str(n.get("type"))

    out: Dict[str, List[int]] = defaultdict(list)
    for nid, neis in nbrs.items():
        out[id_to_type.get(nid, "UNKNOWN")].append(len(neis))
    return out


def directed_degrees_by_type(g: Dict[str, Any]) -> Dict[str, Dict[str, List[int]]]:
    """Directed degrees (unique in/out neighbors), grouped by node type."""
    out_nbrs, in_nbrs = build_directed_neighbors(g)
    id_to_type: Dict[str, str] = {}
    for n in iter_nodes(g):
        nid = n.get("id")
        if nid is None:
            continue
        id_to_type[str(nid)] = str(n.get("type"))

    out: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"out": [], "in": []})
    for nid in out_nbrs.keys():
        t = id_to_type.get(nid, "UNKNOWN")
        out[t]["out"].append(len(out_nbrs.get(nid, set())))
        out[t]["in"].append(len(in_nbrs.get(nid, set())))
    return out


def graph_signature(g: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight graph fingerprint (Nice-to-have 5)."""
    # Node types (by unique ids)
    ids = node_id_set(g)
    id_to_type: Dict[str, str] = {}
    for n in iter_nodes(g):
        nid = n.get("id")
        if nid is None:
            continue
        id_to_type[str(nid)] = str(n.get("type"))
    nt = Counter(id_to_type.get(nid, "UNKNOWN") for nid in ids)

    # Edge rel types (valid typed)
    et = Counter(rel for _, _, rel in iter_valid_typed_edges(g))

    # Degree summary (undirected, valid)
    nbrs = build_undirected_neighbors(g)
    degrees = [len(nbrs.get(nid, set())) for nid in ids]
    dq = quantiles([float(d) for d in degrees])
    comps = connected_components_sizes(g)

    # Normalize counts to stable ordering
    node_part = ",".join([f"{t}={int(nt.get(t,0))}" for t in ["MaLo", "MeLo", "TR", "NeLo", "UNKNOWN"] if t in nt or t != "UNKNOWN"])
    edge_part = ",".join([f"{r}={int(et.get(r,0))}" for r in ["MEMA", "METR", "MENE", "MEME"]])

    sig_str = f"NT:{node_part}|ET:{edge_part}|CC:{len(comps)}|DEG:{int(dq['q0']) if degrees else 0},{int(dq['q50']) if degrees else 0},{int(dq['q100']) if degrees else 0}"
    sig_hash = hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:10]

    return {
        "signature": sig_str,
        "signature_id": sig_hash,
        "node_types": dict(nt),
        "edge_types": dict(et),
        "cc_count": len(comps),
        "deg_min": int(dq["q0"]) if degrees else 0,
        "deg_med": int(dq["q50"]) if degrees else 0,
        "deg_max": int(dq["q100"]) if degrees else 0,
    }


# -------------------------
# Validation + Extraction
# -------------------------


@dataclass
class ValidationIssue:
    severity: str  # "ERROR" | "WARN"
    where: str  # graph_id or context
    message: str


def validate_graph(g: Dict[str, Any], kind: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    gid = str(g.get("graph_id", "<no-graph_id>"))

    for k in ["graph_id", "nodes", "edges", "graph_attrs"]:
        if k not in g:
            issues.append(ValidationIssue("ERROR", gid, f"Missing top-level key: {k}"))

    # node IDs unique
    ids: List[str] = []
    for n in iter_nodes(g):
        nid = n.get("id")
        if nid is None:
            issues.append(ValidationIssue("ERROR", gid, "Node without id"))
            continue
        ids.append(str(nid))
    if len(ids) != len(set(ids)):
        dup = [k for k, c in Counter(ids).items() if c > 1][:10]
        issues.append(ValidationIssue("ERROR", gid, f"Duplicate node ids (sample): {dup}"))

    idset = set(ids)

    # edges reference existing nodes
    missing_refs = 0
    missing_fields = 0
    for e in iter_edges(g):
        s = e.get("src")
        t = e.get("dst")
        rel = _edge_rel_raw(e)
        if s is None or t is None or rel is None:
            missing_fields += 1
            continue
        if str(s) not in idset or str(t) not in idset:
            missing_refs += 1

    if missing_fields:
        issues.append(ValidationIssue("ERROR", gid, f"{missing_fields} edges missing src/dst/rel"))
    if missing_refs:
        issues.append(ValidationIssue("ERROR", gid, f"{missing_refs} edges reference missing node ids"))

    # unknown node/edge types
    known_node_types = {"MaLo", "MeLo", "TR", "NeLo"}
    unknown_nt = Counter()
    for n in iter_nodes(g):
        t = str(n.get("type"))
        if t not in known_node_types:
            unknown_nt[t] += 1
    if unknown_nt:
        issues.append(ValidationIssue("WARN", gid, f"Unknown node types: {dict(unknown_nt)}"))

    known_edge_types = {"MEMA", "METR", "MENE", "MEME"}
    unknown_rel = Counter()
    for _, _, r in iter_valid_typed_edges(g):
        if r not in known_edge_types:
            unknown_rel[r] += 1
    if unknown_rel:
        issues.append(ValidationIssue("WARN", gid, f"Unknown edge rel types: {dict(unknown_rel)}"))

    # Graph attrs count consistency (Ist only)
    ga = g.get("graph_attrs", {}) or {}
    if kind == "ist":
        actual = Counter(str(n.get("type")) for n in iter_nodes(g))
        for t, key in [("MaLo", "malo_count"), ("MeLo", "melo_count"), ("TR", "tr_count"), ("NeLo", "nelo_count")]:
            if key in ga:
                try:
                    declared = int(ga[key])
                except Exception:
                    issues.append(ValidationIssue("WARN", gid, f"graph_attrs.{key} not int: {ga[key]}"))
                    continue
                if declared != int(actual.get(t, 0)):
                    issues.append(
                        ValidationIssue(
                            "WARN", gid, f"Count mismatch {key}: declared={declared} actual={actual.get(t,0)}"
                        )
                    )

    # Template bounds sanity
    if kind == "soll":
        for t in ["malo", "melo", "tr", "nelo"]:
            mn = ga.get(f"{t}_min")
            mx = ga.get(f"{t}_max")
            if mn is None or mx is None:
                continue
            try:
                mn_i = int(mn)
            except Exception:
                issues.append(ValidationIssue("WARN", gid, f"{t}_min not int: {mn}"))
                continue
            if mx is not None:
                try:
                    mx_i = int(mx)
                    if mx_i < mn_i:
                        issues.append(ValidationIssue("ERROR", gid, f"Invalid bounds: {t}_max({mx_i}) < {t}_min({mn_i})"))
                except Exception:
                    issues.append(ValidationIssue("WARN", gid, f"{t}_max not int/None: {mx}"))

    return issues


def extract_global_keysets(graphs: List[Dict[str, Any]]) -> Tuple[Counter, Counter, Counter]:
    """Returns: graph_attrs keys, node attrs keys, edge dict keys."""
    ga_keys = Counter()
    node_attr_keys = Counter()
    edge_keys = Counter()
    for g in graphs:
        ga = g.get("graph_attrs", {}) or {}
        for k in ga.keys():
            ga_keys[str(k)] += 1
        for n in iter_nodes(g):
            attrs = n.get("attrs", {}) or {}
            for k in attrs.keys():
                node_attr_keys[str(k)] += 1
        for e in iter_edges(g):
            for k in e.keys():
                edge_keys[str(k)] += 1
    return ga_keys, node_attr_keys, edge_keys


def node_type_counts(graphs: List[Dict[str, Any]]) -> Counter:
    c = Counter()
    for g in graphs:
        for n in iter_nodes(g):
            c[str(n.get("type"))] += 1
    return c


def edge_type_counts(graphs: List[Dict[str, Any]]) -> Counter:
    c = Counter()
    for g in graphs:
        for _, _, rel in iter_valid_typed_edges(g):
            c[str(rel)] += 1
    return c


def per_graph_basic_df(graphs: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for g in graphs:
        ga = g.get("graph_attrs", {}) or {}

        # keep raw sizes (backward compatible)
        node_count_raw = len(list(iter_nodes(g)))
        edge_count_raw = len(list(iter_edges(g)))

        sm = graph_struct_metrics(g)

        row: Dict[str, Any] = {
            "kind": kind,
            "graph_id": g.get("graph_id"),
            "node_count": node_count_raw,
            "edge_count": edge_count_raw,
            **sm,
        }

        if kind == "ist":
            row["dataset"] = ga.get("dataset")
            for k in ["malo_count", "melo_count", "tr_count", "nelo_count"]:
                row[k] = ga.get(k)
        else:
            row["lbs_code"] = ga.get("lbs_code") or g.get("label")
            for t in ["malo", "melo", "tr", "nelo"]:
                row[f"{t}_min"] = ga.get(f"{t}_min")
                row[f"{t}_max"] = ga.get(f"{t}_max")
                row[f"{t}_node_types"] = ga.get(f"{t}_node_types")
            row["attachment_rules_n"] = len(ga.get("attachment_rules") or [])
            row["constraints_n"] = len(ga.get("optionality_constraints") or [])

        # Edge type breakdown (valid typed)
        etc = Counter(rel for _, _, rel in iter_valid_typed_edges(g))
        for rel in ["MEMA", "METR", "MENE", "MEME"]:
            row[f"edge_{rel}"] = int(etc.get(rel, 0))

        rows.append(row)

    return rows


# -------------------------
# Feature-Encoding alignment
# -------------------------


def _normalize_direction_fallback(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if "consumption" in s and "generation" in s:
        return "both"
    if "verbrauch" in s:
        return "consumption"
    if "einspeis" in s or "generation" in s:
        return "generation"
    if s in ("consumption", "generation", "both"):
        return s
    return s


def encoder_alignment(graphs: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    """Compute how often values fall into 'unknown' bucket according to graph_pipeline mappings."""
    if gp is not None:
        NODE_TYPES = getattr(gp, "NODE_TYPES", {"MaLo": 0, "MeLo": 1, "TR": 2, "NeLo": 3})
        DIRECTIONS = getattr(gp, "DIRECTIONS", {"consumption": 0, "generation": 1, "both": 2})
        MELO_FUNCTIONS = getattr(gp, "MELO_FUNCTIONS", {"N": 0, "H": 1, "D": 2, "S": 3})
        VOLTAGE_LEVELS = getattr(gp, "VOLTAGE_LEVELS", {"E05": 0, "E06": 1})
        normalize_dir = getattr(gp, "_normalize_direction", _normalize_direction_fallback)
    else:
        NODE_TYPES = {"MaLo": 0, "MeLo": 1, "TR": 2, "NeLo": 3}
        DIRECTIONS = {"consumption": 0, "generation": 1, "both": 2}
        MELO_FUNCTIONS = {"N": 0, "H": 1, "D": 2, "S": 3}
        VOLTAGE_LEVELS = {"E05": 0, "E06": 1}
        normalize_dir = _normalize_direction_fallback

    counts = Counter()
    unknown = Counter()
    raw_dir_vals = Counter()
    raw_fn_vals = Counter()
    raw_volt_vals = Counter()
    raw_nt_vals = Counter()
    raw_rel_vals = Counter()

    for g in graphs:
        for n in iter_nodes(g):
            ntype = str(n.get("type"))
            raw_nt_vals[ntype] += 1
            if ntype not in NODE_TYPES:
                unknown["node_type_unknown"] += 1
            counts["nodes_total"] += 1

            attrs = n.get("attrs", {}) or {}

            # direction (MaLo/TR only)
            if ntype == "TR":
                raw_dir = attrs.get("tr_direction") or attrs.get("direction")
            elif ntype == "MaLo":
                raw_dir = attrs.get("direction") or attrs.get("direction_hint")
            else:
                raw_dir = None

            if raw_dir is not None:
                raw_dir_vals[str(raw_dir)] += 1
            canon = normalize_dir(raw_dir)
            if ntype in ("MaLo", "TR"):
                counts["dir_nodes_total"] += 1
                if canon is None or canon not in DIRECTIONS:
                    unknown["dir_unknown"] += 1

            # melo function (MeLo only)
            if ntype == "MeLo":
                raw_fn = attrs.get("function") or attrs.get("melo_function")
                if raw_fn is not None:
                    raw_fn_vals[str(raw_fn)] += 1
                counts["melo_fn_nodes_total"] += 1
                if raw_fn is None or str(raw_fn) not in MELO_FUNCTIONS:
                    unknown["melo_fn_unknown"] += 1

                # voltage (MeLo only)
                raw_volt = attrs.get("voltage_level")
                if raw_volt is not None:
                    raw_volt_vals[str(raw_volt)] += 1
                counts["volt_nodes_total"] += 1
                if raw_volt is None or str(raw_volt) not in VOLTAGE_LEVELS:
                    unknown["volt_unknown"] += 1

        for _, _, rel in iter_valid_typed_edges(g):
            raw_rel_vals[rel] += 1
            counts["edges_total"] += 1

    return {
        "kind": kind,
        "counts": dict(counts),
        "unknown": dict(unknown),
        "raw_node_types": raw_nt_vals,
        "raw_edge_types": raw_rel_vals,
        "raw_direction_values": raw_dir_vals,
        "raw_melo_function_values": raw_fn_vals,
        "raw_voltage_values": raw_volt_vals,
    }


# -------------------------
# Ist vs Soll coverage (bounds)
# -------------------------


def _as_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    try:
        return int(x)
    except Exception:
        return None


def within_bounds(val: int, mn: Optional[int], mx: Optional[int]) -> bool:
    if mn is not None and val < mn:
        return False
    if mx is not None and val > mx:
        return False
    return True


def ist_vs_template_coverage(ist_graphs: List[Dict[str, Any]], soll_templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For each template, count how many ist graphs satisfy the template's min/max bounds."""
    ist_rows: List[Dict[str, Any]] = []
    for g in ist_graphs:
        ga = g.get("graph_attrs", {}) or {}
        actual = Counter(str(n.get("type")) for n in iter_nodes(g))
        ist_rows.append(
            {
                "graph_id": g.get("graph_id"),
                "dataset": ga.get("dataset"),
                "malo": _as_int_or_none(ga.get("malo_count")) if ga.get("malo_count") is not None else int(actual.get("MaLo", 0)),
                "melo": _as_int_or_none(ga.get("melo_count")) if ga.get("melo_count") is not None else int(actual.get("MeLo", 0)),
                "tr": _as_int_or_none(ga.get("tr_count")) if ga.get("tr_count") is not None else int(actual.get("TR", 0)),
                "nelo": _as_int_or_none(ga.get("nelo_count")) if ga.get("nelo_count") is not None else int(actual.get("NeLo", 0)),
            }
        )

    out: List[Dict[str, Any]] = []
    for t in soll_templates:
        ga = t.get("graph_attrs", {}) or {}
        code = ga.get("lbs_code") or t.get("label") or t.get("graph_id")
        bounds = {}
        for k in ["malo", "melo", "tr", "nelo"]:
            bounds[f"{k}_min"] = _as_int_or_none(ga.get(f"{k}_min"))
            bounds[f"{k}_max"] = _as_int_or_none(ga.get(f"{k}_max"))
        ok = 0
        ok_by_dataset = Counter()
        for r in ist_rows:
            if (
                within_bounds(r["malo"], bounds["malo_min"], bounds["malo_max"])
                and within_bounds(r["melo"], bounds["melo_min"], bounds["melo_max"])
                and within_bounds(r["tr"], bounds["tr_min"], bounds["tr_max"])
                and within_bounds(r["nelo"], bounds["nelo_min"], bounds["nelo_max"])
            ):
                ok += 1
                ok_by_dataset[str(r["dataset"])] += 1
        row = {"lbs_code": code, "templates_graph_id": t.get("graph_id"), **bounds, "ist_graphs_fitting_total": ok}
        for ds, cnt in ok_by_dataset.items():
            row[f"ist_fitting_{ds}"] = int(cnt)
        out.append(row)

    out.sort(key=lambda r: (-int(r["ist_graphs_fitting_total"]), str(r["lbs_code"])))
    return out


# -------------------------
# Plot helpers
# -------------------------


def plot_hist(values: List[int], title: str, xlabel: str, out_path: Path) -> None:
    if plt is None:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(values, bins=min(60, max(5, int(math.sqrt(len(values))) if values else 5)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


# -------------------------
# Topology checklist (Nice-to-have 6)
# -------------------------


def topology_checklist_row(g: Dict[str, Any], kind: str) -> Dict[str, Any]:
    ids = node_id_set(g)
    id_to_type: Dict[str, str] = {}
    for n in iter_nodes(g):
        nid = n.get("id")
        if nid is None:
            continue
        id_to_type[str(nid)] = str(n.get("type"))

    nt = Counter(id_to_type.get(nid, "UNKNOWN") for nid in ids)
    et = Counter(rel for _, _, rel in iter_valid_typed_edges(g))

    melo_n = int(nt.get("MeLo", 0))
    malo_n = int(nt.get("MaLo", 0))
    tr_n = int(nt.get("TR", 0))
    nelo_n = int(nt.get("NeLo", 0))

    expects_mema = melo_n > 0 and malo_n > 0
    expects_metr = melo_n > 0 and tr_n > 0
    expects_mene = melo_n > 0 and nelo_n > 0

    has_mema = int(et.get("MEMA", 0)) > 0
    has_metr = int(et.get("METR", 0)) > 0
    has_mene = int(et.get("MENE", 0)) > 0
    has_meme = int(et.get("MEME", 0)) > 0

    ga = g.get("graph_attrs", {}) or {}
    lbs_code = ga.get("lbs_code") or g.get("label")

    ev = edge_validity_summary(g)

    rel_set = sorted(set(et.keys()))

    row: Dict[str, Any] = {
        "kind": kind,
        "graph_id": g.get("graph_id"),
        "lbs_code": lbs_code,
        "MaLo": malo_n,
        "MeLo": melo_n,
        "TR": tr_n,
        "NeLo": nelo_n,
        "edge_MEMA": int(et.get("MEMA", 0)),
        "edge_METR": int(et.get("METR", 0)),
        "edge_MENE": int(et.get("MENE", 0)),
        "edge_MEME": int(et.get("MEME", 0)),
        "rel_types_present": ",".join(rel_set) if rel_set else "",
        "expects_MEMA": expects_mema,
        "missing_MEMA": bool(expects_mema and not has_mema),
        "expects_METR": expects_metr,
        "missing_METR": bool(expects_metr and not has_metr),
        "expects_MENE": expects_mene,
        "missing_MENE": bool(expects_mene and not has_mene),
        "multi_MeLo": bool(melo_n > 1),
        "has_MEME": bool(has_meme),
        "attachment_rules_n": len(ga.get("attachment_rules") or []),
        "edges_missing_refs": int(ev.get("edges_missing_refs", 0)),
        "edges_missing_srcdst": int(ev.get("edges_missing_srcdst", 0)),
        "edges_missing_rel": int(ev.get("edges_missing_rel", 0)),
    }
    return row


# -------------------------
# Main analysis routine
# -------------------------


def run_analysis(
    ist_path: Path,
    soll_path: Path,
    out_dir: Path,
    max_graphs: Optional[int] = None,
    do_plots: bool = True,
) -> Path:
    rw = ReportWriter(out_dir)
    rw.h1("Deskriptive Analyse: Ist-Graphen vs. LBS-Soll-Templates")

    rw.h2("Inputs")
    rw.bullet(
        [
            f"Ist-Graphs: `{ist_path}`",
            f"Soll-Templates: `{soll_path}`",
            f"Max graphs (debug): `{max_graphs}`" if max_graphs is not None else "Max graphs: _alle_",
            f"Plots: {'an' if do_plots else 'aus'} (matplotlib {'gefunden' if plt is not None else 'nicht installiert'})",
            f"pandas: {'gefunden' if pd is not None else 'nicht installiert'}",
            f"graph_pipeline Import: {'ok' if gp is not None else 'nicht verfügbar (Fallback-Mappings)'}",
            "Strukturmetriken: nur valide Kanten (src/dst referenzieren existierende Nodes).",
        ]
    )

    # Load
    ist_graphs = load_jsonl(ist_path, max_graphs=max_graphs)
    soll_graphs = load_jsonl(soll_path, max_graphs=max_graphs)

    rw.p(f"Geladen: **{len(ist_graphs)}** Ist-Graphen, **{len(soll_graphs)}** Template-Graphen.")

    # Validate
    rw.h2("Schema- & Qualitätschecks")
    all_issues: List[ValidationIssue] = []
    for g in ist_graphs:
        all_issues.extend(validate_graph(g, "ist"))
    for g in soll_graphs:
        all_issues.extend(validate_graph(g, "soll"))

    by_sev = Counter(i.severity for i in all_issues)
    rw.bullet(
        [
            f"Issues total: {len(all_issues)}",
            f"ERROR: {by_sev.get('ERROR',0)}",
            f"WARN: {by_sev.get('WARN',0)}",
        ]
    )

    top_msgs = Counter(i.message for i in all_issues)
    issue_rows = [{"message": msg, "count": int(cnt)} for msg, cnt in top_msgs.most_common(25)]
    rw.h3("Häufigste Issue-Typen")
    rw.add_table_markdown(Table("issues_top", issue_rows))

    if by_sev.get("ERROR", 0) > 0:
        sample_err = [i for i in all_issues if i.severity == "ERROR"][:30]
        rw.h3("Beispielhafte ERROR-Kontexte")
        rw.add_table_markdown(Table("issues_error_samples", [{"graph_id": e.where, "message": e.message} for e in sample_err]))

    # Keysets
    rw.h2("Inventar: Welche Felder sind enthalten?")
    ist_ga_keys, ist_node_attr_keys, ist_edge_keys = extract_global_keysets(ist_graphs)
    soll_ga_keys, soll_node_attr_keys, soll_edge_keys = extract_global_keysets(soll_graphs)

    def top_counter_table(name: str, c: Counter, total_graphs: int) -> Table:
        rows = []
        for k, cnt in c.most_common(50):
            rows.append({"key": k, "count": int(cnt), "pct": f"{pct(cnt,total_graphs):.1f}%"})
        return Table(name, rows)

    rw.h3("Ist: graph_attrs Keys (wie oft pro Graph vorhanden)")
    rw.add_table_markdown(top_counter_table("ist_graph_attrs_keys", ist_ga_keys, len(ist_graphs)))

    rw.h3("Soll: graph_attrs Keys (wie oft pro Graph vorhanden)")
    rw.add_table_markdown(top_counter_table("soll_graph_attrs_keys", soll_ga_keys, len(soll_graphs)))

    rw.h3("Ist: Node-Attribute Keys (wie oft über alle Nodes)")
    rw.add_table_markdown(Table("ist_node_attr_keys", [{"key": k, "count": int(v)} for k, v in ist_node_attr_keys.most_common(60)]))

    rw.h3("Soll: Node-Attribute Keys (wie oft über alle Nodes)")
    rw.add_table_markdown(Table("soll_node_attr_keys", [{"key": k, "count": int(v)} for k, v in soll_node_attr_keys.most_common(60)]))

    # Node/Edge type distributions
    rw.h2("Struktur: Knotentypen & Kantentypen")
    ist_nt = node_type_counts(ist_graphs)
    soll_nt = node_type_counts(soll_graphs)
    ist_et = edge_type_counts(ist_graphs)
    soll_et = edge_type_counts(soll_graphs)

    rw.h3("Knotentyp-Verteilung (gesamt)")
    rows = []
    for t in sorted(set(ist_nt.keys()) | set(soll_nt.keys())):
        rows.append({"type": t, "ist_count": int(ist_nt.get(t, 0)), "soll_count": int(soll_nt.get(t, 0))})
    rw.add_table_markdown(Table("node_type_distribution", rows))

    rw.h3("Kantentyp-Verteilung (gesamt; nur valide Kanten)")
    rows = []
    for r in sorted(set(ist_et.keys()) | set(soll_et.keys())):
        rows.append({"rel": r, "ist_count": int(ist_et.get(r, 0)), "soll_count": int(soll_et.get(r, 0))})
    rw.add_table_markdown(Table("edge_type_distribution", rows))

    # Per-graph stats
    rw.h2("Pro-Graph Kennzahlen")
    ist_rows = per_graph_basic_df(ist_graphs, "ist")
    soll_rows = per_graph_basic_df(soll_graphs, "soll")

    # Export (sample in markdown, full in CSV)
    rw.h3("Per-Graph-Tabelle (Sample)")
    rw.p("Hinweis: Vollständige Tabellen liegen als CSV unter `tables/`.")
    rw.add_table_markdown(Table("per_graph_ist", ist_rows), max_rows=25)
    rw.add_table_markdown(Table("per_graph_soll", soll_rows), max_rows=25)

    # Size distributions
    ist_nodes = [int(r["node_count"]) for r in ist_rows]
    ist_edges = [int(r["edge_count"]) for r in ist_rows]
    soll_nodes = [int(r["node_count"]) for r in soll_rows]
    soll_edges = [int(r["edge_count"]) for r in soll_rows]

    rw.h3("Größenverteilungen (Nodes/Edges, raw)")
    size_rows = []
    for kind, nodes, edges in [("ist", ist_nodes, ist_edges), ("soll", soll_nodes, soll_edges)]:
        qn = quantiles([float(x) for x in nodes])
        qe = quantiles([float(x) for x in edges])
        size_rows.append(
            {
                "kind": kind,
                "graphs": len(nodes),
                "nodes_min": min(nodes) if nodes else None,
                "nodes_med": qn["q50"],
                "nodes_max": max(nodes) if nodes else None,
                "edges_min": min(edges) if edges else None,
                "edges_med": qe["q50"],
                "edges_max": max(edges) if edges else None,
            }
        )
    rw.add_table_markdown(Table("size_distributions", size_rows))

    if do_plots:
        plot_hist(ist_nodes, "Ist-Graphen: Node count (raw)", "node_count", rw.out_dir / "plots" / "ist_node_count.png")
        plot_hist(ist_edges, "Ist-Graphen: Edge count (raw)", "edge_count", rw.out_dir / "plots" / "ist_edge_count.png")
        plot_hist(soll_nodes, "Templates: Node count (raw)", "node_count", rw.out_dir / "plots" / "soll_node_count.png")
        plot_hist(soll_edges, "Templates: Edge count (raw)", "edge_count", rw.out_dir / "plots" / "soll_edge_count.png")
        rw.p("Plots gespeichert unter `plots/` (siehe Dateien).")

    # Connectivity + new structural stats
    rw.h3("Connectedness / Komponenten (undirected; valide Kanten)")
    comp_rows = []
    for kind, rows_ in [("ist", ist_rows), ("soll", soll_rows)]:
        cc_counts = [int(r["cc_count_undirected"]) for r in rows_]
        largest = [int(r["cc_largest_undirected"]) for r in rows_]
        comp_rows.append(
            {
                "kind": kind,
                "graphs": len(rows_),
                "cc_count_mean": round(statistics.mean(cc_counts), 3) if cc_counts else None,
                "cc_count_max": max(cc_counts) if cc_counts else None,
                "largest_cc_mean": round(statistics.mean(largest), 3) if largest else None,
                "largest_cc_min": min(largest) if largest else None,
            }
        )
    rw.add_table_markdown(Table("connectivity_summary", comp_rows))

    rw.h3("Dichte / E/N / Degree / Isolierte Knoten (undirected; valide Kanten)")

    def _struct_summary(rows_: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
        dens = [float(r["density_undirected_simple"]) for r in rows_]
        e_over_n = [float(r["edges_per_node_valid_endpoints"]) for r in rows_]
        avg_deg = [float(r["avg_degree_undirected"]) for r in rows_]
        iso_pct = [float(r["isolated_pct_undirected"]) for r in rows_]
        return {
            "kind": kind,
            "graphs": len(rows_),
            "density_mean": round(statistics.mean(dens), 6) if dens else None,
            "density_med": round(quantiles(dens)["q50"], 6) if dens else None,
            "E_per_N_mean": round(statistics.mean(e_over_n), 6) if e_over_n else None,
            "avg_deg_mean": round(statistics.mean(avg_deg), 6) if avg_deg else None,
            "isolated_pct_mean": round(statistics.mean(iso_pct), 4) if iso_pct else None,
        }

    rw.add_table_markdown(Table("struct_metrics_summary", [_struct_summary(ist_rows, "ist"), _struct_summary(soll_rows, "soll")]))

    # Degree stats per type (undirected) + directed degree stats
    rw.h3("Degree-Statistiken pro Knotentyp (undirected; unique neighbors)")

    def agg_degree_undirected(graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        acc: Dict[str, List[int]] = defaultdict(list)
        for g in graphs:
            per_t = degrees_by_type(g)
            for t, vals in per_t.items():
                acc[t].extend(vals)
        out = []
        for t, vals in sorted(acc.items()):
            out.append(
                {
                    "type": t,
                    "count": len(vals),
                    "deg_mean": round(statistics.mean(vals), 3) if vals else None,
                    "deg_min": min(vals) if vals else None,
                    "deg_max": max(vals) if vals else None,
                }
            )
        return out

    rw.add_table_markdown(Table("degree_stats_ist", agg_degree_undirected(ist_graphs)))
    rw.add_table_markdown(Table("degree_stats_soll", agg_degree_undirected(soll_graphs)))

    rw.h3("Degree-Statistiken pro Knotentyp (directed; in/out, unique neighbors)")

    def agg_degree_directed(graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        acc_out: Dict[str, List[int]] = defaultdict(list)
        acc_in: Dict[str, List[int]] = defaultdict(list)
        for g in graphs:
            per_t = directed_degrees_by_type(g)
            for t, d in per_t.items():
                acc_out[t].extend(d["out"])
                acc_in[t].extend(d["in"])
        out = []
        for t in sorted(set(acc_out.keys()) | set(acc_in.keys())):
            outs = acc_out.get(t, [])
            ins = acc_in.get(t, [])
            out.append(
                {
                    "type": t,
                    "count": len(outs),
                    "out_mean": round(statistics.mean(outs), 3) if outs else None,
                    "out_max": max(outs) if outs else None,
                    "in_mean": round(statistics.mean(ins), 3) if ins else None,
                    "in_max": max(ins) if ins else None,
                }
            )
        return out

    rw.add_table_markdown(Table("degree_stats_directed_ist", agg_degree_directed(ist_graphs)))
    rw.add_table_markdown(Table("degree_stats_directed_soll", agg_degree_directed(soll_graphs)))

    # Degree distributions (plots) - Quick Win 3
    rw.h3("Degree-Verteilungen (Plots; undirected, unique neighbors)")
    if do_plots:
        def collect_all_degrees(graphs: List[Dict[str, Any]]) -> Dict[str, List[int]]:
            acc: Dict[str, List[int]] = defaultdict(list)
            for g in graphs:
                per = degrees_by_type(g)
                # overall
                for t, vals in per.items():
                    acc["ALL"].extend(vals)
                    acc[t].extend(vals)
            return acc

        ist_deg = collect_all_degrees(ist_graphs)
        soll_deg = collect_all_degrees(soll_graphs)

        # overall
        plot_hist(ist_deg.get("ALL", []), "Ist: Degree distribution (undirected)", "degree", rw.out_dir / "plots" / "ist_degree_all.png")
        plot_hist(soll_deg.get("ALL", []), "Soll: Degree distribution (undirected)", "degree", rw.out_dir / "plots" / "soll_degree_all.png")

        # per type (bounded set)
        for t in ["MaLo", "MeLo", "TR", "NeLo"]:
            if ist_deg.get(t):
                plot_hist(ist_deg[t], f"Ist: Degree ({t}, undirected)", "degree", rw.out_dir / "plots" / f"ist_degree_{t}.png")
            if soll_deg.get(t):
                plot_hist(soll_deg[t], f"Soll: Degree ({t}, undirected)", "degree", rw.out_dir / "plots" / f"soll_degree_{t}.png")

        rw.p("Degree-Plots gespeichert unter `plots/` (ist_degree_*.png, soll_degree_*.png).")
    else:
        rw.p("Plots sind deaktiviert (--no-plots).")

    # Graph signatures (Nice-to-have 5)
    rw.h2("Graph-Signaturen (Struktur-Fingerprints)")

    def signatures_table(graphs: List[Dict[str, Any]], kind: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        sig_counter = Counter()
        sig_to_example: Dict[str, str] = {}
        sig_to_text: Dict[str, str] = {}
        for g in graphs:
            s = graph_signature(g)
            sid = s["signature_id"]
            sig_counter[sid] += 1
            sig_to_text.setdefault(sid, s["signature"])
            if sid not in sig_to_example:
                sig_to_example[sid] = str(g.get("graph_id"))
        top = []
        total = sum(sig_counter.values())
        for sid, cnt in sig_counter.most_common(25):
            top.append(
                {
                    "kind": kind,
                    "signature_id": sid,
                    "count": int(cnt),
                    "pct": f"{pct(cnt,total):.2f}%",
                    "example_graph_id": sig_to_example.get(sid, ""),
                    "signature": sig_to_text.get(sid, ""),
                }
            )
        meta = {"kind": kind, "unique_signatures": len(sig_counter), "graphs": total}
        return top, meta

    top_ist, meta_ist = signatures_table(ist_graphs, "ist")
    top_soll, meta_soll = signatures_table(soll_graphs, "soll")

    rw.bullet(
        [
            f"Ist: {meta_ist['graphs']} Graphen, {meta_ist['unique_signatures']} einzigartige Signaturen",
            f"Soll: {meta_soll['graphs']} Graphen, {meta_soll['unique_signatures']} einzigartige Signaturen",
            "Signaturen basieren auf Node-Typ-Counts, Edge-Rel-Counts, CC-Anzahl und Degree-Min/Median/Max.",
        ]
    )
    rw.h3("Top Signaturen (Ist)")
    rw.add_table_markdown(Table("signatures_top_ist", top_ist), max_rows=25)
    rw.h3("Top Signaturen (Soll)")
    rw.add_table_markdown(Table("signatures_top_soll", top_soll), max_rows=25)

    # Attribute completeness for key attrs
    rw.h2("Attribut-Vollständigkeit & Value-Distributionen")
    key_attrs_by_type = {
        "MaLo": ["direction", "direction_hint"],
        "MeLo": ["voltage_level", "function", "melo_function", "dynamic", "direction_hint"],
        "TR": ["tr_direction", "direction"],
        "NeLo": [],
    }

    def attr_coverage(graphs: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        totals = Counter()
        present = Counter()
        for g in graphs:
            for n in iter_nodes(g):
                ntype = str(n.get("type"))
                attrs = n.get("attrs", {}) or {}
                for a in key_attrs_by_type.get(ntype, []):
                    totals[(ntype, a)] += 1
                    if a in attrs and attrs.get(a) is not None and str(attrs.get(a)).strip() != "":
                        present[(ntype, a)] += 1
        rows = []
        for (ntype, a), tot in sorted(totals.items()):
            pre = present.get((ntype, a), 0)
            rows.append(
                {
                    "kind": kind,
                    "type": ntype,
                    "attr": a,
                    "present": int(pre),
                    "total": int(tot),
                    "present_pct": f"{pct(pre, tot):.1f}%",
                }
            )
        return rows

    rw.h3("Coverage: zentrale Attribute (Ist)")
    rw.add_table_markdown(Table("attr_coverage_ist", attr_coverage(ist_graphs, "ist")))

    rw.h3("Coverage: zentrale Attribute (Soll/Templates)")
    rw.add_table_markdown(Table("attr_coverage_soll", attr_coverage(soll_graphs, "soll")))

    def top_values(graphs: List[Dict[str, Any]], kind: str, ntype: str, attr: str, topn: int = 20) -> Table:
        c = Counter()
        for g in graphs:
            for n in iter_nodes(g):
                if str(n.get("type")) != ntype:
                    continue
                attrs = n.get("attrs", {}) or {}
                if attr in attrs and attrs.get(attr) is not None:
                    c[str(attrs.get(attr))] += 1
        rows = [{"kind": kind, "type": ntype, "attr": attr, "value": v, "count": int(cnt)} for v, cnt in c.most_common(topn)]
        return Table(f"top_values_{kind}_{ntype}_{attr}", rows)

    rw.h3("Top Values: Richtung (MaLo)")
    rw.add_table_markdown(top_values(ist_graphs, "ist", "MaLo", "direction"))
    rw.add_table_markdown(top_values(soll_graphs, "soll", "MaLo", "direction"))

    rw.h3("Top Values: Spannungsebene (MeLo.voltage_level)")
    rw.add_table_markdown(top_values(ist_graphs, "ist", "MeLo", "voltage_level"))
    rw.add_table_markdown(top_values(soll_graphs, "soll", "MeLo", "voltage_level"))

    rw.h3("Top Values: MeLo-Funktion (Templates sollten function/melo_function tragen)")
    rw.add_table_markdown(top_values(ist_graphs, "ist", "MeLo", "function"))
    rw.add_table_markdown(top_values(ist_graphs, "ist", "MeLo", "melo_function"))
    rw.add_table_markdown(top_values(soll_graphs, "soll", "MeLo", "function"))
    rw.add_table_markdown(top_values(soll_graphs, "soll", "MeLo", "melo_function"))

    rw.h3("Top Values: TR Richtung (tr_direction)")
    rw.add_table_markdown(top_values(ist_graphs, "ist", "TR", "tr_direction"))
    rw.add_table_markdown(top_values(soll_graphs, "soll", "TR", "tr_direction"))

    # Template-specific
    rw.h2("Template-spezifische Analyse (LBS-Schema)")

    # Topology checklist (Nice-to-have 6)
    rw.h3("Topologie-Checkliste: erwartete vs vorhandene Relationen")

    tmpl_rows = [topology_checklist_row(t, "soll") for t in soll_graphs]
    # sort: flagged first
    tmpl_rows.sort(key=lambda r: (not (r["missing_MEMA"] or r["missing_METR"] or r["missing_MENE"]), str(r.get("lbs_code") or "")))

    rw.add_table_markdown(Table("template_topology_checklist", tmpl_rows), max_rows=60)

    def checklist_summary(rows_: List[Dict[str, Any]], kind: str) -> List[str]:
        total = len(rows_)
        m_mema = sum(1 for r in rows_ if r.get("missing_MEMA"))
        m_metr = sum(1 for r in rows_ if r.get("missing_METR"))
        m_mene = sum(1 for r in rows_ if r.get("missing_MENE"))
        invalid_refs = sum(1 for r in rows_ if int(r.get("edges_missing_refs", 0)) > 0)
        return [
            f"{kind}: total={total}",
            f"{kind}: missing_MEMA={m_mema} (nur wenn MeLo+MaLo vorhanden)",
            f"{kind}: missing_METR={m_metr} (nur wenn MeLo+TR vorhanden)",
            f"{kind}: missing_MENE={m_mene} (nur wenn MeLo+NeLo vorhanden)",
            f"{kind}: graphs_with_invalid_edge_refs={invalid_refs}",
        ]

    rw.bullet(checklist_summary(tmpl_rows, "Templates"))

    # Ist: show only anomalies to keep report small
    ist_check = [topology_checklist_row(g, "ist") for g in ist_graphs]
    ist_anom = [
        r
        for r in ist_check
        if r.get("missing_MEMA")
        or r.get("missing_METR")
        or r.get("edges_missing_refs", 0)
        or r.get("edges_missing_srcdst", 0)
        or r.get("edges_missing_rel", 0)
    ]

    rw.h3("Topologie-Checkliste: Ist-Anomalien (Sample)")
    rw.bullet(checklist_summary(ist_check, "Ist"))
    ist_anom.sort(key=lambda r: (not (r.get("missing_MEMA") or r.get("missing_METR")), int(r.get("edges_missing_refs", 0)), str(r.get("graph_id") or "")))
    rw.add_table_markdown(Table("ist_topology_anomalies", ist_anom), max_rows=40)

    # Optionality summary (existing)
    opt_rows: List[Dict[str, Any]] = []
    for t in soll_graphs:
        ga = t.get("graph_attrs", {}) or {}
        code = ga.get("lbs_code") or t.get("label") or t.get("graph_id")
        optional_nodes = 0
        flexible_nodes = 0
        min0_nodes = 0
        total_nodes = 0
        per_type = Counter()
        for n in iter_nodes(t):
            total_nodes += 1
            ntype = str(n.get("type"))
            per_type[ntype] += 1
            attrs = n.get("attrs", {}) or {}
            mn = attrs.get("min_occurs")
            flex = attrs.get("flexibility")
            opt = attrs.get("optional")
            if opt is True:
                optional_nodes += 1
            if flex is True:
                flexible_nodes += 1
            if mn is not None and _as_int_or_none(mn) == 0:
                min0_nodes += 1
        opt_rows.append(
            {
                "lbs_code": code,
                "graph_id": t.get("graph_id"),
                "nodes_total": total_nodes,
                "MaLo_nodes": int(per_type.get("MaLo", 0)),
                "MeLo_nodes": int(per_type.get("MeLo", 0)),
                "TR_nodes": int(per_type.get("TR", 0)),
                "NeLo_nodes": int(per_type.get("NeLo", 0)),
                "optional_nodes": optional_nodes,
                "min0_nodes": min0_nodes,
                "flexible_nodes": flexible_nodes,
                "attachment_rules_n": len(ga.get("attachment_rules") or []),
                "constraints_n": len(ga.get("optionality_constraints") or []),
            }
        )
    opt_rows.sort(key=lambda r: str(r["lbs_code"]))
    rw.add_table_markdown(Table("template_optionality_summary", opt_rows), max_rows=60)

    # Coverage: bounds
    rw.h2("Ist vs Soll: Bounds-Coverage (Kandidatenfilter ohne pattern)")
    cov = ist_vs_template_coverage(ist_graphs, soll_graphs)
    rw.p("Diese Tabelle zeigt pro Template, wie viele Ist-Graphen die Min/Max-Bounds erfüllen.")
    rw.add_table_markdown(Table("template_coverage", cov), max_rows=60)

    # Encoder alignment
    rw.h2("Abgleich mit Feature-Encoding (graph_pipeline.py)")
    align_ist = encoder_alignment(ist_graphs, "ist")
    align_soll = encoder_alignment(soll_graphs, "soll")

    def alignment_table(al: Dict[str, Any]) -> List[Dict[str, Any]]:
        c = al["counts"]
        u = al["unknown"]
        rows = []
        rows.append({"kind": al["kind"], "metric": "nodes_total", "count": int(c.get("nodes_total", 0))})
        rows.append({"kind": al["kind"], "metric": "edges_total(valid typed)", "count": int(c.get("edges_total", 0))})
        dir_tot = int(c.get("dir_nodes_total", 0))
        fn_tot = int(c.get("melo_fn_nodes_total", 0))
        volt_tot = int(c.get("volt_nodes_total", 0))
        rows.append(
            {
                "kind": al["kind"],
                "metric": "dir_unknown_rate",
                "count": f"{fmt_pct(u.get('dir_unknown',0), dir_tot)} ({u.get('dir_unknown',0)}/{dir_tot})",
            }
        )
        rows.append(
            {
                "kind": al["kind"],
                "metric": "melo_fn_unknown_rate",
                "count": f"{fmt_pct(u.get('melo_fn_unknown',0), fn_tot)} ({u.get('melo_fn_unknown',0)}/{fn_tot})",
            }
        )
        rows.append(
            {
                "kind": al["kind"],
                "metric": "volt_unknown_rate",
                "count": f"{fmt_pct(u.get('volt_unknown',0), volt_tot)} ({u.get('volt_unknown',0)}/{volt_tot})",
            }
        )
        return rows

    rw.h3("Unknown-Raten für die verwendeten Feature-Blöcke")
    rw.add_table_markdown(Table("encoder_unknown_rates", alignment_table(align_ist) + alignment_table(align_soll)))

    rw.h3("Welche Felder gehen aktuell ins Modell ein?")
    rw.p(
        "Der aktuelle Feature-Encoder (graph_pipeline.py) reduziert die reichhaltigen Node-Attrs "
        "auf wenige One-Hot-Blöcke. Das ist wichtig, um zu beurteilen, ob im Graph unnötige Daten "
        "mitgeschleppt werden oder ob entscheidende Informationen fehlen."
    )
    rw.bullet(
        [
            "**Knotentyp**: `node['type']` → One-Hot über {MaLo, MeLo, TR, NeLo}",
            "**Richtung** (nur MaLo/TR): `attrs['direction']` bzw. TR: `attrs['tr_direction']`; Fallback MaLo: `attrs['direction_hint']` → {consumption, generation, both, unknown}",
            "**MeLo-Funktion** (nur MeLo): `attrs['function']` oder `attrs['melo_function']` → {N, H, D, S, unknown}",
            "**Spannungsebene** (nur MeLo): `attrs['voltage_level']` → {E05, E06, unknown}",
            "**Kantentyp**: `edge['rel']` → One-Hot über {MEMA, METR, MENE, MEME, unknown}",
            "Alle anderen Node-Attribute (z.B. `min_occurs/max_occurs/flexibility/optional/object_code/level/dynamic/attachment_rules`) werden aktuell **nicht** als ML-Feature kodiert.",
        ]
    )

    used_attr_keys = {"direction", "tr_direction", "direction_hint", "function", "melo_function", "voltage_level"}
    ist_unused = [k for k, _ in ist_node_attr_keys.most_common() if k not in used_attr_keys]
    soll_unused = [k for k, _ in soll_node_attr_keys.most_common() if k not in used_attr_keys]

    rw.h3("Welche Node-Attribute sind im Datensatz vorhanden, werden aber vom Encoder ignoriert?")

    def unused_table(name: str, keys: List[str], counter: Counter) -> Table:
        rows = []
        for k in keys[:60]:
            rows.append({"key": k, "count": int(counter.get(k, 0))})
        return Table(name, rows)

    rw.p("Ist-Graphen (Top ignorierte Keys):")
    rw.add_table_markdown(unused_table("ist_unused_node_attrs", ist_unused, ist_node_attr_keys))

    rw.p("Template-Graphen (Top ignorierte Keys):")
    rw.add_table_markdown(unused_table("soll_unused_node_attrs", soll_unused, soll_node_attr_keys))

    # Highlight key differences automatically
    rw.h2("Automatische Beobachtungen (Differenzen & mögliche Ursachen)")
    observations: List[str] = []

    ist_types = set(ist_nt.keys())
    soll_types = set(soll_nt.keys())
    only_in_soll = sorted(soll_types - ist_types)
    only_in_ist = sorted(ist_types - soll_types)
    if only_in_soll:
        observations.append(
            f"Templates enthalten Knotentyp(e) {only_in_soll}, die in Ist-Graphen nicht vorkommen (typisch: NeLo). Ursache: Ist-Converter modelliert diese Relationen/Knoten derzeit nicht."
        )
    if only_in_ist:
        observations.append(
            f"Ist-Graphen enthalten Knotentyp(e) {only_in_ist}, die in Templates nicht vorkommen (prüfen: SAP-Artefakte oder Export-Drift)."
        )

    ist_rels = set(ist_et.keys())
    soll_rels = set(soll_et.keys())
    rel_only_soll = sorted((soll_rels - ist_rels) & {"MENE", "MEME"})
    if rel_only_soll:
        observations.append(f"Templates enthalten Kanten-Typ(e) {rel_only_soll}, die in Ist-Graphen fehlen. Ursache: Ist-Converter baut aktuell nur MEMA/METR.")
    rel_only_ist = sorted(ist_rels - soll_rels)
    if rel_only_ist:
        observations.append(f"Ist-Graphen enthalten Kanten-Typ(e) {rel_only_ist}, die Templates nicht enthalten (prüfen, ob diese in Soll bewusst weggelassen wurden).")

    u_ist = align_ist["unknown"]
    c_ist = align_ist["counts"]
    u_soll = align_soll["unknown"]
    c_soll = align_soll["counts"]

    if c_ist.get("melo_fn_nodes_total", 0) and pct(u_ist.get("melo_fn_unknown", 0), c_ist.get("melo_fn_nodes_total", 0)) > 80:
        observations.append(
            "MeLo-Funktion ist in Ist-Graphen fast immer 'unknown' (Feature-Block wird vom Modell kaum genutzt). Ursache: Ist-Graphen tragen i.d.R. kein function/melo_function."
        )
    if c_soll.get("volt_nodes_total", 0) and pct(u_soll.get("volt_unknown", 0), c_soll.get("volt_nodes_total", 0)) > 80:
        observations.append(
            "Spannungsebene ist in Template-Graphen fast immer 'unknown' (Feature-Block wird vom Modell kaum genutzt). Ursache: Templates enthalten i.d.R. keine voltage_level-Angabe."
        )
    if c_soll.get("dir_nodes_total", 0) and pct(u_soll.get("dir_unknown", 0), c_soll.get("dir_nodes_total", 0)) > 80:
        observations.append("Richtung (MaLo/TR) ist in Templates oft 'unknown' → Richtung als Feature unterscheidet Templates ggf. kaum.")

    # new: topology checklist highlights
    tmpl_missing_mema = sum(1 for r in tmpl_rows if r.get("missing_MEMA"))
    tmpl_missing_metr = sum(1 for r in tmpl_rows if r.get("missing_METR"))
    if tmpl_missing_mema or tmpl_missing_metr:
        observations.append(
            f"Topologie-Check: {tmpl_missing_mema} Templates fehlen MEMA obwohl MeLo+MaLo vorhanden, {tmpl_missing_metr} Templates fehlen METR obwohl MeLo+TR vorhanden. Prüfen: Absicht (attachment_rules/Unklarheit) vs. Mapping-Lücke."
        )

    if observations:
        rw.bullet(observations)
    else:
        rw.p("_Keine auffälligen Differenzen automatisch erkannt (ungewöhnlich)._")

    report_path = rw.finalize()
    return report_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ist", type=str, default="data/ist_graphs_all.jsonl", help="Pfad zu Ist-Graphen JSONL")
    ap.add_argument("--soll", type=str, default="data/lbs_soll_graphs_pro.jsonl", help="Pfad zu Soll-Template JSONL")
    ap.add_argument("--out_dir", type=str, default="analysis/descriptive", help="Output-Verzeichnis")
    ap.add_argument("--max_graphs", type=int, default=None, help="Begrenze Anzahl Graphen (Debug)")
    ap.add_argument("--no-plots", action="store_true", help="Deaktiviere Plots")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_analysis(
        ist_path=Path(args.ist),
        soll_path=Path(args.soll),
        out_dir=Path(args.out_dir),
        max_graphs=args.max_graphs,
        do_plots=not args.no_plots,
    )

    print(f"[OK] Report geschrieben: {report}")
