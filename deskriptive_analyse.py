"""
Deskriptive Analyse für Ist-Graphen (Instanzen) und Template-Graphen (Schema)

In Thesis erwähnen, nicht vergessen:
- Was ist drin? Wie oft? In welcher Form?
- Wo/Wie unterscheiden sich Ist und Soll?
- Was nutzt das Modell tatsächlich?
- Fehlt etwas? Ist etwas unbenutzt?

Eingabe:
Jede Zeile ist ein Graph als JSON-Dict mit mindestens:
{
  "graph_id": "...",
  "nodes": [{"id": "...", "type": "MaLo|MeLo|TR|NeLo", "attrs": {...}}, ...],
  "edges": [{"src": "...", "dst": "...", "rel": "MEMA|METR|MENE|MEME|..."}, ...],
  "graph_attrs": {...}
}
Templates können "label" haben
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

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

gp = None
try:
    import graph_pipeline as gp
except Exception:
    try:
        import sys

        this_dir = str(Path(__file__).resolve().parent)
        if this_dir not in sys.path:
            sys.path.insert(0, this_dir)
        import graph_pipeline as gp
    except Exception:
        gp = None


# -------------------------
# DATEIEN UND RECHNEN
# -------------------------

def _file_reader(pfad: Path) -> str:
    """
    Liest Dateien ein und probiert verschiedene Encodings.

    :param pfad: Pfad zur Datei
    :return str: Datei (richtiges Encoding)
    """
    rohdaten = pfad.read_bytes()
    try:
        return rohdaten.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return rohdaten.decode("cp1252")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Datei scheint weder UTF-8 noch cp1252 zu sein: {pfad}",
            ) from e


def jsonl_loader(path: Path, obergrenze: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load JSONL file into list of dicts. Reads ALL lines by default (no sampling).

    :param path: Pfad zur Datei
    :param obergrenze: Maximale Anzahl an Graphen, die eingelesen werden sollen (optional)
    """
    text = _file_reader(path)
    # Graphen sind Dict, also für jeden Eintrag einmal eins erstellen
    graphen: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if obergrenze is not None and len(graphen) >= obergrenze:
            break
        line = line.strip()
        if not line:
            continue
        # Zeilen parsen
        try:
            zeilen = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON dekodieren lief schief: {path} Zeile {i}: {e}") from e
        graphen.append(zeilen)
    return graphen


def analysis_directory(verzeichnis: Path) -> None:
    """
    Verzeichnis und ggf. fehlende Elternverzeichnisse erstellen

    :param verzeichnis: Beschreibt VErzeichnis
    """
    verzeichnis.mkdir(parents=True, exist_ok=True)


def safe_div(zaehler: float, nenner: float) -> float:
    """
    Normale Division mit Nullfehler
    """
    return zaehler / nenner if nenner else float("NaN")


def prozentrechner(z1: float, z2: float) -> float:
    """
    Prozentsatz ausrechnen
    :return: Prozentsatz wenn z2 > 0, sonst 0.0
    """
    return 100.0 * safe_div(z1, z2) if z2 else 0.0


def prozent_nachkomma(z1: float, z2: float, nachkomma: int = 1) -> str:
    """
    Für Nachkommastelle
    """
    return f"{prozentrechner(z1, z2):.{nachkomma}f}%"


def infinity(z: Optional[int]) -> str:
    """
    0..N ist unbegrenzt, also lesbare Darstellung für die deskriptive Analyse (unendlich)
    """
    return "∞" if z is None else str(z)


def quantiles(values: Sequence[float], q_pos: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> Dict[str, float]:
    """
    Nach Quantilen analysieren, mit Median "typischer" Graph

    :param values: Sequenz von Zahlen
    :param q_pos: Quantilpositionen zwischen 0 und 1 - Minimum, Q1, Median, Q3, Maximum
    :return: Dictionary aus besagten Quantilen
    """
    # Auf Werte überprüfen
    if not values:
        return {f"q{int(q*100)}": float("NaN") for q in q_pos}
    s = sorted(values)
    erg: Dict[str, float] = {}
    for q in q_pos:
        # Zum nächsten Index runden
        idx = int(round(q * (len(s) - 1)))
        erg[f"q{int(q*100)}"] = float(s[idx])
    return erg


@dataclass
class Table:
    """
    Tabellenstruktur
    """
    tabelle_name: str
    tabelle_zeilen: List[Dict[str, Any]]


class ReportWriter:
    """
    Ausgabe des Berichts
    """
    def __init__(self, out_dir: Path) -> None:
        """
        Initialisiert Ausgabeverzeichnis
        :param out_dir: Pfad der Ausgabe
        """
        self.out_dir = out_dir
        # Unterordner
        analysis_directory(out_dir)
        analysis_directory(out_dir / "tables")
        analysis_directory(out_dir / "plots")
        self.lines: List[str] = []

    """
    Überschriften in gängigen Formaten
    """
    def h1(self, title: str) -> None:
        self.lines.append(f"# {title}\n")

    def h2(self, title: str) -> None:
        self.lines.append(f"## {title}\n")

    def h3(self, title: str) -> None:
        self.lines.append(f"### {title}\n")

    def text(self, text: str) -> None:
        """
        Absatz
        """
        self.lines.append(text.strip() + "\n")

    def bullet(self, items: Sequence[str]) -> None:
        """
        Aufzählung
        :param items: Bullet-Strings
        """
        for item in items:
            self.lines.append(f"- {item}")
        self.lines.append("")

    def codeblock(self, code: str, lang: str = "") -> None:
        """
        Formatierter Code
        :param code: Code
        :param lang: Programmiersprache
        """
        self.lines.append(f"```Geschrieben in: {lang}")
        self.lines.append(code.rstrip())
        self.lines.append("```")
        self.lines.append("")

    def csv_generator(self, table: Table) -> Path:
        out = self.out_dir / "tables" / f"{table.tabelle_name}.csv"
        if not table.tabelle_zeilen:
            out.write_text("", encoding="utf-8")
            return out
        # Spaltenreihenfolge bestimmen
        # Schlüssel über alle Zeilen
        schluessel = set().union(*(r.keys() for r in table.tabelle_zeilen))
        bevorzugte_reihenfolge = [ #TODO - pct vorher umbenannt
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
        spalten = [r for r in bevorzugte_reihenfolge if r in schluessel] + sorted([s for s in schluessel if s not in bevorzugte_reihenfolge])
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=spalten)
            w.writeheader()
            for r in table.tabelle_zeilen:
                w.writerow({s: r.get(s, "") for s in spalten})
        return out

    def insert_table(self, table: Table, max_rows: int = 40) -> None:
        """
        Fügt Tabelle in den Report ein und schreibt dazu die volle CSV

        :param max_rows: maximale Zeilen, die im Report sein sollen (Rest in jeweiliger CSV)
        """
        csv_path = self.csv_generator(table)
        self.text(f"Full table stored in: `{csv_path.as_posix()}`")
        # Eingrenzung
        zeilen = table.tabelle_zeilen[:max_rows]
        if not zeilen:
            self.text("_No data._")
            return
        if pd is not None:
            # Baut DataFrame
            df = pd.DataFrame(zeilen)
            self.lines.append(df.to_markdown(index=False))
            self.lines.append("")
        else:
            spalten = list(zeilen[0].keys())
            self.lines.append("| " + " | ".join(spalten) + " |")
            self.lines.append("| " + " | ".join(["---"] * len(spalten)) + " |")
            for r in zeilen:
                self.lines.append("| " + " | ".join(str(r.get(c, "")) for c in spalten) + " |")
            self.lines.append("")
        if len(table.tabelle_zeilen) > max_rows:
            self.text(f"_Table has been shortened to {max_rows} rows (originally {len(table.tabelle_zeilen)})._")

    def save_plot(self, fig, filename: str) -> Optional[Path]:
        """
        Speichert einen Plot

        :param fig: Matplotlib Figur
        :param filename: Dateiname
        """

        if plt is None:
            return None
        out = self.out_dir / "plots" / filename
        fig.savefig(out, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return out

    def finalize(self) -> Path:
        """
        Schreibt den Report als Markdown ins Ausgabeverzeichnis
        """

        out = self.out_dir / "report.md"
        out.write_text("\n".join(self.lines), encoding="utf-8")
        return out


# -------------------------
# GETTER UND VALIDIERER
# -------------------------

def get_nodes(graph: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Geht über alle Knoten eines Graphen und gibt sie zurück
    """
    for n in graph.get("nodes", []) or []:
        if isinstance(n, dict):
            yield n


def get_edges(graph: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Geht über alle Kanten eines Graphen und gibt sie zurück
    """
    for e in graph.get("edges", []) or []:
        if isinstance(e, dict):
            yield e


def set_of_node_ids(graph: Dict[str, Any]) -> set:
    """
    Menge aller Node-IDs als String
    """
    # Einzigartige IDs
    return {str(n.get("id")) for n in get_nodes(graph) if n.get("id") is not None}


def _get_edge_relations(edge: Dict[str, Any]) -> Optional[str]:
    """
    Gibt Relationstypen einer Kante
    """
    # Verschiedene Key-Namen
    rel = edge.get("rel") or edge.get("type") or edge.get("edge_type") or edge.get("relation")
    if rel is None:
        return None
    s = str(rel).strip()
    return s if s else None


def edge_validator(graph: Dict[str, Any]) -> Dict[str, int]:
    """
    Zählt valide Kanten pro Graph
    """
    ids = set_of_node_ids(graph)

    total = 0               # alle Kanten, auch wenn kaputt
    with_src_and_dst = 0    # Kanten mit source und destination
    with_rel = 0            # Kanten mit Relationstyp

    valid_endpoints = 0     # Kanten, deren src und dst in ids existieren
    valid_typed = 0         # Kanten mit valid_endpoints und Relationstyp

    # Fehlende Werte (analog)
    missing_src_or_dst = 0
    missing_rel = 0
    missing_refs = 0
    self_loops_valid = 0    # valide Selfloops im src=dst Fall

    for e in get_edges(graph):
        total += 1
        s = e.get("src")
        t = e.get("dst")
        rel = _get_edge_relations(e)

        if s is None or t is None:
            missing_src_or_dst += 1
            continue
        with_src_and_dst += 1

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
        "edges_with_srcdst": with_src_and_dst,
        "edges_with_rel": with_rel,
        "edges_valid_endpoints": valid_endpoints,
        "edges_valid_typed": valid_typed,
        "edges_missing_srcdst": missing_src_or_dst,
        "edges_missing_rel": missing_rel,
        "edges_missing_refs": missing_refs,
        "edges_self_loops_valid": self_loops_valid,
    }


def get_valid_structured_edges(graph: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """
    Alle validen Kanten (mit src und dst)
    """
    ids = set_of_node_ids(graph)
    for e in get_edges(graph):
        s = e.get("src")
        t = e.get("dst")
        if s is None or t is None:
            continue
        s = str(s)
        t = str(t)
        if s not in ids or t not in ids:
            continue
        yield s, t


def get_valid_typed_edges(graph: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    """
    Alle validen Kanten (mit src und dst) umit rel
    """
    ids = set_of_node_ids(graph)
    for e in get_edges(graph):
        s = e.get("src")
        t = e.get("dst")
        if s is None or t is None:
            continue
        rel = _get_edge_relations(e)
        if rel is None:
            continue
        s = str(s)
        t = str(t)
        if s not in ids or t not in ids:
            continue
        yield s, t, rel.strip().upper()


def undirected_neighbor_builder(graph: Dict[str, Any]) -> Dict[str, set]:
    """
    Baut Set aus ungerichteten, einzigartigen Nachbarn (nur valide Kanten), ohne self-loop!
    """
    ids = set_of_node_ids(graph)
    nachbarn: Dict[str, set] = {nachbar_id: set() for nachbar_id in ids}
    for s, t in get_valid_structured_edges(graph):
        if s == t:
            continue
        nachbarn[s].add(t)
        nachbarn[t].add(s)
    return nachbarn


def directed_neighbor_builder(graph: Dict[str, Any]) -> Tuple[Dict[str, set], Dict[str, set]]:
    """
    Baut Set aus gerichteten, einzigartigen Nachbran (ohne self-loop!)
    """
    ids = set_of_node_ids(graph)
    ausgehend: Dict[str, set] = {nachbar_id: set() for nachbar_id in ids}
    eingehend: Dict[str, set] = {nachbar_id: set() for nachbar_id in ids}
    for s, t in get_valid_structured_edges(graph):
        if s == t:
            continue
        ausgehend[s].add(t)
        eingehend[t].add(s)
    return ausgehend, eingehend


def connected_components_sizes(graph: Dict[str, Any]) -> List[int]:
    """
    Berechnet Größe aller zusammenhängenden Komponenten (ungerichteter Graph, valide Kanten)
    Prüft, ob Graphen in einem Stück sind
    Wir erwarten für LBS zusammenhängende Strukturen
    """
    nachbarn = undirected_neighbor_builder(graph)
    angesehen = set()
    component_sizes: List[int] = []
    for start in nachbarn.keys():
        if start in angesehen:
            continue
        # Mit Stack
        stack = [start]
        angesehen.add(start)
        size = 0
        while stack:
            n1 = stack.pop()
            size += 1
            for n2 in nachbarn.get(n1, set()):
                if n2 not in angesehen:
                    angesehen.add(n2)
                    stack.append(n2)
        component_sizes.append(size)
    component_sizes.sort(reverse=True)
    return component_sizes


def strongly_connected_components_sizes(graph: Dict[str, Any]) -> List[int]:
    """
    Mximale Teilmenge von Knoten in einem gerichteten Graphen, bei der jeder Knoten
    von jedem anderen Knoten innerhalb dieser Teilmenge direkt oder indirekt über gerichtete
    Kanten erreichbar ist
    """
    ausgehend, eingehend = directed_neighbor_builder(graph)

    besucht = set()
    reihenfolge: List[str] = []

    # Tiefensuche durchführen
    def dfs1(u: str) -> None:
        besucht.add(u)
        for v in ausgehend.get(u, set()):
            if v not in besucht:
                dfs1(v)
        reihenfolge.append(u)

    for u in ausgehend.keys():
        if u not in besucht:
            dfs1(u)

    besucht_2 = set()
    sizes: List[int] = []

    def dfs2(u: str) -> int:
        besucht_2.add(u)
        size = 1
        for v in eingehend.get(u, set()):
            if v not in besucht_2:
                size += dfs2(v)
        return size

    # in umgekehrter Reihenfolge, um Prüfung zu finalisieren
    for u in reversed(reihenfolge):
        if u not in besucht_2:
            sizes.append(dfs2(u))

    sizes.sort(reverse=True)
    return sizes


def generate_graph_metrics(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Graph-Metriken erzeugen: Valide, Größe, Komponenten, SCC, ...
    """
    ids = set_of_node_ids(graph)
    nodes = len(ids)

    ev = edge_validator(graph)

    # Berechnung ungerichteter Grade
    nachbarn = undirected_neighbor_builder(graph)
    degrees = [len(nachbarn.get(nachbar_id, set())) for nachbar_id in ids]
    isolierte_nodes = sum(1 for d in degrees if d == 0)

    # Berechnung ungerichteter Kanten (einzigartig, valide)
    undirected_pairs = set()
    for s, t in get_valid_structured_edges(graph):
        # Self-Loops
        if s == t:
            continue
        n1, n2 = (s, t) if s <= t else (t, s)
        undirected_pairs.add((n1, n2))
    ungerichtet_anzahl = len(undirected_pairs)

    # Dichte ("wie voll": maximal mögliche Anzahl vs. tatsächliche Anzahl)
    dichte = (2.0 * ungerichtet_anzahl / (nodes * (nodes - 1))) if nodes > 1 else 0.0

    components = connected_components_sizes(graph)
    sc_components = strongly_connected_components_sizes(graph)

    ausgehend, eingehend = directed_neighbor_builder(graph)
    ausgehend_degree = [len(ausgehend.get(nachbar_id, set())) for nachbar_id in ids]
    eingehen_degree = [len(eingehend.get(nachbar_id, set())) for nachbar_id in ids]

    ungerichtet_degree_quantile = quantiles([float(d) for d in degrees])

    return {
        # Zähler direkt in die Ergebnisstruktur "entpacken"
        **ev,
        # Kerngrößen
        "node_count_unique": nodes,
        "edge_pairs_undirected_unique": ungerichtet_anzahl,
        # Ungerichtet - Struktur
        "density_undirected_simple": dichte,
        "edges_per_node_valid_endpoints": safe_div(float(ev["edges_valid_endpoints"]), float(nodes)),
        "edges_per_node_valid_typed": safe_div(float(ev["edges_valid_typed"]), float(nodes)),
        "avg_degree_undirected": float(statistics.mean(degrees)) if degrees else float("nan"),
        "deg_min_undirected": int(min(degrees)) if degrees else None,
        "deg_med_undirected": int(ungerichtet_degree_quantile["q50"]) if degrees else None,
        "deg_max_undirected": int(max(degrees)) if degrees else None,
        "isolated_nodes_undirected": int(isolierte_nodes),
        "isolated_pct_undirected": prozentrechner(float(isolierte_nodes), float(nodes)),
        "cc_count_undirected": int(len(components)),
        "cc_largest_undirected": int(components[0]) if components else 0,
        # Gerichtet
        "avg_out_degree_directed": float(statistics.mean(ausgehend_degree)) if ausgehend_degree else float("nan"),
        "avg_in_degree_directed": float(statistics.mean(eingehen_degree)) if eingehen_degree else float("nan"),
        "out_degree_max_directed": int(max(ausgehend_degree)) if ausgehend_degree else 0,
        "in_degree_max_directed": int(max(eingehen_degree)) if eingehen_degree else 0,
        "scc_count_directed": int(len(sc_components)),
        "scc_largest_directed": int(sc_components[0]) if sc_components else 0,
    }

def degree_per_typ(graph: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Ungerichtete Gradfolgen, per Nodetyp
    """
    nachbarn = undirected_neighbor_builder(graph)
    id_to_type: Dict[str, str] = {}
    for n in get_nodes(graph):
        nachbar_id = n.get("id")
        if nachbar_id is None:
            continue
        id_to_type[str(nachbar_id)] = str(n.get("type"))

    out: Dict[str, List[int]] = defaultdict(list)
    for nachbar_id, neis in nachbarn.items():
        out[id_to_type.get(nachbar_id, "UNKNOWN")].append(len(neis))
    return out


def gerichtet_degree_per_typ(graph: Dict[str, Any]) -> Dict[str, Dict[str, List[int]]]:
    """
    Gerichtete GRadfolgen per Nodetyp
    """
    ausgehend, eingehend = directed_neighbor_builder(graph)
    id_to_type: Dict[str, str] = {}
    for n in get_nodes(graph):
        nachbar_id = n.get("id")
        if nachbar_id is None:
            continue
        id_to_type[str(nachbar_id)] = str(n.get("type"))

    out: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"out": [], "in": []})
    for nachbar_id in ausgehend.keys():
        t = id_to_type.get(nachbar_id, "UNKNOWN")
        out[t]["out"].append(len(ausgehend.get(nachbar_id, set())))
        out[t]["in"].append(len(eingehend.get(nachbar_id, set())))
    return out


def graph_signature(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Erzeugt eine Graph-Signatur
    """
    #Nodetypen pro ID
    ids = set_of_node_ids(graph)
    id_to_type: Dict[str, str] = {}
    for n in get_nodes(graph):
        nachbar_id = n.get("id")
        if nachbar_id is None:
            continue
        id_to_type[str(nachbar_id)] = str(n.get("type"))
    node_typen = Counter(id_to_type.get(nid, "UNKNOWN") for nid in ids)

    #Valide Edge Typen
    edge_typen = Counter(rel for _, _, rel in get_valid_typed_edges(graph))

    #Degrees
    nachbarn = undirected_neighbor_builder(graph)
    degrees = [len(nachbarn.get(nid, set())) for nid in ids]
    dq = quantiles([float(d) for d in degrees])
    comps = connected_components_sizes(graph)

    #Anzahl
    node_part = ",".join([f"{t}={int(node_typen.get(t,0))}" for t in ["MaLo", "MeLo", "TR", "NeLo", "UNKNOWN"] if t in node_typen or t != "UNKNOWN"])
    edge_part = ",".join([f"{r}={int(edge_typen.get(r,0))}" for r in ["MEMA", "METR", "MENE", "MEME"]])

    signatur_txt = f"NT:{node_part}|ET:{edge_part}|CC:{len(comps)}|DEG:{int(dq['q0']) if degrees else 0},{int(dq['q50']) if degrees else 0},{int(dq['q100']) if degrees else 0}"
    signatur_aus_hash = hashlib.md5(signatur_txt.encode("utf-8")).hexdigest()[:10]

    return {
        "signature": signatur_txt,
        "signature_id": signatur_aus_hash,
        "node_types": dict(node_typen),
        "edge_types": dict(edge_typen),
        "cc_count": len(comps),
        "deg_min": int(dq["q0"]) if degrees else 0,
        "deg_med": int(dq["q50"]) if degrees else 0,
        "deg_max": int(dq["q100"]) if degrees else 0,
    }


# -------------------------
# VALIDIERUNG
# -------------------------


@dataclass
class ValidationIssue:
    schwere_grad: str  # "ERROR" | "WARN"
    problem_bei: str  # graph_id or context
    message: str


def graph_validator(graph: Dict[str, Any], art: str) -> List[ValidationIssue]:
    probleme: List[ValidationIssue] = []
    graph_id = str(graph.get("graph_id", "<no-graph_id>"))

    for k in ["graph_id", "nodes", "edges", "graph_attrs"]:
        if k not in graph:
            probleme.append(ValidationIssue("ERROR", graph_id, f"Missing top-level key: {k}"))

    #Node IDS
    ids: List[str] = []
    for n in get_nodes(graph):
        node_id = n.get("id")
        if node_id is None:
            probleme.append(ValidationIssue("ERROR", graph_id, "Node without id"))
            continue
        ids.append(str(node_id))
    if len(ids) != len(set(ids)):
        dup = [k for k, c in Counter(ids).items() if c > 1][:10]
        probleme.append(ValidationIssue("ERROR", graph_id, f"Duplicate node ids (sample): {dup}"))

    idset = set(ids)

    # Edges
    fehlende_references = 0
    fehlende_felder = 0
    for e in get_edges(graph):
        s = e.get("src")
        t = e.get("dst")
        rel = _get_edge_relations(e)
        if s is None or t is None or rel is None:
            fehlende_felder += 1
            continue
        if str(s) not in idset or str(t) not in idset:
            fehlende_references += 1

    if fehlende_felder:
        probleme.append(ValidationIssue("ERROR", graph_id, f"{fehlende_felder} edges missing src/dst/rel"))
    if fehlende_references:
        probleme.append(ValidationIssue("ERROR", graph_id, f"{fehlende_references} edges reference missing node ids"))

    # Unbekannte Typen
    valide_node_types = {"MaLo", "MeLo", "TR", "NeLo"}
    unbekannte_node_types = Counter()
    for n in get_nodes(graph):
        t = str(n.get("type"))
        if t not in valide_node_types:
            unbekannte_node_types[t] += 1
    if unbekannte_node_types:
        probleme.append(ValidationIssue("WARN", graph_id, f"Unknown node types: {dict(unbekannte_node_types)}"))

    valide_edge_types = {"MEMA", "METR", "MENE", "MEME"}
    unbekannte_edge_types = Counter()
    for _, _, r in get_valid_typed_edges(graph):
        if r not in valide_edge_types:
            unbekannte_edge_types[r] += 1
    if unbekannte_edge_types:
        probleme.append(ValidationIssue("WARN", graph_id, f"Unknown edge rel types: {dict(unbekannte_edge_types)}"))

    # Konsistente Graphen in Ist bzgl. Attribute
    graph_attribute = graph.get("graph_attrs", {}) or {}
    if art == "ist":
        actual = Counter(str(n.get("type")) for n in get_nodes(graph))
        for t, key in [("MaLo", "malo_count"), ("MeLo", "melo_count"), ("TR", "tr_count"), ("NeLo", "nelo_count")]:
            if key in graph_attribute:
                try:
                    declared = int(graph_attribute[key])
                except Exception:
                    probleme.append(ValidationIssue("WARN", graph_id, f"graph_attrs.{key} not int: {graph_attribute[key]}"))
                    continue
                if declared != int(actual.get(t, 0)):
                    probleme.append(
                        ValidationIssue(
                            "WARN", graph_id, f"Count mismatch {key}: declared={declared} actual={actual.get(t,0)}"
                        )
                    )

    # Template Bounds prüfen - soll
    if art == "soll":
        for t in ["malo", "melo", "tr", "nelo"]:
            mn = graph_attribute.get(f"{t}_min")
            mx = graph_attribute.get(f"{t}_max")
            if mn is None or mx is None:
                continue
            try:
                mn_i = int(mn)
            except Exception:
                probleme.append(ValidationIssue("WARN", graph_id, f"{t}_min not int: {mn}"))
                continue
            if mx is not None:
                try:
                    mx_i = int(mx)
                    if mx_i < mn_i:
                        probleme.append(ValidationIssue("ERROR", graph_id, f"Invalid bounds: {t}_max({mx_i}) < {t}_min({mn_i})"))
                except Exception:
                    probleme.append(ValidationIssue("WARN", graph_id, f"{t}_max not int/None: {mx}"))

    return probleme

def keyset_extrahieren(graphs: List[Dict[str, Any]]) -> Tuple[Counter, Counter, Counter]:
    """
    Gibt graph_attrs keys, node attrs keys, edge dict keys zurück
    """
    grattibute_keys = Counter()
    node_attribute_keys = Counter()
    edge_keys = Counter()

    for g in graphs:
        graph_attributes = g.get("graph_attrs", {}) or {}
        for k in graph_attributes.keys():
            grattibute_keys[str(k)] += 1
        for n in get_nodes(g):
            attrs = n.get("attrs", {}) or {}
            for k in attrs.keys():
                node_attribute_keys[str(k)] += 1
        for e in get_edges(g):
            for k in e.keys():
                edge_keys[str(k)] += 1
    return grattibute_keys, node_attribute_keys, edge_keys


def node_type_counts(graphs: List[Dict[str, Any]]) -> Counter:
    """
    Zählt Node Typen
    """
    c = Counter()
    for g in graphs:
        for n in get_nodes(g):
            c[str(n.get("type"))] += 1
    return c


def edge_type_counts(graphs: List[Dict[str, Any]]) -> Counter:
    """
    Zählt Edge Typen
    """
    c = Counter()
    for g in graphs:
        for _, _, rel in get_valid_typed_edges(g):
            c[str(rel)] += 1
    return c


def dataframe_per_graph(graphs: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    """
    Baut eine Liste von Dict. pro Graph, die danach als DataFrame weiterverarbeitet werden
    """
    zeilen: List[Dict[str, Any]] = []

    for g in graphs:
        graph_attribute = g.get("graph_attrs", {}) or {}

        #Rohe Zählung
        node_count_raw = len(list(get_nodes(g)))
        edge_count_raw = len(list(get_edges(g)))

        sm = generate_graph_metrics(g)

        row: Dict[str, Any] = {
            "kind": kind,
            "graph_id": g.get("graph_id"),
            "node_count": node_count_raw,
            "edge_count": edge_count_raw,
            **sm,                               #Entpacke alle Key-Value-Paare und füge sie ein - neu gelernt, noch irgendwo aufschreiben
        }

        if kind == "ist":
            row["dataset"] = graph_attribute.get("dataset")
            for k in ["malo_count", "melo_count", "tr_count", "nelo_count"]:
                row[k] = graph_attribute.get(k)
        else:
            row["lbs_code"] = graph_attribute.get("lbs_code") or g.get("label")
            for t in ["malo", "melo", "tr", "nelo"]:
                row[f"{t}_min"] = graph_attribute.get(f"{t}_min")
                row[f"{t}_max"] = graph_attribute.get(f"{t}_max")
                row[f"{t}_node_types"] = graph_attribute.get(f"{t}_node_types")
            row["attachment_rules_n"] = len(graph_attribute.get("attachment_rules") or [])
            row["constraints_n"] = len(graph_attribute.get("optionality_constraints") or [])

        #Edge Typen
        edge_typ_zaehler = Counter(rel for _, _, rel in get_valid_typed_edges(g))
        for rel in ["MEMA", "METR", "MENE", "MEME"]:
            row[f"edge_{rel}"] = int(edge_typ_zaehler.get(rel, 0))

        zeilen.append(row)

    return zeilen

# -------------------------
# FEATURES ALLE RICHTIG?
# -------------------------


def _direction_normalizer(dir_raw: Any) -> Optional[str]:
    """
    Richtungen vereinheitlichen (ich weiß nicht mehr, ob nötig, wenn wir pipeline importieren)
    """
    if dir_raw is None:
        return None
    s = str(dir_raw).strip()
    if not s:
        return None

    s_low = s.lower()

    #TR Typen aus UTILMD
    s_up = s.upper()
    if s_up == "Z17":
        return "consumption"
    if s_up == "Z50":
        return "generation"
    if s_up == "Z56":
        return "both"

    # Speicher - Both
    if "storage" in s_low or "speicher" in s_low:
        return "both"

    # Wenn bereits einheitlich
    if s_low in ("consumption", "generation", "both"):
        return s_low

    # Falls was falsch geschrieben sein sollte
    if ("consumption" in s_low and "generation" in s_low) or ("einspeis" in s_low and "ausspeis" in s_low):
        return "both"

    #Ableitungen daraus
    if "einspeis" in s_low or "erzeug" in s_low:
        return "generation"
    if "ausspeis" in s_low or "bezug" in s_low or "verbrauch" in s_low:
        return "consumption"

    return None


def is_tatsaechlich_nutzbar(graphs: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    """
    Um zu schauen wie oft Werte in unknown fallen
    """
    if gp is not None:
        NODE_TYPES = getattr(gp, "NODE_TYPES", {"MaLo": 0, "MeLo": 1, "TR": 2, "NeLo": 3})
        EDGE_TYPES = getattr(gp, "EDGE_TYPES", {"MEMA": 0, "METR": 1, "MENE": 2, "MEME": 3})
        DIRECTIONS = getattr(gp, "DIRECTIONS", {"consumption": 0, "generation": 1, "both": 2})
        normalize_dir = getattr(gp, "_normalize_direction", _direction_normalizer)
    else:
        NODE_TYPES = {"MaLo": 0, "MeLo": 1, "TR": 2, "NeLo": 3}
        EDGE_TYPES = {"MEMA": 0, "METR": 1, "MENE": 2, "MEME": 3}
        DIRECTIONS = {"consumption": 0, "generation": 1, "both": 2}
        normalize_dir = _direction_normalizer

    counts = Counter()
    unknown = Counter()

    direction_values_c = Counter()
    node_type_values_c = Counter()
    rel_values_c = Counter()

    for g in graphs:
        for n in get_nodes(g):
            ntype = str(n.get("type"))
            node_type_values_c[ntype] += 1
            counts["nodes_total"] += 1
            if ntype not in NODE_TYPES:
                unknown["node_type_unknown"] += 1

            attrs = n.get("attrs", {}) or {}

            # Richtungs-Logik aus pipeline - kann man das noch in eine Funktion schreiben?
            direction_raw = None
            if ntype == "TR":
                direction_raw = attrs.get("direction")
                if direction_raw is None:
                    direction_raw = attrs.get("tr_direction")
                if direction_raw is None:
                    direction_raw = attrs.get("tr_type_code") or attrs.get("art_der_technischen_ressource")
            elif ntype in ("MaLo", "MeLo"):
                direction_raw = attrs.get("direction")

            if direction_raw is not None:
                direction_values_c[str(direction_raw)] += 1

            # NeLo egal
            if ntype in ("MaLo", "MeLo", "TR"):
                counts["dir_nodes_total"] += 1
                canon = normalize_dir(direction_raw)
                if canon is None or canon not in DIRECTIONS:
                    unknown["dir_unknown"] += 1

        for _, _, rel in get_valid_typed_edges(g):
            rel = str(rel).strip().upper()
            rel_values_c[rel] += 1
            counts["edges_total"] += 1
            if rel not in EDGE_TYPES:
                unknown["edge_rel_unknown"] += 1

    return {
        "kind": kind,
        "counts": dict(counts),
        "unknown": dict(unknown),
        "raw_node_types": node_type_values_c,
        "raw_edge_types": rel_values_c,
        "raw_direction_values": direction_values_c,
    }
# -------------------------
# BOUNDS PRÜFEN
# -------------------------

def _as_int_or_none(x: Any) -> Optional[int]:
    """
    Gibt Parameter als int zurück oder eben nicht
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    try:
        return int(x)
    except Exception:
        return None


def within_bounds(wert: int, minimum: Optional[int], maximum: Optional[int]) -> bool:
    """
    Ist der Wert innerhalb der Grenzen?
    """
    if minimum is not None and wert < minimum:
        return False
    if maximum is not None and wert > maximum:
        return False
    return True


def in_bounds_per_template(ist_graphs: List[Dict[str, Any]], soll_templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pro Template messen wie viele Instance Graphs die Grenzen erfüllen
    """
    ist_zeilen: List[Dict[str, Any]] = []
    for g in ist_graphs:
        graph_attribute = g.get("graph_attrs", {}) or {}
        actual = Counter(str(n.get("type")) for n in get_nodes(g))
        ist_zeilen.append(
            {
                "graph_id": g.get("graph_id"),
                "dataset": graph_attribute.get("dataset"),
                "malo": _as_int_or_none(graph_attribute.get("malo_count")) if graph_attribute.get("malo_count") is not None else int(actual.get("MaLo", 0)),
                "melo": _as_int_or_none(graph_attribute.get("melo_count")) if graph_attribute.get("melo_count") is not None else int(actual.get("MeLo", 0)),
                "tr": _as_int_or_none(graph_attribute.get("tr_count")) if graph_attribute.get("tr_count") is not None else int(actual.get("TR", 0)),
                "nelo": _as_int_or_none(graph_attribute.get("nelo_count")) if graph_attribute.get("nelo_count") is not None else int(actual.get("NeLo", 0)),
            }
        )

    out: List[Dict[str, Any]] = []
    for t in soll_templates:
        graph_attribute = t.get("graph_attrs", {}) or {}
        code = graph_attribute.get("lbs_code") or t.get("label") or t.get("graph_id")
        bounds = {}
        for k in ["malo", "melo", "tr", "nelo"]:
            bounds[f"{k}_min"] = _as_int_or_none(graph_attribute.get(f"{k}_min"))
            bounds[f"{k}_max"] = _as_int_or_none(graph_attribute.get(f"{k}_max"))
        ok = 0
        ok_per_dataset = Counter()
        for r in ist_zeilen:
            if (
                within_bounds(r["malo"], bounds["malo_min"], bounds["malo_max"])
                and within_bounds(r["melo"], bounds["melo_min"], bounds["melo_max"])
                and within_bounds(r["tr"], bounds["tr_min"], bounds["tr_max"])
                and within_bounds(r["nelo"], bounds["nelo_min"], bounds["nelo_max"])
            ):
                ok += 1
                ok_per_dataset[str(r["dataset"])] += 1
        zeile = {"lbs_code": code, "templates_graph_id": t.get("graph_id"), **bounds, "ist_graphs_fitting_total": ok}
        for ds, cnt in ok_per_dataset.items():
            zeile[f"ist_fitting_{ds}"] = int(cnt)
        out.append(zeile)

    out.sort(key=lambda r: (-int(r["ist_graphs_fitting_total"]), str(r["lbs_code"])))
    return out


# -------------------------
# PLOT HELFER
# -------------------------

def plot_histogramm(werte: List[int], titel: str, xlabel: str, out_path: Path) -> None:
    """
    Erzeugt Histogramm
    """
    if plt is None:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(werte, bins=min(60, max(5, int(math.sqrt(len(werte))) if werte else 5)))
    ax.set_title(titel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


# -------------------------
# TOPOLOGY CHECKS
# -------------------------

def topology_per_row(graph: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """
    Topologie zeilenweise überprüfen
    """
    ids = set_of_node_ids(graph)
    id_to_type: Dict[str, str] = {}
    for n in get_nodes(graph):
        node_id = n.get("id")
        if node_id is None:
            continue
        id_to_type[str(node_id)] = str(n.get("type"))

    node_types = Counter(id_to_type.get(nid, "UNKNOWN") for nid in ids)
    edge_types = Counter(rel for _, _, rel in get_valid_typed_edges(graph))

    melo_count = int(node_types.get("MeLo", 0))
    malo_count = int(node_types.get("MaLo", 0))
    tr_count = int(node_types.get("TR", 0))
    nelo_count = int(node_types.get("NeLo", 0))

    erwartete_mema = melo_count > 0 and malo_count > 0
    erwartete_metr = melo_count > 0 and tr_count > 0
    erwartete_mene = melo_count > 0 and nelo_count > 0

    hat_mema = int(edge_types.get("MEMA", 0)) > 0
    hat_metr = int(edge_types.get("METR", 0)) > 0
    hat_mene = int(edge_types.get("MENE", 0)) > 0
    hat_meme = int(edge_types.get("MEME", 0)) > 0

    graph_attributes = graph.get("graph_attrs", {}) or {}
    lbs_code = graph_attributes.get("lbs_code") or graph.get("label")

    ev = edge_validator(graph)

    rel_set = sorted(set(edge_types.keys()))

    row: Dict[str, Any] = {
        "kind": kind,
        "graph_id": graph.get("graph_id"),
        "lbs_code": lbs_code,
        "MaLo": malo_count,
        "MeLo": melo_count,
        "TR": tr_count,
        "NeLo": nelo_count,
        "edge_MEMA": int(edge_types.get("MEMA", 0)),
        "edge_METR": int(edge_types.get("METR", 0)),
        "edge_MENE": int(edge_types.get("MENE", 0)),
        "edge_MEME": int(edge_types.get("MEME", 0)),
        "rel_types_present": ",".join(rel_set) if rel_set else "",
        "expects_MEMA": erwartete_mema,
        "missing_MEMA": bool(erwartete_mema and not hat_mema),
        "expects_METR": erwartete_metr,
        "missing_METR": bool(erwartete_metr and not hat_metr),
        "expects_MENE": erwartete_mene,
        "missing_MENE": bool(erwartete_mene and not hat_mene),
        "multi_MeLo": bool(melo_count > 1),
        "has_MEME": bool(hat_meme),
        "attachment_rules_n": len(graph_attributes.get("attachment_rules") or []),
        "edges_missing_refs": int(ev.get("edges_missing_refs", 0)),
        "edges_missing_srcdst": int(ev.get("edges_missing_srcdst", 0)),
        "edges_missing_rel": int(ev.get("edges_missing_rel", 0)),
    }
    return row


# -------------------------
# MAIN REPORT
# -------------------------


def run_analysis(
    ist_path: Path,
    soll_path: Path,
    ausgabe_path: Path,
    max_graphs: Optional[int] = None,
    create_plots: bool = True,
) -> Path:
    rw = ReportWriter(ausgabe_path)
    rw.h1("DESCRIPTIVE ANALYSIS")

    rw.h2("Inputs")
    rw.bullet(
        [
            f"Instance-Graphs: `{ist_path}`",
            f"Graph-Templates: `{soll_path}`",
            f"Graphs covered (debug): `{max_graphs}`" if max_graphs is not None else "Graphs covered: _all_",
            f"Plots: {'created' if create_plots else 'not created'} (matplotlib {'installed' if plt is not None else 'matplotlib not found'})",
            f"pandas: {'installed' if pd is not None else 'not installed'}",
            f"graph_pipeline Import: {'ok' if gp is not None else 'not available (fallback-mappings used)'}",
            "Structural metrics: only _valid_ edges (src/dst reference existing nodes).",
        ]
    )

    # Laden
    ist_graphen = jsonl_loader(ist_path, obergrenze=max_graphs)
    soll_graphen = jsonl_loader(soll_path, obergrenze=max_graphs)
    rw.text(f"Loaded: **{len(ist_graphen)}** Instance-Graphs, **{len(soll_graphen)}** Graph-Templates.")

    # Validierung
    rw.h2("Quality-Check")
    all_issues: List[ValidationIssue] = []
    for g in ist_graphen:
        all_issues.extend(graph_validator(g, "ist"))
    for g in soll_graphen:
        all_issues.extend(graph_validator(g, "soll"))

    by_sev = Counter(i.schwere_grad for i in all_issues)
    rw.bullet(
        [
            f"Issues total: {len(all_issues)}",
            f"ERRORS: {by_sev.get('ERROR',0)}",
            f"WARNINGS: {by_sev.get('WARN',0)}",
        ]
    )

    rw.text(f"Errors would include missing node/edge keys, nodes without ID, duplicate IDs, edges referencing non-existent nodes.")
    rw.text(f"Warnings would include unknown node types/edge relations or wrong node counts.")

    top_msgs = Counter(i.message for i in all_issues)
    issue_rows = [{"message": msg, "count": int(cnt)} for msg, cnt in top_msgs.most_common(25)]
    rw.h3("Most common issues")
    rw.insert_table(Table("issues_top", issue_rows))

    if by_sev.get("ERROR", 0) > 0:
        sample_err = [i for i in all_issues if i.schwere_grad == "ERROR"][:30]
        rw.h3("ERROR contexts")
        rw.insert_table(Table("issues_error_samples", [{"graph_id": e.problem_bei, "message": e.message} for e in sample_err]))

    # Keysets
    rw.h2("Fields present in Instance-Graphs")
    rw.text(f"Metadata fields that are consistently available across Instance-Graphs.")
    ist_ga_keys, ist_node_attr_keys, ist_edge_keys = keyset_extrahieren(ist_graphen)
    soll_ga_keys, soll_node_attr_keys, soll_edge_keys = keyset_extrahieren(soll_graphen)

    def top_counter_table(name: str, c: Counter, total_graphs: int) -> Table:
        rows = []
        for k, cnt in c.most_common(50):
            rows.append({"KEY": k, "COUNT": int(cnt), "%": f"{prozentrechner(cnt, total_graphs):.1f}%"})
        return Table(name, rows)

    rw.h3("Instances: graph_attrs Keys (How often per Graph?)")
    rw.text(f"graph_attr is a metadata dictionary.")
    rw.insert_table(top_counter_table("ist_graph_attrs_keys", ist_ga_keys, len(ist_graphen)))

    rw.h3("Templates: graph_attrs Keys (How often per Graph?)")
    rw.text(f"graph_attr is a metadata dictionary.")
    rw.insert_table(top_counter_table("soll_graph_attrs_keys", soll_ga_keys, len(soll_graphen)))

    rw.h3("Instances: Node-Attribute Keys (How often per Node?)")
    rw.insert_table(Table("ist_node_attr_keys", [{"KEY": k, "COUNT": int(v)} for k, v in ist_node_attr_keys.most_common(60)]))

    rw.h3("Templates: Node-Attribute Keys (How often per Node?)")
    rw.insert_table(Table("soll_node_attr_keys", [{"KEY": k, "COUNT": int(v)} for k, v in soll_node_attr_keys.most_common(60)]))

    # Node/Edge-Typ-Verteilung
    rw.h2("Structure: Node Types & Edge Types")
    ist_nt = node_type_counts(ist_graphen)
    soll_nt = node_type_counts(soll_graphen)
    ist_et = edge_type_counts(ist_graphen)
    soll_et = edge_type_counts(soll_graphen)

    rw.h3("Node Types in Dataset (Full)")
    rows = []
    for t in sorted(set(ist_nt.keys()) | set(soll_nt.keys())):
        rows.append({"TYPE": t, "IN INSTANCES": int(ist_nt.get(t, 0)), "IN TEMPLATES": int(soll_nt.get(t, 0))})
    rw.insert_table(Table("node_type_distribution", rows))

    rw.h3("Edge Types in Dataset (Full)")
    rw.text("Only valid edges are considered. Edges are considered valid, when they have a valid source and a valid destination.")
    rows = []
    for r in sorted(set(ist_et.keys()) | set(soll_et.keys())):
        rows.append({"RELATION": r, "IN INSTANCES": int(ist_et.get(r, 0)), "IN TEMPLATES": int(soll_et.get(r, 0))})
    rw.insert_table(Table("edge_type_distribution", rows))

    # Statistik pro Graph
    rw.h2("Metrics per Graph")
    ist_rows = dataframe_per_graph(ist_graphen, "ist")
    soll_rows = dataframe_per_graph(soll_graphen, "soll")

    rw.h3("Per-Graph-Table (Sample)")
    rw.text("Full table is stored in `tables/`.")
    rw.insert_table(Table("per_graph_ist", ist_rows), max_rows=25)
    rw.insert_table(Table("per_graph_soll", soll_rows), max_rows=25)

    # Größenverteilung
    ist_nodes = [int(r["node_count"]) for r in ist_rows]
    ist_edges = [int(r["edge_count"]) for r in ist_rows]
    soll_nodes = [int(r["node_count"]) for r in soll_rows]
    soll_edges = [int(r["edge_count"]) for r in soll_rows]

    rw.h3("Dimensions (Nodes/Edges)")
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
    rw.insert_table(Table("size_distributions", size_rows))

    if create_plots:
        plot_histogramm(ist_nodes, "Instance-Graphs: Node count ", "node_count", rw.out_dir / "plots" / "ist_node_count.png")
        plot_histogramm(ist_edges, "Instance-Graphs: Edge count ", "edge_count", rw.out_dir / "plots" / "ist_edge_count.png")
        plot_histogramm(soll_nodes, "Templates: Node count ", "node_count", rw.out_dir / "plots" / "soll_node_count.png")
        plot_histogramm(soll_edges, "Templates: Edge count ", "edge_count", rw.out_dir / "plots" / "soll_edge_count.png")
        rw.text("Plots saved in `plots/` (siehe Dateien).")

    # Connectivity + new structural stats
    rw.h3("Number and Size of Connected Components")
    rw.text("This considers valid and undirected connections only to approximate message-parsing reachability.")
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
    rw.insert_table(Table("connectivity_summary", comp_rows))

    rw.h3("Density, Edges per Node, Degree and Isolated Nodes")
    rw.text(f"Insights on how structurally rich the given dataset is.")
    rw.text(f"Density shows how many edges the graph has, compared to the maximum possible amount.")
    rw.text(f"Edges per Node shows structural repetition.")
    rw.text(f"Degree shows the number of nodes connected to a node.")
    rw.text(f"Isolated nodes are nodes with degree 0.")

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

    rw.insert_table(Table("struct_metrics_summary", [_struct_summary(ist_rows, "ist"), _struct_summary(soll_rows, "soll")]))

    # Grad pro Knotentyp
    rw.h3("Degree Statistics per Node Type - Undirected")

    def agg_degree_undirected(graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        acc: Dict[str, List[int]] = defaultdict(list)
        for g in graphs:
            per_t = degree_per_typ(g)
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

    rw.insert_table(Table("degree_stats_ist", agg_degree_undirected(ist_graphen)))
    rw.insert_table(Table("degree_stats_soll", agg_degree_undirected(soll_graphen)))

    rw.h3("Degree Statistics per Node Type - Directed")

    def agg_degree_directed(graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        acc_out: Dict[str, List[int]] = defaultdict(list)
        acc_in: Dict[str, List[int]] = defaultdict(list)
        for g in graphs:
            per_t = gerichtet_degree_per_typ(g)
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

    rw.insert_table(Table("degree_stats_directed_ist", agg_degree_directed(ist_graphen)))
    rw.insert_table(Table("degree_stats_directed_soll", agg_degree_directed(soll_graphen)))

    # Grad-Verteilung
    rw.h3("Degree-Distribution (Plots; undirected, unique neighbors)")
    if create_plots:
        def collect_all_degrees(graphs: List[Dict[str, Any]]) -> Dict[str, List[int]]:
            acc: Dict[str, List[int]] = defaultdict(list)
            for g in graphs:
                per = degree_per_typ(g)
                # overall
                for t, vals in per.items():
                    acc["ALL"].extend(vals)
                    acc[t].extend(vals)
            return acc

        ist_deg = collect_all_degrees(ist_graphen)
        soll_deg = collect_all_degrees(soll_graphen)

        # overall
        plot_histogramm(ist_deg.get("ALL", []), "Ist: Degree distribution (undirected)", "degree", rw.out_dir / "plots" / "ist_degree_all.png")
        plot_histogramm(soll_deg.get("ALL", []), "Soll: Degree distribution (undirected)", "degree", rw.out_dir / "plots" / "soll_degree_all.png")

        # per type (bounded set)
        for t in ["MaLo", "MeLo", "TR", "NeLo"]:
            if ist_deg.get(t):
                plot_histogramm(ist_deg[t], f"Ist: Degree ({t}, undirected)", "degree", rw.out_dir / "plots" / f"ist_degree_{t}.png")
            if soll_deg.get(t):
                plot_histogramm(soll_deg[t], f"Soll: Degree ({t}, undirected)", "degree", rw.out_dir / "plots" / f"soll_degree_{t}.png")

        rw.text("Degree-Plots saved in `plots/` (ist_degree_*.png, soll_degree_*.png).")
    else:
        rw.text("Plots are deactivated (no-plots should be turned true, if you want otherwise).")

    # Graph-Signaturen
    rw.h2("Graph Signatures - Structural Diversity in the Dataset")

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
                    "%": f"{prozentrechner(cnt, total):.2f}%",
                    "example_graph_id": sig_to_example.get(sid, ""),
                    "signature": sig_to_text.get(sid, ""),
                }
            )
        meta = {"kind": kind, "unique_signatures": len(sig_counter), "graphs": total}
        return top, meta

    top_ist, meta_ist = signatures_table(ist_graphen, "ist")
    top_soll, meta_soll = signatures_table(soll_graphen, "soll")

    rw.bullet(
        [
            f"Instances: {meta_ist['graphs']} Graphs, {meta_ist['unique_signatures']} unique signatures",
            f"Templates: {meta_soll['graphs']} Graphs, {meta_soll['unique_signatures']} unique signatures",
            f"Signatures are based on node type counts, edge relation counts, number of connected components and degree-Min/Median/Max. \n"
            f"Connected Components are a great indicator, whether the structure forms a single coherent graph or splits into disconnected subgraphs.",
        ]
    )
    rw.h3("Top Signatures (Instances)")
    rw.insert_table(Table("signatures_top_ist", top_ist), max_rows=25)
    rw.h3("Top Signatures (Templates)")
    rw.insert_table(Table("signatures_top_soll", top_soll), max_rows=25)

    # Attribut-Vollständigkeit und values Verteilung
    rw.h2("Attributes & Value-Distribution")
    key_attrs_by_type = {
        # These keys are either used directly as features or used as *fallbacks* to derive direction.
        "MaLo": ["direction"],
        "MeLo": ["direction"],
        "TR": ["direction"],
        "NeLo": [],
    }

    def attr_coverage(graphs: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        totals = Counter()
        present = Counter()
        for g in graphs:
            for n in get_nodes(g):
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
                    "present %": f"{prozentrechner(pre, tot):.1f}%",
                }
            )
        return rows

    rw.h3("Coverage: Main Attributes (Instances)")
    rw.insert_table(Table("attr_coverage_ist", attr_coverage(ist_graphen, "ist")))

    rw.h3("Coverage: Main Attributes (Templates)")
    rw.insert_table(Table("attr_coverage_soll", attr_coverage(soll_graphen, "soll")))

    def top_values(graphs: List[Dict[str, Any]], kind: str, ntype: str, attr: str, topn: int = 20) -> Table:
        c = Counter()
        for g in graphs:
            for n in get_nodes(g):
                if str(n.get("type")) != ntype:
                    continue
                attrs = n.get("attrs", {}) or {}
                if attr in attrs and attrs.get(attr) is not None:
                    c[str(attrs.get(attr))] += 1
        rows = [{"kind": kind, "type": ntype, "attr": attr, "value": v, "count": int(cnt)} for v, cnt in c.most_common(topn)]
        return Table(f"top_values_{kind}_{ntype}_{attr}", rows)

    rw.h3("Top Values: Direction (MaLo)")
    rw.insert_table(top_values(ist_graphen, "ist", "MaLo", "direction"))
    rw.insert_table(top_values(soll_graphen, "soll", "MaLo", "direction"))

    rw.h3("Top Values: Direction (MeLo)")
    rw.insert_table(top_values(ist_graphen, "ist", "MeLo", "direction"))
    rw.insert_table(top_values(soll_graphen, "soll", "MeLo", "direction"))

    rw.h3("Top Values: Direction (TR)")
    rw.insert_table(top_values(ist_graphen, "ist", "TR", "direction"))
    rw.insert_table(top_values(soll_graphen, "soll", "TR", "direction"))

    # Template spezifisch
    rw.h2("Template-specific Analysis (LBS-Scheme)")

    # Topologie-Checkliste
    rw.h3("Topology Checklist: Expected vs. Existing Relations")

    tmpl_rows = [topology_per_row(t, "soll") for t in soll_graphen]
    tmpl_rows.sort(key=lambda r: (not (r["missing_MEMA"] or r["missing_METR"] or r["missing_MENE"]), str(r.get("lbs_code") or "")))

    rw.insert_table(Table("template_topology_checklist", tmpl_rows), max_rows=60)

    def checklist_summary(rows_: List[Dict[str, Any]], kind: str) -> List[str]:
        total = len(rows_)
        m_mema = sum(1 for r in rows_ if r.get("missing_MEMA"))
        m_metr = sum(1 for r in rows_ if r.get("missing_METR"))
        m_mene = sum(1 for r in rows_ if r.get("missing_MENE"))
        invalid_refs = sum(1 for r in rows_ if int(r.get("edges_missing_refs", 0)) > 0)
        return [
            f"{kind}: total={total}",
            f"{kind}: missing_MEMA={m_mema} (only when MeLo+MaLo exists)",
            f"{kind}: missing_METR={m_metr} (only when MeLo+TR exists)",
            f"{kind}: missing_MENE={m_mene} (only when MeLo+NeLo exists)",
            f"{kind}: graphs_with_invalid_edge_refs={invalid_refs}",
        ]

    rw.bullet(checklist_summary(tmpl_rows, "Templates"))

    # Ist: show only anomalies to keep report small
    ist_check = [topology_per_row(g, "ist") for g in ist_graphen]
    ist_anom = [
        r
        for r in ist_check
        if r.get("missing_MEMA")
        or r.get("missing_METR")
        or r.get("edges_missing_refs", 0)
        or r.get("edges_missing_srcdst", 0)
        or r.get("edges_missing_rel", 0)
    ]

    rw.h3("Topology Checklist: Instance-Anomalies (Sample)")
    rw.bullet(checklist_summary(ist_check, "Ist"))
    ist_anom.sort(key=lambda r: (not (r.get("missing_MEMA") or r.get("missing_METR")), int(r.get("edges_missing_refs", 0)), str(r.get("graph_id") or "")))
    rw.insert_table(Table("ist_topology_anomalies", ist_anom), max_rows=40)

    # Optionalität
    opt_rows: List[Dict[str, Any]] = []
    for t in soll_graphen:
        ga = t.get("graph_attrs", {}) or {}
        code = ga.get("lbs_code") or t.get("label") or t.get("graph_id")
        optional_nodes = 0
        flexible_nodes = 0
        min0_nodes = 0
        total_nodes = 0
        per_type = Counter()
        for n in get_nodes(t):
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
    rw.insert_table(Table("template_optionality_summary", opt_rows), max_rows=60)

    # Min/Max Infos
    rw.h2("Instances vs. Templates: Flexibility Compliance")
    cov = in_bounds_per_template(ist_graphen, soll_graphen)
    rw.text("Shows how many Instance-Graphs comply to the min/max-bounds given by the governing body BDEW (per Template).")
    rw.insert_table(Table("template_coverage", cov), max_rows=60)

    # Pipeline Abgleich
    rw.h2("Feature Encoding Alignment (graph_pipeline.py)")
    align_ist = is_tatsaechlich_nutzbar(ist_graphen, "ist")
    align_soll = is_tatsaechlich_nutzbar(soll_graphen, "soll")
    def alignment_table(al: Dict[str, Any]) -> List[Dict[str, Any]]:
        c = al["counts"]
        u = al["unknown"]
        rows: List[Dict[str, Any]] = []

        nodes_total = int(c.get("nodes_total", 0))
        edges_total = int(c.get("edges_total", 0))
        dir_tot = int(c.get("dir_nodes_total", 0))

        rows.append({"kind": al["kind"], "metric": "nodes_total", "count": nodes_total})
        rows.append({"kind": al["kind"], "metric": "edges_total(valid typed)", "count": edges_total})

        nt_unk = int(u.get("node_type_unknown", 0))
        rows.append(
            {
                "kind": al["kind"],
                "metric": "node_type_unknown_rate",
                "count": f"{prozent_nachkomma(nt_unk, nodes_total)} ({nt_unk}/{nodes_total})",
            }
        )

        dir_unk = int(u.get("dir_unknown", 0))
        rows.append(
            {
                "kind": al["kind"],
                "metric": "dir_unknown_rate(MaLo/MeLo/TR)",
                "count": f"{prozent_nachkomma(dir_unk, dir_tot)} ({dir_unk}/{dir_tot})",
            }
        )

        rel_unk = int(u.get("edge_rel_unknown", 0))
        rows.append(
            {
                "kind": al["kind"],
                "metric": "edge_rel_unknown_rate",
                "count": f"{prozent_nachkomma(rel_unk, edges_total)} ({rel_unk}/{edges_total})",
            }
        )

        return rows

    rw.h3("Unknown-rates for used features")
    rw.insert_table(Table("encoder_unknown_rates", alignment_table(align_ist) + alignment_table(align_soll)))

    rw.h3("Which Features are Considered for DGMC?")
    rw.text(
        "The feature encoder (graph_pipeline.py) reduces node and edge information to compact one-hot blocks. "
        "This section verifies which raw fields can actually affect the model input, and which are currently unused."
    )

    rw.bullet(
        [
            "**Node type**: `node['type']` → one-hot over {MaLo, MeLo, TR, NeLo}",
            "**Direction** (MaLo/MeLo/TR): derived from `attrs['direction']` with fallbacks (TR: `tr_direction`, `tr_type_code` / `art_der_technischen_ressource`; MaLo/MeLo: `direction_hint`) → {consumption, generation, both, unknown}",
            "**Edge relation**: `edge['rel']` (or legacy keys like `type`/`edge_type`) → one-hot over {MEMA, METR, MENE, MEME, unknown}",
            "All other node/graph attributes are currently **not** encoded as ML features.",
        ]
    )


    used_attr_keys = {"direction", "tr_direction", "direction_hint", "tr_type_code", "art_der_technischen_ressource"}
    ist_unused = [k for k, _ in ist_node_attr_keys.most_common() if k not in used_attr_keys]
    soll_unused = [k for k, _ in soll_node_attr_keys.most_common() if k not in used_attr_keys]

    rw.h3("Which Node-Attributes are within the Dataset, however unused in Encoding?")

    def unused_table(name: str, keys: List[str], counter: Counter) -> Table:
        rows = []
        for k in keys[:60]:
            rows.append({"key": k, "count": int(counter.get(k, 0))})
        return Table(name, rows)

    rw.text("Instance-Graphs (Top unused Keys):")
    rw.insert_table(unused_table("ist_unused_node_attrs", ist_unused, ist_node_attr_keys))

    rw.text("Template-Graphs (Top unused Keys):")
    rw.insert_table(unused_table("soll_unused_node_attrs", soll_unused, soll_node_attr_keys))

    # Automatische Beobachtungen
    rw.h2("Observations (Automatic) and Possible Causes")
    observations: List[str] = []

    ist_types = set(ist_nt.keys())
    soll_types = set(soll_nt.keys())
    only_in_soll = sorted(soll_types - ist_types)
    only_in_ist = sorted(ist_types - soll_types)
    if only_in_soll:
        observations.append(
            f"Templates have node type(s) {only_in_soll}, that are not in Instance-Graphs. Possible reason: Instance-Converter currently doesn't model this type."
        )
    if only_in_ist:
        observations.append(
            f"Instances have node type(s) {only_in_ist}, that are not in Template-Graphs. Possible reason: SAP-Export incomplete or customized templates inaccurate."
        )

    ist_rels = set(ist_et.keys())
    soll_rels = set(soll_et.keys())
    rel_only_soll = sorted((soll_rels - ist_rels) & {"MENE", "MEME"})
    if rel_only_soll:
        observations.append(f"Templates have edge relation(s) {rel_only_soll}, that Instance-Graphs are missing. Possible reason: Instance-Converter currently doesn't model this type.")
    rel_only_ist = sorted(ist_rels - soll_rels)
    if rel_only_ist:
        observations.append(f"Instances have edge relation(s) {rel_only_ist}, that are not in Template-Graphs (check if expected or noise).")

    u_soll = align_soll["unknown"]
    c_soll = align_soll["counts"]

    if c_soll.get("dir_nodes_total", 0) and prozentrechner(u_soll.get("dir_unknown", 0), c_soll.get("dir_nodes_total", 0)) > 80:
        observations.append("Direction (MaLo/TR) in Template-Graphs often 'unknown' → Direction as feature hardly differentiates Templates.")

    # new: topology checklist highlights
    tmpl_missing_mema = sum(1 for r in tmpl_rows if r.get("missing_MEMA"))
    tmpl_missing_metr = sum(1 for r in tmpl_rows if r.get("missing_METR"))
    if tmpl_missing_mema or tmpl_missing_metr:
        observations.append(
            f"Topology-Check: {tmpl_missing_mema} Template(s) don't have MEMA although MeLo+MaLo exists, {tmpl_missing_metr} Template(s) miss METR although MeLo+TR exists. Check if: Model choice (attachment_rules) vs. mapping-gap."
        )

    if observations:
        rw.bullet(observations)
    else:
        rw.text("_No noticeable observations._")

    report_path = rw.finalize()
    return report_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ist", type=str, default="data/ist_graphs_all.jsonl", help="Pfad zu Ist-Graphen JSONL")
    ap.add_argument("--soll", type=str, default="data/lbs_soll_graphs.jsonl", help="Pfad zu Soll-Template JSONL")
    ap.add_argument("--out_dir", type=str, default="analysis/descriptive", help="Output-Verzeichnis")
    ap.add_argument("--max_graphs", type=int, default=None, help="Begrenze Anzahl Graphen (Debug)")
    ap.add_argument("--no-plots", action="store_true", help="Deaktiviere Plots")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_analysis(
        ist_path=Path(args.ist),
        soll_path=Path(args.soll),
        ausgabe_path=Path(args.out_dir),
        max_graphs=args.max_graphs,
        create_plots=not args.no_plots,
    )

    print(f"Deskriptive Analyse geschrieben: {report}")
