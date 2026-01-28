"""
baseline_constraints.py

Rule/checklist baseline
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonGraph = Dict[str, Any]


# ------------------------------
# PFADE
# ------------------------------


MAIN_VERZEICHNIS = Path(__file__).resolve().parent


def _pfad_resolver(path: Path) -> Path:
    """
    Pfad-Fallbacks für robustes Error-Handling
    """
    if path.exists():
        return path
    neu = MAIN_VERZEICHNIS / path.name
    if neu.exists():
        return neu
    return path


def _jsonl_iterieren(path: Path, max_lines: Optional[int] = None) -> Iterable[JsonGraph]:
    """
    JSONL durchlaufen
    """
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_lines is not None and i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _jsonl_loader(path: Path, max_lines: Optional[int] = None) -> List[JsonGraph]:
    """
    JSONL laden
    """
    return list(_jsonl_iterieren(path, max_lines=max_lines))


def _ensure_dir(p: Path) -> None:
    """
    Verzeichnis erstellen, wenn nötig
    """
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------
# DIRECTIONS VEREINHEITLICHEN
# ------------------------------


def _direction_normalizer(dir_eingabe: Any) -> Optional[str]:
    """
    Direction vereinheitlichen {consumption, generation, both} oder none
    """
    if dir_eingabe is None:
        return None
    s = str(dir_eingabe).strip()
    if not s:
        return None
    s_low = s.lower()

    # TR Direction s. UTILMD
    s_up = s.upper()
    if s_up == "Z17":
        return "consumption"
    if s_up == "Z50":
        return "generation"
    if s_up == "Z56":
        return "both"

    # Speicher ist both
    if "storage" in s_low or "speicher" in s_low:
        return "both"

    # alles mal vereinheitlichen
    if s_low in ("consumption", "generation", "both"):
        return s_low

    # Kombination daraus
    if ("consumption" in s_low and "generation" in s_low) or (
        "einspeis" in s_low and "ausspeis" in s_low
    ):
        return "both"

    # Heuriostik wie bereits in anderen Codes
    if "einspeis" in s_low or "erzeug" in s_low:
        return "generation"
    if "ausspeis" in s_low or "bezug" in s_low or "verbrauch" in s_low:
        return "consumption"

    return None


def _node_direction(node: Dict[str, Any]) -> str:
    """
    Direction für Nodes
    """
    node_types = node.get("type")
    attrs = node.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    direction_alt = attrs.get("direction")
    if direction_alt is None and node_types == "TR":
        direction_alt = attrs.get("tr_direction")
    if direction_alt is None and node_types == "TR":
        direction_alt = attrs.get("tr_type_code") or attrs.get("art_der_technischen_ressource")
    if direction_alt is None and node_types in ("MaLo", "MeLo"):
        direction_alt = attrs.get("direction_hint")

    canon = _direction_normalizer(direction_alt)
    return canon if canon is not None else "unknown"


def _lbs_object_direction(obj: Dict[str, Any]) -> str:
    """
    Direction für Objekte aus _lbs_object
    """
    if not isinstance(obj, dict):
        return "unknown"

    typ = str(obj.get("object_type") or "").strip()

    direction_alt = obj.get("direction")
    if direction_alt is None:
        direction_alt = obj.get("direction_hint")

    #TR
    if direction_alt is None and typ == "TR":
        direction_alt = obj.get("tr_direction")
    if direction_alt is None and typ == "TR":
        direction_alt = obj.get("tr_type_code") or obj.get("art_der_technischen_ressource")

    # Weil man es nicht immer direkt ableiten kann, indirekte Wege auch mit reinnehmen
    if direction_alt is None:
        attrs = obj.get("attrs")
        if isinstance(attrs, dict):
            direction_alt = attrs.get("direction") or attrs.get("direction_hint")

    canon = _direction_normalizer(direction_alt)
    return canon if canon is not None else "unknown"


# ------------------------------
# CONSTRAINTS AUS TEMPLATES
# ------------------------------

def _as_int(wert: Any, default: int = 0) -> int:
    """
    Als Integer speichern
    """
    try:
        return int(wert)
    except Exception:
        return default


def _is_unbounded_max(wert: Any) -> bool:
    """
    Checken, ob theoretisch unendlich viele Objekte nötig
    """
    if wert is None:
        return True
    if isinstance(wert, str) and wert.strip().upper() == "N":
        return True
    return False


def optionality_aus_json(lbs_json_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    LBS JSON auf dessen optionality-Block, Pfad und Version mappen (aktuell fnale Version)
    """
    catalog: Dict[str, Dict[str, Any]] = {}

    for path in sorted(lbs_json_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        opt = obj.get("_lbs_optionality")
        if not isinstance(opt, dict):
            continue

        lbs_code = str(opt.get("lbs_code") or "").strip()
        if not lbs_code:
            continue

        catalog[lbs_code] = {"opt": opt, "path": str(path)}

    return catalog


@dataclass(frozen=True)
class Bounds:
    """
    Um Kardinalitäten darzustellen
    """
    min: int
    max: Optional[int]  # None heißt unbounded


def _bounds_from_lbs_objects(lbs_objects: List[Dict[str, Any]], object_type: str) -> Bounds:
    """
    Aggregate min/max occurrences for a node type across all LBS roles
    """
    min = 0
    max_summe = 0
    unbounded = False

    for o in lbs_objects:
        if not isinstance(o, dict):
            continue
        if str(o.get("object_type")) != object_type:
            continue

        min += _as_int(o.get("min_occurs"), 0)

        max_roh = o.get("max_occurs")
        if _is_unbounded_max(max_roh):
            unbounded = True
        else:
            max_summe += _as_int(max_roh, 0)

    return Bounds(min=int(min), max=None if unbounded else int(max_summe))


def _bounds_from_graph_attrs(graph_attribute: Dict[str, Any], prefix: str) -> Bounds:
    """
    Holt sich Grenzen aus den Graph-Attributes
    """
    min = _as_int(graph_attribute.get(f"{prefix}_min"), 0)
    max_roh = graph_attribute.get(f"{prefix}_max")
    if _is_unbounded_max(max_roh):
        max: Optional[int] = None
    else:
        try:
            max = int(max_roh)
        except Exception:
            max = None
    return Bounds(min=min, max=max)


def _ref_to_melo_kandidaten(kandidat: Any) -> List[str]:
    """
    reference_to_melo als Liste von MeLo object_code Kandidaten
    """
    if kandidat is None:
        return []
    if isinstance(kandidat, str):
        s = kandidat.strip()
        return [s] if s else []
    if isinstance(kandidat, (list, tuple)):
        out: List[str] = []
        for x in kandidat:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(kandidat).strip()
    return [s] if s else []


def _equal_count_constraints_extrahieren(
    constraints: Any,
    code_to_type: Dict[str, str],
) -> List[Dict[str, str]]:
    """
    Equal-Count Constraints beachten. Wir haben keine Objektcodes in Instanz-Graphen, also können wir nur auf Typ-Level scoren.
    """
    out: List[Dict[str, str]] = []

    if not isinstance(constraints, dict):
        return out

    cardinality_constraints = constraints.get("cardinality_constraints")
    if not isinstance(cardinality_constraints, list):
        return out

    for item in cardinality_constraints:
        if not isinstance(item, dict):
            continue
        paar = item.get("equal_count_between_object_codes")
        if not (isinstance(paar, (list, tuple)) and len(paar) == 2):
            continue
        n1 = str(paar[0])
        n2 = str(paar[1])
        typ_n1 = code_to_type.get(n1, "")
        typ_n2 = code_to_type.get(n2, "")
        if not typ_n1 or not typ_n2:
            continue
        out.append({"node1_code": n1, "node1_type": typ_n1, "node2_code": n2, "node2_type": typ_n2})

    return out


@dataclass
class TemplateSignature:
    """
    Signatur für Templates
    """
    template_graph_id: str
    template_label: str
    # Bounds pro Typ (Summe min/max über alle Rollen)
    bounds_by_type: Dict[str, Bounds]
    #min_occurs>=1 aufgeteilt pro direction, gezählt in min occurrences
    mandatory_dir_counts: Dict[str, Counter]
    # Rollen pro Typ
    role_counts: Counter
    # Edge rel count in Template
    edge_counts: Counter
    # Optionality Quellen
    optionality_source_path: Optional[str]
    #optionality_version: Optional[int]

    # Strukturhinweise aus Optionality-Block
    mandatory_melo_roles: List[str]  # MeLo min_occurs>=1
    melo_min_attachments: Dict[str, Counter]  # melo_code -> Counter(type -> min_required)
    mehrdeutige_attachments_ref: List[Dict[str, Any]]  # reference_to_melo Liste
    # Constraint Zusammenfassung
    reference_rules: List[Dict[str, Any]]
    equal_count_constraints: List[Dict[str, str]]


@dataclass
class IstSignature:
    """
    Signatur für Instanzen
    """
    ist_graph_id: str
    node_counts: Counter
    direction_counts: Dict[str, Counter]
    edge_counts: Counter
    # Attachments im Graphen
    attachment_ratio_per_typ: Dict[str, float]
    # EInzigartige Nachbarn pro MeLo
    melo_neighbour_counts: Dict[str, Counter]


def build_template_signature(
    template: JsonGraph,
    *,
    optionality_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TemplateSignature:
    """
    Baut die Template Signatur und nutzt dabei auch Optionality-Block
    """

    graph_attributes = template.get("graph_attrs") or {}
    if not isinstance(graph_attributes, dict):
        graph_attributes = {}

    template_graph_id = str(template.get("graph_id") or "")
    template_label = str(template.get("label") or template.get("graph_id") or "")

    #Optionality-Block analysieren
    opt: Optional[Dict[str, Any]] = None
    opt_path: Optional[str] = None
    #opt_ver: Optional[int] = None
    lbs_objects: List[Dict[str, Any]] = []

    if optionality_catalog is not None:
        entry = optionality_catalog.get(template_label)
        if entry is not None:
            opt = entry.get("opt") if isinstance(entry, dict) else None
            if isinstance(opt, dict):
                lbs_objects_raw = opt.get("lbs_objects")
                if isinstance(lbs_objects_raw, list):
                    lbs_objects = [x for x in lbs_objects_raw if isinstance(x, dict)]
                opt_path = str(entry.get("path") or "") or None
                #try:
                #    opt_ver = int(entry.get("version"))
                #except Exception:
                #    opt_ver = None

    # Bounds beachten
    if lbs_objects:
        bounds_by_type: Dict[str, Bounds] = {
            "MaLo": _bounds_from_lbs_objects(lbs_objects, "MaLo"),
            "MeLo": _bounds_from_lbs_objects(lbs_objects, "MeLo"),
            "TR": _bounds_from_lbs_objects(lbs_objects, "TR"),
            "NeLo": _bounds_from_lbs_objects(lbs_objects, "NeLo"),
        }
    else:
        # graph_attrs als Fallback
        bounds_by_type = {
            "MaLo": _bounds_from_graph_attrs(graph_attributes, "malo"),
            "MeLo": _bounds_from_graph_attrs(graph_attributes, "melo"),
            "TR": _bounds_from_graph_attrs(graph_attributes, "tr"),
            "NeLo": _bounds_from_graph_attrs(graph_attributes, "nelo"),
        }

    #ROllen und direction counts
    role_counts = Counter()
    mandatory_direction_counts: Dict[str, Counter] = defaultdict(Counter)

    if lbs_objects:
        for o in lbs_objects:
            t = str(o.get("object_type") or "")
            if t not in ("MaLo", "MeLo", "TR", "NeLo"):
                continue
            role_counts[t] += 1

            mindest = _as_int(o.get("min_occurs"), 0)
            if mindest >= 1:
                d = _lbs_object_direction(o)
                mandatory_direction_counts[t][d] += mindest
    else:
        # Template Nodes Fallback
        for n in template.get("nodes", []) or []:
            if not isinstance(n, dict):
                continue
            t = str(n.get("type"))
            if t not in ("MaLo", "MeLo", "TR", "NeLo"):
                continue
            role_counts[t] += 1

            attrs = n.get("attrs") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            if _as_int(attrs.get("min_occurs"), 0) >= 1:
                d = _node_direction(n)
                mandatory_direction_counts[t][d] += 1

    # Template Edges
    edge_counts = Counter()
    for e in template.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
        if isinstance(rel, str):
            rel = rel.strip().upper()
        edge_counts[str(rel)] += 1

    # Referenzen und Struktur aus Optionality
    mandatory_melo_roles: List[str] = []
    min_melo_attachments: Dict[str, Counter] = defaultdict(Counter)
    mehrdeutige_attachments_req: List[Dict[str, Any]] = []

    reference_rules: List[Dict[str, Any]] = []
    equal_count_constraints: List[Dict[str, str]] = []

    if lbs_objects:
        code_to_type = {str(o.get("object_code")): str(o.get("object_type")) for o in lbs_objects}

        # MeLo Rollen - Pflicht
        for o in lbs_objects:
            if str(o.get("object_type")) != "MeLo":
                continue
            if _as_int(o.get("min_occurs"), 0) >= 1:
                mandatory_melo_roles.append(str(o.get("object_code")))

        # Attachment requirements nur Pflicht und eindeutig
        for o in lbs_objects:
            t = str(o.get("object_type") or "")
            if t not in ("MaLo", "TR", "NeLo"):
                continue

            mindest = _as_int(o.get("min_occurs"), 0)
            if mindest < 1:
                continue

            candidates = _ref_to_melo_kandidaten(o.get("reference_to_melo"))
            if not candidates:
                continue

            if len(candidates) == 1:
                melo = candidates[0]
                min_melo_attachments[melo][t] += mindest
            else:
                # Rolle könnte an mehreren MeLo hängen
                mehrdeutige_attachments_req.append(
                    {
                        "object_code": str(o.get("object_code")),
                        "object_type": t,
                        "min_occurs": mindest,
                        "candidates": candidates,
                    }
                )

        # Constraints block (if present)
        cons = (opt or {}).get("constraints") if isinstance(opt, dict) else None
        if isinstance(cons, dict):
            rr = cons.get("reference_rules")
            if isinstance(rr, list):
                reference_rules = [x for x in rr if isinstance(x, dict)]
            equal_count_constraints = _equal_count_constraints_extrahieren(cons, code_to_type)

    return TemplateSignature(
        template_graph_id=template_graph_id,
        template_label=template_label,
        bounds_by_type=bounds_by_type,
        mandatory_dir_counts=dict(mandatory_direction_counts),
        role_counts=role_counts,
        edge_counts=edge_counts,
        optionality_source_path=opt_path,
        #optionality_version=opt_ver,
        mandatory_melo_roles=mandatory_melo_roles,
        melo_min_attachments=dict(min_melo_attachments),
        mehrdeutige_attachments_ref=mehrdeutige_attachments_req,
        reference_rules=reference_rules,
        equal_count_constraints=equal_count_constraints,
    )


def build_ist_signature(graph: JsonGraph) -> IstSignature:
    """
    Baut die Instance Signatur
    """
    ist_graph_id = str(graph.get("graph_id") or "")

    node_counts = Counter()
    direction_counts: Dict[str, Counter] = defaultdict(Counter)

    id_per_typ: Dict[str, str] = {}
    malos: List[str] = []
    melos: List[str] = []
    trs: List[str] = []
    nelos: List[str] = []

    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    for n in nodes:
        t = str(n.get("type"))
        node_id = n.get("id")
        if node_id is None:
            continue
        id_string = str(node_id)
        id_per_typ[id_string] = t

        if t not in ("MaLo", "MeLo", "TR", "NeLo"):
            continue

        node_counts[t] += 1
        destination = _node_direction(n)
        direction_counts[t][destination] += 1

        if t == "MaLo":
            malos.append(id_string)
        elif t == "MeLo":
            melos.append(id_string)
        elif t == "TR":
            trs.append(id_string)
        elif t == "NeLo":
            nelos.append(id_string)

    #Edge rel zählen
    edge_counts = Counter()
    edges = [e for e in (graph.get("edges", []) or []) if isinstance(e, dict)]
    for e in edges:
        rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
        if isinstance(rel, str):
            rel = rel.strip().upper()
        edge_counts[str(rel)] += 1

    # Hat eine Node eines gewissen Typen eine erwartete Kante zu einer MeLo?
    melo_set = set(melos)

    def melo_rel_vorhanden(node_id: str, rel: str) -> bool:
        """
        Prüft ob eine Node eine entsprechende Kante zu einer MeLo hat
        """
        rel = rel.upper()
        for e in edges:
            r = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
            if isinstance(r, str):
                r = r.strip().upper()
            if str(r) != rel:
                continue
            s = str(e.get("src"))
            d = str(e.get("dst"))
            if s == node_id and d in melo_set:
                return True
            if d == node_id and s in melo_set:
                return True
        return False

    attach_ratio_per_typ: Dict[str, float] = {}
    if malos:
        ok = sum(1 for nid in malos if melo_rel_vorhanden(nid, "MEMA"))
        attach_ratio_per_typ["MaLo"] = ok / max(1, len(malos))
    if trs:
        ok = sum(1 for nid in trs if melo_rel_vorhanden(nid, "METR"))
        attach_ratio_per_typ["TR"] = ok / max(1, len(trs))
    if nelos:
        ok = sum(1 for nid in nelos if melo_rel_vorhanden(nid, "MENE"))
        attach_ratio_per_typ["NeLo"] = ok / max(1, len(nelos))

    # Pro MeLo einzigartige Nachbarn über Kanten zählen
    nachbar_sets: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    erwartete_relation = {"MEMA": "MaLo", "METR": "TR", "MENE": "NeLo"}
    for e in edges:
        rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
        if isinstance(rel, str):
            rel = rel.strip().upper()
        rel = str(rel)
        if rel not in erwartete_relation:
            continue
        erwarteter_typ = erwartete_relation[rel]

        source = str(e.get("src"))
        destination = str(e.get("dst"))
        source_typ = id_per_typ.get(source)
        destination_typ = id_per_typ.get(destination)

        if source_typ == "MeLo" and destination_typ == erwarteter_typ:
            nachbar_sets[source][erwarteter_typ].add(destination)
        elif destination_typ == "MeLo" and source_typ == erwarteter_typ:
            nachbar_sets[destination][erwarteter_typ].add(source)

    melo_neighbour_counts: Dict[str, Counter] = {}
    for melo_id, by_type in nachbar_sets.items():
        melo_neighbour_counts[melo_id] = Counter({t: len(ids) for t, ids in by_type.items()})

    return IstSignature(
        ist_graph_id=ist_graph_id,
        node_counts=node_counts,
        direction_counts=dict(direction_counts),
        edge_counts=edge_counts,
        attachment_ratio_per_typ=attach_ratio_per_typ,
        melo_neighbour_counts=melo_neighbour_counts,
    )


# ------------------------------
# BEWERTUNG
# ------------------------------

def _hard_bounds_ok(ist: IstSignature, soll: TemplateSignature) -> bool:
    """
    Sind counts innerhalb der Bounds?
    """
    for typ in ("MaLo", "MeLo", "TR", "NeLo"):
        wert = int(ist.node_counts.get(typ, 0))
        bounds = soll.bounds_by_type.get(typ, Bounds(0, None))
        if wert < int(bounds.min):
            return False
        if bounds.max is not None and wert > int(bounds.max):
            return False
    return True


def _bounds_strafe(ist: IstSignature, soll: TemplateSignature) -> float:
    """
    Wie stark ein Ist-Graph die Kardinalitäten verletzt
    """
    strafe = 0.0
    for t in ("MaLo", "MeLo", "TR", "NeLo"):
        wert = float(int(ist.node_counts.get(t, 0)))
        b = soll.bounds_by_type.get(t, Bounds(0, None))
        if wert < float(b.min):
            strafe += float(b.min) - wert
        elif b.max is not None and wert > float(b.max):
            strafe += wert - float(b.max)
    return float(strafe)


def _spezi_faktor(bounds: Bounds) -> float:
    """
    Gewichtsfaktor, der spezifische Grenzen bevorzugt. Unspezifische Bounds kriegen einen niedrigeren Faktor (näher an 0.5).
    """
    if bounds.max is None:
        return 0.70
    intervall = max(0, int(bounds.max) - int(bounds.min))
    faktor = 1.0 / (1.0 + float(intervall))
    return 0.5 + 0.5 * faktor


def _direction_vorhanden_score(benoetigt: Counter, beobachtet: Counter) -> float:
    """
    Anteil benötigter Directions, die man tatsächlich auffinden kann
    """
    total = sum(benoetigt.values())
    if total == 0:
        return 1.0
    gefunden = 0
    for d, k in benoetigt.items():
        gefunden += min(int(k), int(beobachtet.get(d, 0)))
    return gefunden / total


def _direction_passt_score(benoetigt: Counter, beobachtet: Counter) -> float:
    """
    Wie ähnlich sich die Richtungen (beobachtet, benoetigt) sind, also eigentlich nur, ob sie passen
    """
    benoetigt_insgesamt = float(sum(benoetigt.values()))
    beobachtet_insgesamt = float(sum(beobachtet.values()))
    if benoetigt_insgesamt <= 0:
        return 1.0
    if beobachtet_insgesamt <= 0:
        return 0.0

    decken_sich = 0.0
    for d, rc in benoetigt.items():
        rp = float(rc) / benoetigt_insgesamt
        op = float(beobachtet.get(d, 0)) / beobachtet_insgesamt
        decken_sich += min(rp, op)
    return float(max(0.0, min(1.0, decken_sich)))


def _equal_count_score(ist: IstSignature, soll: TemplateSignature) -> float:
    """
    Equal-Count Constraints auf Node-Ebene scoren
    """
    constraints = soll.equal_count_constraints
    if not constraints:
        return 1.0

    scores: List[float] = []
    for c in constraints:
        typ_a = c.get("a_type")
        typ_b = c.get("b_type")
        if typ_a not in ("MaLo", "MeLo", "TR", "NeLo") or typ_b not in ("MaLo", "MeLo", "TR", "NeLo"):
            continue
        a = int(ist.node_counts.get(typ_a, 0))
        b = int(ist.node_counts.get(typ_b, 0))
        denom = max(1, max(a, b))
        scores.append(1.0 - abs(a - b) / denom)

    return float(sum(scores) / max(1, len(scores))) if scores else 1.0


def _melo_structure_score(ist: IstSignature, soll: TemplateSignature) -> float:
    """
    Score über wie gut Pflicht-MeLos von Instanz-MeLos abgedeckt werden
    """
    pflicht_rollen = [r for r in soll.mandatory_melo_roles if r]
    if not pflicht_rollen:
        return 1.0

    instanz_melos = list(ist.melo_neighbour_counts.keys())
    if not instanz_melos:
        #Wenn Instanz keine Melo-Rolle hat
        return 0.0

    def role_specificity(code: str) -> int:
        """
        Erst die am stärksten eingeschränkten Slots matchen, damit sie nicht
        von einem „schlecht passenden“ Ist-Knoten blockiert werden
        Wenn man bspw. zwei MeLo-Rollen im Template setzen soll, kann man jeden nur einmal verwenden
        Wir schauen also auf die Nachbarn und setzen auf die passendere (spezifischere) Rolle
        """
        benoetigt = soll.melo_min_attachments.get(code, Counter())
        return int(sum(benoetigt.values()))

    roles_sorted = sorted(pflicht_rollen, key=role_specificity, reverse=True)

    used: set = set()
    scores: List[float] = []

    for role in roles_sorted:
        benoetigt = soll.melo_min_attachments.get(role, Counter())

        best = 0.0
        best_id: Optional[str] = None
        for melo_id in instanz_melos:
            if melo_id in used:
                continue
            beobachtet = ist.melo_neighbour_counts.get(melo_id, Counter())

            if not benoetigt:
                s = 1.0
            else:
                parts: List[float] = []
                for t, k in benoetigt.items():
                    if k <= 0:
                        continue
                    parts.append(min(int(beobachtet.get(t, 0)), int(k)) / float(k))
                s = float(sum(parts) / max(1, len(parts))) if parts else 1.0

            if s > best:
                best = s
                best_id = melo_id

        if best_id is None:
            scores.append(0.0)
        else:
            used.add(best_id)
            scores.append(best)

    return float(sum(scores) / max(1, len(scores))) if scores else 1.0


@dataclass
class ScoreBreakdown:
    """
    Container für Teil-Scores
    """
    counts: float
    mandatory: float
    dirs: float
    structure: float
    edges: float
    attachments: float
    total: float


def score_pair(
    ist: IstSignature,
    soll: TemplateSignature,
    *,
    weight_counts: float = 0.50,
    weight_mandatory: float = 0.22,
    weigh_direction: float = 0.13,
    weight_structure: float = 0.10,
    weight_edges: float = 0.03,
    weight_attachments: float = 0.02,
) -> ScoreBreakdown:
    """
    Similarity Score, min/max sind harte constraints
    """
    if not _hard_bounds_ok(ist, soll):
        # Harte Constraints nicht eingehalten
        return ScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # In-Bound bereits garantiert, jetzt nur spezifizieren
    type_weights = {"MaLo": 0.45, "MeLo": 0.45, "TR": 0.05, "NeLo": 0.05}
    score_inbound = 0.0
    weight_summe = 0.0
    for t, wt in type_weights.items():
        bound = soll.bounds_by_type.get(t, Bounds(0, None))
        score_inbound += wt * _spezi_faktor(bound)
        weight_summe += wt
    score_inbound = score_inbound / max(1e-9, weight_summe)

    # min_occurs>=1 und type+direction
    pflicht_typ_weights = {"MaLo": 0.70, "MeLo": 0.25, "TR": 0.03, "NeLo": 0.02}
    score_pflichttyp = 0.0
    weight_summe = 0.0
    for t, wt in pflicht_typ_weights.items():
        benoetigt = soll.mandatory_dir_counts.get(t, Counter())
        beobachtet = ist.direction_counts.get(t, Counter())
        score_pflichttyp += wt * _direction_vorhanden_score(benoetigt, beobachtet)
        weight_summe += wt
    score_pflichttyp = score_pflichttyp / max(1e-9, weight_summe)

    # Direction
    direction_typ_weights = {"MaLo": 0.75, "TR": 0.20, "MeLo": 0.05}
    score_directiontyp = 0.0
    weight_summe = 0.0
    for t, wt in direction_typ_weights.items():
        benoetigt = soll.mandatory_dir_counts.get(t, Counter())
        beobachtet = ist.direction_counts.get(t, Counter())
        score_directiontyp += wt * _direction_passt_score(benoetigt, beobachtet)
        weight_summe += wt
    score_directiontyp = score_directiontyp / max(1e-9, weight_summe)

    #reference_to_melo und equal counts
    score_melo_reference = _melo_structure_score(ist, soll)
    score_equal_counts = _equal_count_score(ist, soll)
    score_structure = 0.80 * score_melo_reference + 0.20 * score_equal_counts

    # Edges (Pflicht)
    benoetigt_malo = int(sum(soll.mandatory_dir_counts.get("MaLo", Counter()).values()))
    benoetigt_tr = int(sum(soll.mandatory_dir_counts.get("TR", Counter()).values()))
    benoetigt_nelo = int(sum(soll.mandatory_dir_counts.get("NeLo", Counter()).values()))
    benoetigt_edges = {"MEMA": benoetigt_malo, "METR": benoetigt_tr, "MENE": benoetigt_nelo}
    scores_edges = []
    for rel, benoetigt_anzahl in benoetigt_edges.items():
        if benoetigt_anzahl <= 0:
            continue
        beobachtet_anzahl = int(ist.edge_counts.get(rel, 0))
        scores_edges.append(min(beobachtet_anzahl, benoetigt_anzahl) / benoetigt_anzahl)
    score_edge = float(sum(scores_edges) / max(1, len(scores_edges))) if scores_edges else 1.0

    # Attachment, wenn Ist Node Typ dafür hat
    scores_attachments = []
    for t, ratio in ist.attachment_ratio_per_typ.items():
        bound = soll.bounds_by_type.get(t)
        if bound is not None and bound.max == 0 and ratio > 0:
            scores_attachments.append(0.0)
        else:
            scores_attachments.append(float(ratio))
    s_attach = float(sum(scores_attachments) / max(1, len(scores_attachments))) if scores_attachments else 1.0

    total = (
            weight_counts * score_inbound
            + weight_mandatory * score_pflichttyp
            + weigh_direction * score_directiontyp
            + weight_structure * score_structure
            + weight_edges * score_edge
            + weight_attachments * s_attach
    )

    return ScoreBreakdown(
        counts=float(score_inbound),
        mandatory=float(score_pflichttyp),
        dirs=float(score_directiontyp),
        structure=float(score_structure),
        edges=float(score_edge),
        attachments=float(s_attach),
        total=float(total),
    )


# ------------------------------
# SUBSET LABELN
# ------------------------------


MCID_TO_LABEL = {
    #Unternehmensspezifisches Mapping in Tabellen gefunden
    "S_A1_A2": "9992000000042",
    "S_C3": "9992000000175",
    "S_A001": "9992000000026",
}


def subset_label_loader(bndl2mc_path: Path) -> Dict[Tuple[str, str], str]:
    """
    Mapping zur MCID in Tabelle, das LBS code angibt
    """
    mapping: Dict[Tuple[str, str], str] = {}
    with bndl2mc_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(";")
        spalten = {name: i for i, name in enumerate(header)}
        relevant = {"Marktlokation", "Messlokation", "MCID"}
        if not relevant.issubset(spalten.keys()):
            raise ValueError(
                f"BNDL2MC header missing required columns {sorted(relevant)}. "
                f"Got: {header}"
            )
        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            try:
                malo = str(int(parts[spalten["Marktlokation"]]))
            except Exception:
                malo = str(parts[spalten["Marktlokation"]]).strip()
            melo = str(parts[spalten["Messlokation"]]).strip()
            mcid = str(parts[spalten["MCID"]]).strip()
            if not malo or not melo or not mcid:
                continue
            mapping[(malo, melo)] = mcid
    return mapping


def labeller(
    graph: JsonGraph,
    pair_to_mcid: Dict[Tuple[str, str], str],
) -> Optional[Dict[str, str]]:
    """
    Wenn MaLo/MeLo-Paar in BNDL2MC auftaucht, haben wir einen gelabelten Graphen
    """
    malos = [
        str(n.get("id"))
        for n in (graph.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "MaLo" and n.get("id") is not None
    ]
    melos = [
        str(n.get("id"))
        for n in (graph.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "MeLo" and n.get("id") is not None
    ]
    for malo in malos:
        for melo in melos:
            mcid = pair_to_mcid.get((malo, melo))
            if mcid:
                return {"mcid": mcid, "template_label": MCID_TO_LABEL.get(mcid, "")}
    return None


# ------------------------------
# MAIN
# ------------------------------


def parse_args() -> argparse.Namespace:
    """
    Um in der Konsole Konfiguration anpassen zu können.
    """
    p = argparse.ArgumentParser(description="Rule/checklist baseline for Ist→Template matching")
    p.add_argument("--ist_path", type=str, default=str("data/ist_graphs_all.jsonl"))
    p.add_argument("--templates_path", type=str, default=str("data/lbs_soll_graphs.jsonl"))
    p.add_argument("--out_path", type=str, default=str("data/rule_baseline_matches.jsonl"))
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_ist", type=int, default=None)
    p.add_argument("--max_templates", type=int, default=None)

    #Evaluierung auf gelabeltes Subset standardmäßig an
    p.add_argument(
        "--bndl2mc_path",
        type=str,
        default=str("data/training_data/BNDL2MC.csv"),
        help="Optional BNDL2MC.csv for evaluation (default: data/training_data/BNDL2MC.csv). Use '' to disable.",
    )

    # LBS JSONs
    p.add_argument(
        "--lbs_json_dir",
        type=str,
        default=str("data/lbs_templates"),
        help="Directory containing the raw LBS JSON files with _lbs_optionality",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ist_path = _pfad_resolver(Path(args.ist_path))
    soll_path = _pfad_resolver(Path(args.templates_path))
    out_path = Path(args.out_path)
    _ensure_dir(out_path.parent)

    if not ist_path.exists():
        raise FileNotFoundError(f"Ist graphs JSONL not found: {ist_path}")
    if not soll_path.exists():
        raise FileNotFoundError(f"Template graphs JSONL not found: {soll_path}")

    # Constraints laden (aus Optionality-Block)
    lbs_json_dir = _pfad_resolver(Path(args.lbs_json_dir))
    if not lbs_json_dir.exists() or not lbs_json_dir.is_dir():
        raise FileNotFoundError(f"lbs_json_dir not found or not a directory: {lbs_json_dir}")
    opt_catalog = optionality_aus_json(lbs_json_dir)

    ist_graphs = _jsonl_loader(ist_path, max_lines=args.max_ist)
    templates = _jsonl_loader(soll_path, max_lines=args.max_templates)
    if not templates:
        raise RuntimeError("No templates loaded.")

    # Labels laden
    pair_to_mcid: Dict[Tuple[str, str], str] = {}
    bndl_arg = str(args.bndl2mc_path or "").strip()
    if bndl_arg:
        bndl_path = _pfad_resolver(Path(bndl_arg))
        if bndl_path.exists():
            pair_to_mcid = subset_label_loader(bndl_path)
            print(f"[baseline] Loaded BNDL2MC pairs: {len(pair_to_mcid)} from {bndl_path}")
        else:
            print(f"[baseline][warn] BNDL2MC.csv not found at: {bndl_path} (evaluation disabled)")

    print(
        f"[baseline] ist_graphs={len(ist_graphs)} | templates={len(templates)} | "
        f"top_k={args.top_k} | lbs_json_dir={lbs_json_dir} | optionality_codes={len(opt_catalog)}"
    )

    template_signaturen = [build_template_signature(t, optionality_catalog=opt_catalog) for t in templates]

    #Ob jedes Template den Block hat
    fehlt = [sig.template_label for sig in template_signaturen if sig.optionality_source_path is None]
    if fehlt:
        print(f"[baseline][warn] Missing _lbs_optionality for templates: {sorted(set(fehlt))}")

    # Pflichtrichtungen fehlen -> Warnung
    verdaechtig = []
    for signatur in template_signaturen:
        for t in ("MaLo", "MeLo"):
            benoetigt = signatur.mandatory_dir_counts.get(t, Counter())
            if sum(benoetigt.values()) <= 0:
                continue
            bekannt = sum(v for k, v in benoetigt.items() if k != "unknown")
            if bekannt == 0:
                verdaechtig.append(signatur.template_label)
                break
    if verdaechtig:
        print(
            "[baseline][warn] Suspicious template directions (mandatory roles only 'unknown') for: "
            + ", ".join(sorted(set(verdaechtig)))
        )

    # Evaluation Zähler
    eval_total = 0
    eval_top1 = 0
    eval_top3 = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for graph in ist_graphs:
            ist_signatur = build_ist_signature(graph)

            scored: List[Tuple[int, float, float, int, ScoreBreakdown]] = []
            # Tupel (hard_ok_int, total, strafe, idx, breakdown)
            for i, tpl_sig in enumerate(template_signaturen):
                bd = score_pair(ist_signatur, tpl_sig)
                hard_ok = 1 if _hard_bounds_ok(ist_signatur, tpl_sig) else 0
                strafe = _bounds_strafe(ist_signatur, tpl_sig)
                scored.append((hard_ok, bd.total, strafe, i, bd))

            scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
            top_k = max(1, int(args.top_k))
            top = scored[:top_k]

            top_templates = []
            for rank, (hard_ok, score, strafe, idx, bd) in enumerate(top, start=1):
                soll_graphen = templates[idx]
                tpl_sig = template_signaturen[idx]

                #Checkliste für Baseline
                checklist = {
                    "hard_constraints": {
                        "bounds_ok": bool(hard_ok),
                        "bounds_violation_penalty": float(strafe),
                    },
                    "optionality": {
                        "source_path": tpl_sig.optionality_source_path,
                        #"version": tpl_sig.optionality_version,
                        "reference_rules_n": len(tpl_sig.reference_rules),
                        "equal_count_constraints_n": len(tpl_sig.equal_count_constraints),
                        "ambiguous_attach_reqs_n": len(tpl_sig.mehrdeutige_attachments_ref),
                    },
                    "node_counts": {
                        t: {
                            "obs": int(ist_signatur.node_counts.get(t, 0)),
                            "min": int(tpl_sig.bounds_by_type[t].min),
                            "max": (None if tpl_sig.bounds_by_type[t].max is None else int(tpl_sig.bounds_by_type[t].max)),
                            "ok": bool(
                                ist_signatur.node_counts.get(t, 0) >= tpl_sig.bounds_by_type[t].min
                                and (
                                    tpl_sig.bounds_by_type[t].max is None
                                    or ist_signatur.node_counts.get(t, 0) <= tpl_sig.bounds_by_type[t].max
                                )
                            ),
                        }
                        for t in ("MaLo", "MeLo", "TR", "NeLo")
                    },
                    "mandatory_dirs": {
                        t: {
                            "required_min": dict(tpl_sig.mandatory_dir_counts.get(t, Counter())),
                            "observed": dict(ist_signatur.direction_counts.get(t, Counter())),
                        }
                        for t in ("MaLo", "MeLo", "TR", "NeLo")
                        if sum(tpl_sig.mandatory_dir_counts.get(t, Counter()).values()) > 0
                    },
                    "structure": {
                        "mandatory_melo_roles": len(tpl_sig.mandatory_melo_roles),
                        "instance_melo_nodes": len(
                            [n for n in (graph.get("nodes") or []) if isinstance(n, dict) and n.get("type") == "MeLo"]
                        ),
                        "instance_melo_neighbour_counts": {
                            melo_id: {k: int(v) for k, v in cnt.items()}
                            for melo_id, cnt in list(ist_signatur.melo_neighbour_counts.items())[:10]
                        },
                        "melo_min_attach_reqs": {
                            melo_code: {k: int(v) for k, v in benoetigt.items()}
                            for melo_code, req in tpl_sig.melo_min_attachments.items()
                        },
                        "equal_count_constraints": [
                            {
                                **c,
                                "obs_a": int(ist_signatur.node_counts.get(c.get("a_type", ""), 0)),
                                "obs_b": int(ist_signatur.node_counts.get(c.get("b_type", ""), 0)),
                            }
                            for c in tpl_sig.equal_count_constraints
                        ],
                    },
                    "edge_counts": {
                        "observed": {k: int(v) for k, v in ist_signatur.edge_counts.items()},
                        "template": {k: int(v) for k, v in tpl_sig.edge_counts.items()},
                    },
                    "attachment_coverage": {k: float(v) for k, v in ist_signatur.attachment_ratio_per_typ.items()},
                }

                top_templates.append(
                    {
                        "rank": rank,
                        "template_graph_id": soll_graphen.get("graph_id"),
                        "template_label": tpl_sig.template_label,
                        "score": float(score),
                        "breakdown": {
                            "counts": bd.counts,
                            "mandatory": bd.mandatory,
                            "dirs": bd.dirs,
                            "structure": bd.structure,
                            "edges": bd.edges,
                            "attachments": bd.attachments,
                        },
                        "checklist": checklist,
                    }
                )

            gt = labeller(graph, pair_to_mcid) if pair_to_mcid else None
            if gt and gt.get("template_label"):
                eval_total += 1
                gt_label = gt["template_label"]
                pred1 = top_templates[0]["template_label"] if top_templates else ""
                if pred1 == gt_label:
                    eval_top1 += 1
                pred_labels = [x["template_label"] for x in top_templates[:3]]
                if gt_label in pred_labels:
                    eval_top3 += 1

            out_obj = {
                "ist_graph_id": ist_signatur.ist_graph_id,
                "top_templates": top_templates,
                "ground_truth": gt,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[baseline] wrote: {out_path}")
    if eval_total > 0:
        print(
            "[baseline] evaluation on labelled subset | "
            f"n={eval_total} | hits@1={eval_top1/eval_total:.3f} | hits@3={eval_top3/eval_total:.3f}"
        )


if __name__ == "__main__":
    main()
