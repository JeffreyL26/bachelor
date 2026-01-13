"""graph_templates.py

Erzeugt Soll-/Template-Graphen aus den SAP-LBS-Exporten.

Warum diese Version `_lbs_optionality` nutzt
-------------------------------------------

Die Template-JSONs enthalten einen kuratierten Block `_lbs_optionality` (Min/Max, Starr/Flex,
teils Referenzinformationen und Constraints). In den eigentlichen SAP-Varianten (`T_VAR`) tauchen
optionale/flexible Objekte (insb. TR) jedoch uneinheitlich auf: manche Templates enthalten diese
als feste Rollen/Knoten, andere nicht.

Wenn wir Template-Graphen nur aus `T_VAR` ableiten, wird diese Modellierungs-Ungleichheit zu einem
Artefakt im Datensatz (und nicht zu einem echten fachlichen Unterschied). Deshalb erzeugt dieses
Modul die Template-Graphen primär aus `_lbs_optionality`:

* **Knoten:** genau 1 Knoten pro LBS-Objektcode aus `_lbs_optionality.lbs_objects`.
  Min/Max und Flexibilität werden als Knotenattribute gespeichert (kein Ausrollen!).
* **Kanten:**
  - explizite Rollenbeziehungen aus `T_ROLEREL` werden (falls möglich) auf Objektcodes gemappt,
  - zusätzlich werden explizite `reference_to_melo`-Angaben aus `_lbs_optionality` genutzt.

Für Fälle, in denen die Zuordnung (z.B. TR → welche MeLo?) nicht explizit ist, werden **keine
"erratenen" Kanten** eingefügt, sondern "attachment_rules" im `graph_attrs` abgelegt.

Damit bleibt das Format kompatibel zu `graph_converter.py`, aber die Templates sind konsistent.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Wir nutzen die Helfer aus graph_converter, damit das Format identisch ist
from graph_converter import make_edge, make_node

#TODO: Brauchen wir classify_pattern?


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Semantik aus Rollen (Fallback / Kanten-Mapping)
# ---------------------------------------------------------------------------

def _core_type_and_id(role_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Mappt SAP-Rollen-IDs auf (Knotentyp, "Kern"-ID).

    Hintergrund: In den SAP-Exports gibt es "Hüllen" wie AME_/AMA_ (Anlagerollen), die fachlich
    dieselbe Einheit wie ME_/MA_ beschreiben. Damit nicht doppelt geknotet wird, mappen wir
    AME_x → ME_x und AMA_x → MA_x.
    """
    if not role_id:
        return None, None
    rid = role_id.upper()
    if rid == "NELO":
        return "NeLo", "NELO"
    if rid.startswith("ME_"):
        return "MeLo", role_id
    if rid.startswith("MA_"):
        return "MaLo", role_id
    if rid.startswith(("TR_", "ASR_")):
        return "TR", role_id
    if rid.startswith("AME_"):
        return "MeLo", "ME_" + role_id.split("AME_", 1)[1]
    if rid.startswith("AMA_"):
        return "MaLo", "MA_" + role_id.split("AMA_", 1)[1]
    return None, None


def _descr_de(role_obj: Dict[str, Any]) -> str:
    """Holt die deutsche Beschreibung aus einem T_ROLE-Eintrag."""
    for d in role_obj.get("T_DESCR", []):
        if d.get("LANG") in ("D", "DE"):
            return d.get("DESCR", "")
    return ""


def _infer_malo_direction(role_id: str, descr_de: str = "") -> Optional[str]:
    """Heuristik: erkennt Erzeuger/Verbraucher für MaLo aus Rollen-ID/Beschreibung."""
    rid = (role_id or "").upper()
    txt = (descr_de or "").upper()
    if "EIN" in rid or "EINSP" in rid or "EINSPEIS" in txt or "ERZEUG" in txt:
        return "generation"
    if "BZG" in rid or "BEZUG" in txt or "AUSSPEIS" in txt:
        return "consumption"
    return None


def _infer_melo_function(role_id: str, descr_de: str = "") -> str:
    """Heuristik: erkennt MeLo-Funktion (H/D/S/N) aus Rollen-ID/Beschreibung."""
    rid = (role_id or "").upper()
    txt = (descr_de or "").lower()
    if "_HS_" in rid or "hinterschalt" in txt:
        return "H"
    if "differenz" in txt or "saldo" in txt or "_DIF" in rid:
        return "D"
    if "speicher" in txt or "_SPEI" in rid:
        return "S"
    return "N"


# ---------------------------------------------------------------------------
# Normalisierung: Objektcodes konsistent machen
# ---------------------------------------------------------------------------

_KNOWN_TYPES = {"MaLo", "MeLo", "TR", "NeLo"}
_ALLOWED_REL_TYPES = {"MEMA", "MENE", "MEME", "METR"}

def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_unbounded_max(max_occurs: Any) -> bool:
    if max_occurs is None:
        return False
    if isinstance(max_occurs, str) and max_occurs.strip().upper() == "N":
        return True
    return False


# ---------------------------------------------------------------------------
# Parsing `_lbs_optionality`
# ---------------------------------------------------------------------------

def _parse_lbs_optionality(lbs_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Liest `_lbs_optionality` und normalisiert Codes.

    Rückgabe ist ein Dict mit mindestens:
      - lbs_code
      - lbs_objects: Liste der Objekte (object_code normalisiert)
      - constraints: unverändert/normalisiert wo möglich
    """
    opt = lbs_json.get("_lbs_optionality")
    if not isinstance(opt, dict):
        return None

    # shallow-copy + gezielte Normalisierung
    out: Dict[str, Any] = {k: v for k, v in opt.items() if k != "lbs_objects"}

    objs_norm: List[Dict[str, Any]] = []
    for obj in opt.get("lbs_objects", []) or []:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("object_type")
        if obj_type not in _KNOWN_TYPES:
            continue
        code = obj.get("object_code")
        if not code:
            continue

        obj_norm = dict(obj)
        obj_norm["object_code"] = code
        # Normalisiere Referenzen
        if "reference_to_melo" in obj_norm:
            obj_norm["reference_to_melo"] = obj_norm.get("reference_to_melo")
        objs_norm.append(obj_norm)

    out["lbs_objects"] = objs_norm

    # Constraints (nur Codes normalisieren, Struktur beibehalten)
    constraints = opt.get("constraints")
    if isinstance(constraints, dict):
        c_norm = json.loads(json.dumps(constraints))  # safe deep copy
        c_norm.pop("notes", None)
        # reference_rules
        if isinstance(c_norm.get("reference_rules"), list):
            for rule in c_norm["reference_rules"]:
                if not isinstance(rule, dict):
                    continue
                # rationale entfernen
                rule.pop("rationale", None)
                if "if_object_code" in rule:
                    rule["if_object_code"] = rule.get("if_object_code")
                if "then_reference_to" in rule:
                    rule["then_reference_to"] = rule.get("then_reference_to")
                # manche Regeln verwenden schon reference_to_melo
                if "reference_to_melo" in rule:
                    rule["reference_to_melo"] = rule.get("reference_to_melo")
        # cardinality_constraints
        if isinstance(c_norm.get("cardinality_constraints"), list):
            for cc in c_norm["cardinality_constraints"]:
                if not isinstance(cc, dict):
                    continue
                # rationale entfernen
                cc.pop("rationale", None)
                if "equal_count_between_object_codes" in cc:
                    cc["equal_count_between_object_codes"] = [
                        str(x).strip()
                        for x in (cc.get("equal_count_between_object_codes") or [])
                        if x is not None and str(x).strip()
                    ]
        out["constraints"] = c_norm
    return out


# ---------------------------------------------------------------------------
# Template-Graph aufbauen
# ---------------------------------------------------------------------------

def _build_nodes_from_optionality(opt: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Erzeugt Knoten aus `_lbs_optionality.lbs_objects`.

    Rückgabe:
      - nodes: Liste `make_node(...)`
      - code_to_type: Mapping object_code -> node_type
    """
    nodes: List[Dict[str, Any]] = []
    code_to_type: Dict[str, str] = {}

    for obj in opt.get("lbs_objects", []) or []:
        obj_type = obj.get("object_type")
        if obj_type not in _KNOWN_TYPES:
            continue
        code = obj.get("object_code")
        if not isinstance(code, str) or not code:
            continue

        level = _as_int(obj.get("level"), 0)
        min_occurs = _as_int(obj.get("min_occurs"), 0)
        max_occurs = obj.get("max_occurs")
        flexibility = obj.get("flexibility")

        attrs: Dict[str, Any] = {
            "level": level,
            "object_code": code,
            "min_occurs": min_occurs,
            "max_occurs": max_occurs,
            "flexibility": flexibility,
            "optional": True if min_occurs == 0 else False,
        }

        # Domain-Features
        if obj_type == "MaLo":
            # bevorzugt aus _lbs_optionality; fallback auf direction_hint
            direction = obj.get("direction") or obj.get("direction_hint")
            if direction:
                attrs["direction"] = direction
        elif obj_type == "MeLo":
            fn = obj.get("melo_function") or "N"
            attrs["function"] = fn
            attrs["dynamic"] = 1 if fn == "H" else 0
            # direction_hint kann in Templates vorkommen; optional speichern
            if obj.get("direction_hint"):
                attrs["direction_hint"] = obj.get("direction_hint")
        elif obj_type == "TR":
            tr_dir = obj.get("tr_direction")
            if tr_dir:
                attrs["tr_direction"] = tr_dir
                # Für einheitliche Feature-Kodierung: direction als Alias
                attrs["direction"] = tr_dir
        # NeLo: keine zusätzlichen Features

        nodes.append(make_node(code, obj_type, attrs))
        code_to_type[code] = obj_type

    return nodes, code_to_type


def _orient_and_add_edge(edges_set: set, src: str, dst: str, rel: str, code_to_type: Dict[str, str]) -> None:
    """Fügt eine Kante hinzu und normalisiert die Richtung (MeLo als Quelle, wo sinnvoll)."""
    if rel not in _ALLOWED_REL_TYPES:
        return
    if src == dst and rel == "MEME":
        return

    t_src = code_to_type.get(src)
    t_dst = code_to_type.get(dst)

    # Richtung konsistent halten: MeLo -> (MaLo/TR/NeLo)
    if rel == "MEMA" and t_src == "MaLo" and t_dst == "MeLo":
        src, dst = dst, src
    elif rel == "METR" and t_src == "TR" and t_dst == "MeLo":
        src, dst = dst, src
    elif rel == "MENE" and t_src == "NeLo" and t_dst == "MeLo":
        src, dst = dst, src

    edges_set.add((src, dst, rel))


def _edges_from_sap_relations(
    var: Dict[str, Any],
    role_meta: Dict[str, Dict[str, Any]],
    code_to_type: Dict[str, str],
) -> set:
    """Extrahiert Kanten aus `T_ROLEREL` und mappt sie auf Objektcodes (falls möglich)."""
    edges_set: set = set()

    for rr in var.get("T_ROLEREL", []) or []:
        rid1 = rr.get("OBJECT_ID_1", "")
        rid2 = rr.get("OBJECT_ID_2", "")

        t1, core1 = _core_type_and_id(rid1)
        t2, core2 = _core_type_and_id(rid2)
        if not t1 or not t2 or not core1 or not core2:
            continue

        meta1 = role_meta.get(core1, role_meta.get(rid1, {}))
        meta2 = role_meta.get(core2, role_meta.get(rid2, {}))
        code1 = meta1.get("code")
        code2 = meta2.get("code")
        if not code1 or not code2:
            continue
        if code1 not in code_to_type or code2 not in code_to_type:
            # Rolle referenziert Objekte, die nicht im _lbs_optionality-Set modelliert sind
            continue

        # REL_TYPE auslesen
        rel_type = None
        for att in rr.get("T_OBJRAT", []) or []:
            if att.get("ATTR_CATEGORY") == "REL_TYPE":
                v = (att.get("ATTR_VALUE") or "").strip()
                if v:
                    rel_type = v
                    break

        # Fallbacks
        if rel_type is None:
            if t1 == "MeLo" and t2 == "MaLo":
                rel_type = "MEMA"
            elif t1 == "MeLo" and t2 == "NeLo":
                rel_type = "MENE"
            elif (t1 == "MeLo" and t2 == "TR") or (t1 == "TR" and t2 == "MeLo"):
                rel_type = "METR"
            elif t1 == "MeLo" and t2 == "MeLo":
                rel_type = "MEME"
            else:
                continue

        if rel_type not in _ALLOWED_REL_TYPES:
            continue

        _orient_and_add_edge(edges_set, code1, code2, rel_type, code_to_type)

    return edges_set


def _edges_and_rules_from_optionality(opt: Dict[str, Any], code_to_type: Dict[str, str]) -> Tuple[set, List[Dict[str, Any]]]:
    """Erzeugt Kanten aus `_lbs_optionality`-Referenzen und zusätzlich attachment_rules.

    * Wenn `reference_to_melo` explizit vorhanden ist, wird daraus eine Kante gebaut.
    * Wenn nicht vorhanden ist, wird **nur dann** automatisch eine Kante gebaut, wenn die
      Zielmenge eindeutig ist (genau eine passende MeLo).
    * Andernfalls wird eine "attachment_rule" abgelegt.
    """
    edges_set: set = set()
    attachment_rules: List[Dict[str, Any]] = []

    objs = opt.get("lbs_objects", []) or []
    melos = [o for o in objs if o.get("object_type") == "MeLo"]
    melos_by_level: Dict[int, List[str]] = {}
    for m in melos:
        lvl = _as_int(m.get("level"), 0)
        melos_by_level.setdefault(lvl, []).append(m.get("object_code"))

    def candidates_for_level(level: int) -> List[str]:
        c = [c for c in (melos_by_level.get(level) or []) if c in code_to_type]
        if c:
            return c
        # Fallback: alle MeLo, falls auf gleicher Ebene keine existiert
        return [m.get("object_code") for m in melos if m.get("object_code") in code_to_type]

    for obj in objs:
        obj_type = obj.get("object_type")
        if obj_type not in _KNOWN_TYPES:
            continue
        code = obj.get("object_code")
        if code not in code_to_type:
            continue

        refs_raw = obj.get("reference_to_melo")

        # `reference_to_melo` kann None, ein einzelner Code oder eine Liste sein
        if not refs_raw:
            refs: List[str] = []
        elif isinstance(refs_raw, list):
            refs = refs_raw
        else:
            refs = [refs_raw]

        # sanitizen + nur valide MeLo-Targets behalten
        refs = [
            str(r).strip()
            for r in refs
            if r is not None and str(r).strip()
               and r in code_to_type
               and code_to_type.get(r) == "MeLo"
        ]

        # Mapping Objekt->Relation
        rel_for_type = {
            "TR": "METR",
            "MaLo": "MEMA",
            "NeLo": "MENE",
        }.get(obj_type)

        if rel_for_type and refs:
            for mcode in refs:
                _orient_and_add_edge(edges_set, mcode, code, rel_for_type, code_to_type)
            continue

        # Keine explizite Referenz: nur bei eindeutiger Zielmenge automatisch verbinden
        if rel_for_type:
            lvl = _as_int(obj.get("level"), 0)
            cands = candidates_for_level(lvl)

            if len(cands) == 1:
                _orient_and_add_edge(edges_set, cands[0], code, rel_for_type, code_to_type)
            elif len(cands) >= 2:
                attachment_rules.append({
                    "object_code": code,
                    "object_type": obj_type,
                    "rel": rel_for_type,
                    "target_type": "MeLo",
                    "target_candidates": cands,
                    "rule": "attach_to_one_of_candidates",
                    "reason": "No explicit reference_to_melo; multiple candidate MeLo nodes.",
                })

    return edges_set, attachment_rules


def _min_max_counts(opt: Dict[str, Any], obj_type: str) -> Tuple[int, Optional[int]]:
    """Summiert min_occurs und max_occurs (wenn endlich) über alle Objekte eines Typs."""
    mins = 0
    max_sum = 0
    unbounded = False
    for obj in opt.get("lbs_objects", []) or []:
        if obj.get("object_type") != obj_type:
            continue
        mins += _as_int(obj.get("min_occurs"), 0)
        mx = obj.get("max_occurs")
        if _is_unbounded_max(mx):
            unbounded = True
        else:
            max_sum += _as_int(mx, 0)
    return mins, None if unbounded else max_sum


def lbsjson_to_template_graph(lbs_json: Dict[str, Any], graph_id: Optional[str] = None) -> Dict[str, Any]:
    """Erstellt aus einem LBS-Export einen Template-Graphen.

    * Primärquelle für Knoten & Min/Max: `_lbs_optionality`
    * Ergänzung für Kanten: `T_ROLEREL` (sofern auf Objektcodes mappbar)
    """

    # Globale Rollentabelle als Nachschlagewerk
    role_meta: Dict[str, Dict[str, Any]] = {}
    for r in lbs_json.get("T_ROLE", []) or []:
        rid = r.get("ID", "")
        descr = _descr_de(r)
        level = r.get("LBS_OBJECT_LEVEL", 0)
        code = r.get("LBS_OBJECT_CODE", 0)
        attrs = {a.get("ATTR_CATEGORY"): a.get("ATTR_VALUE") for a in r.get("T_ATTR", []) or []}
        role_meta[rid] = {"id": rid, "descr": descr, "level": level, "code": code, "attrs": attrs}

    mc_list = lbs_json.get("T_MC", [])
    if not mc_list:
        raise ValueError("LBS-JSON ohne Messkonzeptkonfiguration (MC)!")
    mc = mc_list[0]
    lbs_code = mc.get("LBS_CODE")
    var_list = mc.get("T_VAR", [])
    if not var_list:
        raise ValueError("LBS-JSON ohne T_VAR in Messkonzeptkonfiguration (MC)!")
    var = var_list[0]  # TODO: ggf. PRIORITY berücksichtigen

    opt = _parse_lbs_optionality(lbs_json)

    # ------------------------------------------------------------------
    # NEUER Pfad: Optionale/Flexible Objekte konsistent über _lbs_optionality
    # ------------------------------------------------------------------
    if opt is not None and opt.get("lbs_objects"):
        nodes, code_to_type = _build_nodes_from_optionality(opt)

        # Kanten aus optionality-Refs + SAP-Relationen
        edges_set_1, attachment_rules = _edges_and_rules_from_optionality(opt, code_to_type)
        edges_set_2 = _edges_from_sap_relations(var, role_meta, code_to_type)
        edges_set = set().union(edges_set_1, edges_set_2)
        edges = [make_edge(src, dst, rel) for (src, dst, rel) in sorted(edges_set)]

        # Pattern / Counts: sinnvollerweise über Min-Occurs (Template = Constraints, nicht instanziiert)
        malo_min, malo_max = _min_max_counts(opt, "MaLo")
        melo_min, melo_max = _min_max_counts(opt, "MeLo")
        tr_min, tr_max = _min_max_counts(opt, "TR")
        nelo_min, nelo_max = _min_max_counts(opt, "NeLo")


        # Counts (einmal: Anzahl Objekt-TYPEN im Template-Graph, einmal: Minimal-Occurrences)
        malo_node_types = sum(1 for n in nodes if n["type"] == "MaLo")
        melo_node_types = sum(1 for n in nodes if n["type"] == "MeLo")
        tr_node_types = sum(1 for n in nodes if n["type"] == "TR")
        nelo_node_types = sum(1 for n in nodes if n["type"] == "NeLo")

        if graph_id is None:
            graph_id = f"catalog-{lbs_code}"

        return {
            "graph_id": graph_id,
            "label": str(lbs_code) if lbs_code is not None else None,
            "nodes": nodes,
            "edges": edges,
            "graph_attrs": {
                "malo_min": malo_min,
                "melo_min": melo_min,
                "tr_min": tr_min,
                "nelo_min": nelo_min,
                "malo_max": malo_max,
                "melo_max": melo_max,
                "tr_max": tr_max,
                "nelo_max": nelo_max,
                "malo_node_types": malo_node_types,
                "melo_node_types": melo_node_types,
                "tr_node_types": tr_node_types,
                "nelo_node_types": nelo_node_types,
                "is_template": True,
                "lbs_code": str(lbs_code) if lbs_code is not None else None,
                # `lbs_objects` ist bereits 1:1 in den Knoten (`nodes[].attrs`) repräsentiert.
                # Deshalb führen wir es nicht noch einmal als Block mit.
                "optionality_constraints": opt.get("constraints"),
                # Regeln für unklare Zuordnungen (nicht "raten")
                "attachment_rules": attachment_rules,
            },
        }

    # ------------------------------------------------------------------
    # FALLBACK: alter Pfad (nur SAP-Rollen), falls `_lbs_optionality` fehlt
    # ------------------------------------------------------------------

    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    for r in var.get("T_ROLE", []) or []:
        role_id = r.get("ROLE_ID", "")
        ntype, core_id = _core_type_and_id(role_id)
        if not ntype or not core_id:
            continue

        meta = role_meta.get(core_id, role_meta.get(role_id, {}))
        level = int(meta.get("level", 0) or 0)
        code = meta.get("code", 0) or 0
        descr = meta.get("descr", "")
        if level == 0 and code == 0:
            continue

        attrs: Dict[str, Any] = {"level": level}
        if ntype == "MaLo":
            direction = _infer_malo_direction(core_id, descr)
            if direction:
                attrs["direction"] = direction
        elif ntype == "MeLo":
            fn = _infer_melo_function(core_id, descr)
            attrs["function"] = fn
            attrs["dynamic"] = 1 if fn == "H" else 0

        if core_id not in nodes_by_id:
            nodes_by_id[core_id] = make_node(core_id, ntype, attrs)

    nodes = list(nodes_by_id.values())

    edges: List[Dict[str, str]] = []
    for rr in var.get("T_ROLEREL", []) or []:
        t1, c1 = _core_type_and_id(rr.get("OBJECT_ID_1", ""))
        t2, c2 = _core_type_and_id(rr.get("OBJECT_ID_2", ""))
        if not t1 or not t2 or c1 not in nodes_by_id or c2 not in nodes_by_id:
            continue

        rel_type = None
        for att in rr.get("T_OBJRAT", []) or []:
            if att.get("ATTR_CATEGORY") == "REL_TYPE":
                v = (att.get("ATTR_VALUE") or "").strip()
                if v:
                    rel_type = v
                    break

        if rel_type is None:
            if t1 == "MeLo" and t2 == "MaLo":
                rel_type = "MEMA"
            elif t1 == "MeLo" and t2 == "NeLo":
                rel_type = "MENE"
            elif (t1 == "MeLo" and t2 == "TR") or (t1 == "TR" and t2 == "MeLo"):
                rel_type = "METR"
            elif t1 == "MeLo" and t2 == "MeLo":
                rel_type = "MEME"
            else:
                continue

        if rel_type not in _ALLOWED_REL_TYPES:
            continue
        if rel_type == "MEME" and c1 == c2:
            continue
        edges.append(make_edge(c1, c2, rel_type))

    malo_count = sum(1 for n in nodes if n["type"] == "MaLo")
    melo_count = sum(1 for n in nodes if n["type"] == "MeLo")

    if graph_id is None:
        graph_id = f"catalog-{lbs_code}"

    return {
        "graph_id": graph_id,
        "label": str(lbs_code) if lbs_code is not None else None,
        "nodes": nodes,
        "edges": edges,
        "graph_attrs": {
            "malo_count": malo_count,
            "melo_count": melo_count,
            "is_template": True,
            "lbs_code": str(lbs_code) if lbs_code is not None else None,
        },
    }


def build_all_templates(lbs_dir: str, out_path: str) -> None:
    """Liest alle JSON in `lbs_dir`, konvertiert sie zu Template-Graphen und schreibt JSONL."""
    print("LBS-Verzeichnis:", lbs_dir)
    pattern = os.path.join(lbs_dir, "*.json")
    files = glob.glob(pattern)
    print("Glob-Pattern:", pattern)
    print("Gefundene Dateien:", len(files))
    for p in files:
        print("  -", os.path.basename(p))

    graphs: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "rb") as f:
            raw = f.read()
        lbs_json = json.loads(raw.decode("utf-8"))
        graphs.append(lbsjson_to_template_graph(lbs_json))

    print("Gebaut:", len(graphs), "Template-Graphen (Soll)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print("JSONL geschrieben nach:", out_path)


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    lbs_dir = os.path.join(BASE, "data", "lbs_templates")
    out_path = os.path.join(BASE, "data", "lbs_soll_graphs.jsonl")
    build_all_templates(lbs_dir, out_path)
