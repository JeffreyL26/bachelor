"""
TODO: In Thesis erwähnen
Nutzung von eigens gebauten _lbs_optionality

Die Template-JSONs enthalten einen selbst gebauten Block _lbs_optionality (Min/Max, Starr/Flexibel,
Referenzinformationen und Constraints). In den eigentlichen SAP-Varianten (T_VAR) tauchen
optionale/flexible Objekte, insbesonders TR, uneinheitlich auf: manche Templates enthalten diese
als feste Rollen/Knoten, andere nicht. Das hat mich einen eigenen Block für jedes Template entwerfen
lassen, die alle im Einklang mit den BDEW-Codelisten sind und Metadata für die Baselines liefern.
Graphen aus TR_VAR abzuleiten wöre uneinheitlich.

Knoten aus genau 1 Knoten pro LBS-Objektcode aus _lbs_optionality.lbs_objects
Min/Max und Flexibilität werden als Knotenattribute gespeichert.
Der Template-Graph beschreibt Constraints, nicht eine konkrete Instanz.

Kanten
  - explizite Rollenbeziehungen aus T_ROLEREL werden (falls möglich) auf Objektcodes gemappt
  - explizite reference_to_melo-Angaben aus _lbs_optionality genutzt.

Für Fälle, in denen die Zuordnung (z.B. TR → welche MeLo?) nicht explizit ist, werden keine
erratenen Kanten eingefügt, sondern "attachment_rules" im graph_attrs abgelegt.

Damit bleibt das Format kompatibel zu graph_converter.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Wir nutzen die Helfer aus graph_converter, damit das Format identisch ist
from graph_converter import make_edge, make_node


# ------------------------------
# MAPPING DER ROLLEN
# ------------------------------

def _id_and_type_mapper(role_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Mappt SAP-Rollen-IDs auf (Knotentyp, "Kern"-ID)

    In den SAP-Exports gibt es Präfixe wie AME_/AMA_ (Anlagerollen), die fachlich
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


def _de_beschreibung(rolle: Dict[str, Any]) -> str:
    """
    Holt die deutsche Beschreibung aus einem T_ROLE-Eintrag
    """
    for d in rolle.get("T_DESCR", []):
        if d.get("LANG") in ("D", "DE"):
            return d.get("DESCR", "")
    return ""


def _malo_direction_finder(role_id: str, de_beschr: str = "") -> Optional[str]:
    """
    Erkennt Erzeuger/Verbraucher für MaLo aus Rollen-ID/Beschreibung
    """
    rid = (role_id or "").upper()
    txt = (de_beschr or "").upper()
    if "EIN" in rid or "EINSP" in rid or "EINSPEIS" in txt or "ERZEUG" in txt:
        return "generation"
    if "BZG" in rid or "BEZUG" in txt or "AUSSPEIS" in txt:
        return "consumption"
    return None


def _melo_function_getter(role_id: str, de_beschr: str = "") -> str:
    """
    Erkennt MeLo-Funktion Hinterschaltung/Speicher/Differenzmessung/Normal aus Rollen-ID/Beschreibung
    """
    rid = (role_id or "").upper()
    txt = (de_beschr or "").lower()
    if "_HS_" in rid or "hinterschalt" in txt:
        return "H"
    if "differenz" in txt or "saldo" in txt or "_DIF" in rid:
        return "D"
    if "speicher" in txt or "_SPEI" in rid:
        return "S"
    return "N"


# ------------------------------
# KONSISTENT MACHEN
# ------------------------------

_NODE_TYPES = {"MaLo", "MeLo", "TR", "NeLo"}
_REL_TYPES = {"MEMA", "MENE", "MEME", "METR"}

def _int_maker(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_no_max_bound(max_occurs: Any) -> bool:
    if max_occurs is None:
        return False
    if isinstance(max_occurs, str) and max_occurs.strip().upper() == "N":
        return True
    return False


# ------------------------------
# OPTIONALITY-BLOCK EINLESEN
# ------------------------------

def _parse_lbs_optionality(lbs_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Liest _lbs_optionality und normalisiert Codes

    :param: Dict aus Code mit Informationen
    :return: Dict mit mindestens:
      - lbs_code
      - lbs_objects: Liste der Objekte (object_code normalisiert)
      - constraints: unverändert/normalisiert wo möglich
    """
    optionalitaeten = lbs_json.get("_lbs_optionality")
    if not isinstance(optionalitaeten, dict):
        return None

    # lbs_obkects auslassen und nur den optionality-Block gezielt normalisieren
    out: Dict[str, Any] = {k: v for k, v in optionalitaeten.items() if k != "lbs_objects"}

    normalisierte_daten: List[Dict[str, Any]] = []
    for obj in optionalitaeten.get("lbs_objects", []) or []:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("object_type")
        if obj_type not in _NODE_TYPES:
            continue
        code = obj.get("object_code")
        if not code:
            continue

        obj_norm = dict(obj)
        obj_norm["object_code"] = code
        # Normalisiere Referenzen
        if "reference_to_melo" in obj_norm:
            obj_norm["reference_to_melo"] = obj_norm.get("reference_to_melo")
        normalisierte_daten.append(obj_norm)

    out["lbs_objects"] = normalisierte_daten

    # Constraints (nur Codes normalisieren, Struktur beibehalten)
    constraints = optionalitaeten.get("constraints")
    if isinstance(constraints, dict):
        constraint_normalisiert = json.loads(json.dumps(constraints)) # Kopie
        constraint_normalisiert.pop("notes", None)
        # reference_rules
        if isinstance(constraint_normalisiert.get("reference_rules"), list):
            for rule in constraint_normalisiert["reference_rules"]:
                if not isinstance(rule, dict):
                    continue
                if "if_object_code" in rule:
                    rule["if_object_code"] = rule.get("if_object_code")
                if "then_reference_to" in rule:
                    rule["then_reference_to"] = rule.get("then_reference_to")
                # manche Regeln verwenden schon reference_to_melo
                if "reference_to_melo" in rule:
                    rule["reference_to_melo"] = rule.get("reference_to_melo")
        # cardinality_constraints
        if isinstance(constraint_normalisiert.get("cardinality_constraints"), list):
            for cc in constraint_normalisiert["cardinality_constraints"]:
                if not isinstance(cc, dict):
                    continue
                if "equal_count_between_object_codes" in cc:
                    cc["equal_count_between_object_codes"] = [
                        str(x).strip()
                        for x in (cc.get("equal_count_between_object_codes") or [])
                        if x is not None and str(x).strip()
                    ]
        out["constraints"] = constraint_normalisiert
    return out


# ------------------------------
# GRAPH-BAU
# ------------------------------

def _build_nodes_optionality_block(opt: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Knoten aus _lbs_optionality.lbs_objects herstellen

    :returns:
      - nodes: Liste make_node(...)
      - code_to_type: Mapping object_code -> node_type
    """
    nodes: List[Dict[str, Any]] = []
    code_to_type: Dict[str, str] = {}

    for obj in opt.get("lbs_objects", []) or []:
        obj_type = obj.get("object_type")
        if obj_type not in _NODE_TYPES:
            continue
        code = obj.get("object_code")
        if not isinstance(code, str) or not code:
            continue

        level = _int_maker(obj.get("level"), 0)
        min_occurs = _int_maker(obj.get("min_occurs"), 0)
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
            # bevorzugt aus _lbs_optionality, ansonsten direction
            direction = obj.get("direction")
            if direction:
                attrs["direction"] = direction
        elif obj_type == "MeLo":
            fn = obj.get("melo_function") or "N"
            attrs["function"] = fn
            attrs["dynamic"] = 1 if fn == "H" else 0
            direction = obj.get("direction")
            if direction:
                attrs["direction"] = direction
        elif obj_type == "TR":
            tr_dir = obj.get("direction")
            if tr_dir:
                # Für einheitliche Feature-Kodierung: direction als Alias
                attrs["direction"] = tr_dir

        nodes.append(make_node(code, obj_type, attrs))
        code_to_type[code] = obj_type

    return nodes, code_to_type


def _edge_direction_maker(edges_set: set, src: str, dst: str, rel: str, code_to_type: Dict[str, str]) -> None:
    """
    Fügt eine Kante hinzu und normalisiert die Richtung (MeLo als Quelle, wo sinnvoll)
    """
    if rel not in _REL_TYPES:
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


def _edges_aus_export(
    var: Dict[str, Any],
    role_meta: Dict[str, Dict[str, Any]],
    code_to_type: Dict[str, str],
) -> set:
    """
    Extrahiert Kanten aus T_ROLEREL und mappt sie auf Objektcodes
    """
    edges_set: set = set()

    for rr in var.get("T_ROLEREL", []) or []:
        rid1 = rr.get("OBJECT_ID_1", "")
        rid2 = rr.get("OBJECT_ID_2", "")

        t1, core1 = _id_and_type_mapper(rid1)
        t2, core2 = _id_and_type_mapper(rid2)
        if not t1 or not t2 or not core1 or not core2:
            continue

        meta1 = role_meta.get(core1, role_meta.get(rid1, {}))
        meta2 = role_meta.get(core2, role_meta.get(rid2, {}))
        code1 = meta1.get("code")
        code2 = meta2.get("code")
        if not code1 or not code2:
            continue
        if code1 not in code_to_type or code2 not in code_to_type:
            # Rolle referenziert Objekte, die nicht in _lbs_optionality-Set sind
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

        if rel_type not in _REL_TYPES:
            continue

        _edge_direction_maker(edges_set, code1, code2, rel_type, code_to_type)

    return edges_set


def _edges_regeln_aus_optionality(opt: Dict[str, Any], code_to_type: Dict[str, str]) -> Tuple[set, List[Dict[str, Any]]]:
    """
    Erzeugt Kanten aus `_lbs_optionality`-Referenzen und zusätzlich attachment_rules

    Wenn reference_to_melo vorhanden ist, wird daraus eine Kante gebaut
    Wenn nicht vorhanden ist, wird nur dann automatisch eine Kante gebaut, wenn genau eine passende MeLo
    Andernfalls wird eine "attachment_rule" abgelegt
    """
    edges_set: set = set()
    attachment_rules: List[Dict[str, Any]] = []

    objs = opt.get("lbs_objects", []) or []
    melos = [o for o in objs if o.get("object_type") == "MeLo"]
    melos_nach_level: Dict[int, List[str]] = {}

    for m in melos:
        lvl = _int_maker(m.get("level"), 0)
        melos_nach_level.setdefault(lvl, []).append(m.get("object_code"))

    def melo_pro_level(level: int) -> List[str]:
        c = [c for c in (melos_nach_level.get(level) or []) if c in code_to_type]
        if c:
            return c
        # Fallback: alle MeLo, falls auf gleicher Ebene keine existiert
        return [m.get("object_code") for m in melos if m.get("object_code") in code_to_type]

    for obj in objs:
        obj_type = obj.get("object_type")
        if obj_type not in _NODE_TYPES:
            continue
        code = obj.get("object_code")
        if code not in code_to_type:
            continue

        referenzen = obj.get("reference_to_melo")

        # reference_to_melo kann None, ein einzelner Code oder eine Liste sein
        if not referenzen:
            refs: List[str] = []
        elif isinstance(referenzen, list):
            refs = referenzen
        else:
            refs = [referenzen]

        #Nur valide MeLo-Targets behalten
        refs = [
            str(r).strip()
            for r in refs
            if r is not None and str(r).strip()
               and r in code_to_type
               and code_to_type.get(r) == "MeLo"
        ]

        #Mapping
        relation_pro_typ = {
            "TR": "METR",
            "MaLo": "MEMA",
            "NeLo": "MENE",
        }.get(obj_type)

        if relation_pro_typ and refs:
            for mcode in refs:
                _edge_direction_maker(edges_set, mcode, code, relation_pro_typ, code_to_type)
            continue

        #Nur bei eindeutiger Zielmenge automatisch verbinden
        if relation_pro_typ:
            lvl = _int_maker(obj.get("level"), 0)
            candidates = melo_pro_level(lvl)

            if len(candidates) == 1:
                _edge_direction_maker(edges_set, candidates[0], code, relation_pro_typ, code_to_type)
            elif len(candidates) >= 2:
                attachment_rules.append({
                    "object_code": code,
                    "object_type": obj_type,
                    "rel": relation_pro_typ,
                    "target_type": "MeLo",
                    "target_candidates": candidates,
                    "rule": "attach_to_one_of_candidates",
                })

    return edges_set, attachment_rules


def _min_max_counts(opt: Dict[str, Any], obj_type: str) -> Tuple[int, Optional[int]]:
    """
    Summiert min_occurs und max_occurs (wenn endlich) über alle Objekte eines Typs
    """
    mins = 0
    max_sum = 0
    unbounded = False
    for obj in opt.get("lbs_objects", []) or []:
        if obj.get("object_type") != obj_type:
            continue
        mins += _int_maker(obj.get("min_occurs"), 0)
        mx = obj.get("max_occurs")
        if _is_no_max_bound(mx):
            unbounded = True
        else:
            max_sum += _int_maker(mx, 0)
    return mins, None if unbounded else max_sum


def json_zu_template(lbs_json: Dict[str, Any], graph_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Erstellt aus einem LBS-Export einen Template-Graphen

    Quelle für Knoten & Min/Max: _lbs_optionality
    Ergänzung für Kanten: T_ROLEREL (sofern auf Objektcodes mappbar)
    """

    #Rollentabelle als Nachschlagewerk
    role_meta: Dict[str, Dict[str, Any]] = {}
    for r in lbs_json.get("T_ROLE", []) or []:
        rid = r.get("ID", "")
        descr = _de_beschreibung(r)
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
    var = var_list[0]

    opt = _parse_lbs_optionality(lbs_json)

    #Optionale Objekte lieber konsistent über lbs_optionality
    if opt is not None and opt.get("lbs_objects"):
        nodes, code_to_type = _build_nodes_optionality_block(opt)

        #Kanten aus optionality-Referenzen & SAP-Relationen
        kanten_set1, attachment_rules = _edges_regeln_aus_optionality(opt, code_to_type)
        kanten_set2 = _edges_aus_export(var, role_meta, code_to_type)
        kanten_set = set().union(kanten_set1, kanten_set2)
        edges = [make_edge(src, dst, rel) for (src, dst, rel) in sorted(kanten_set)]

        #Counts über Min-Occurs
        malo_min, malo_max = _min_max_counts(opt, "MaLo")
        melo_min, melo_max = _min_max_counts(opt, "MeLo")
        tr_min, tr_max = _min_max_counts(opt, "TR")
        nelo_min, nelo_max = _min_max_counts(opt, "NeLo")

        #Counts (Anzahl Objekttypen im Template und min Occurrences)
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
                #lbs_objects ist bereits 1:1 in den Knoten (nodes[].attrs) repräsentiert
                "optionality_constraints": opt.get("constraints"),
                #Regeln für unklare Zuordnungen
                "attachment_rules": attachment_rules,
            },
        }


    #Als Alternative den alten Pfad (nur SAP-Rollen), falls _lbs_optionality fehl
    nodes_ueber_id: Dict[str, Dict[str, Any]] = {}
    for r in var.get("T_ROLE", []) or []:
        role_id = r.get("ROLE_ID", "")
        node_type, core_id = _id_and_type_mapper(role_id)
        if not node_type or not core_id:
            continue

        meta = role_meta.get(core_id, role_meta.get(role_id, {}))
        level = int(meta.get("level", 0) or 0)
        code = meta.get("code", 0) or 0
        descr = meta.get("descr", "")
        if level == 0 and code == 0:
            continue

        attrs: Dict[str, Any] = {"level": level}
        if node_type == "MaLo":
            direction = _malo_direction_finder(core_id, descr)
            if direction:
                attrs["direction"] = direction
        elif node_type == "MeLo":
            fn = _melo_function_getter(core_id, descr)
            attrs["function"] = fn
            attrs["dynamic"] = 1 if fn == "H" else 0

        if core_id not in nodes_ueber_id:
            nodes_ueber_id[core_id] = make_node(core_id, node_type, attrs)

    nodes = list(nodes_ueber_id.values())

    edges: List[Dict[str, str]] = []
    for rolerel in var.get("T_ROLEREL", []) or []:
        t1, c1 = _id_and_type_mapper(rolerel.get("OBJECT_ID_1", ""))
        t2, c2 = _id_and_type_mapper(rolerel.get("OBJECT_ID_2", ""))
        if not t1 or not t2 or c1 not in nodes_ueber_id or c2 not in nodes_ueber_id:
            continue

        rel_type = None
        for att in rolerel.get("T_OBJRAT", []) or []:
            if att.get("ATTR_CATEGORY") == "REL_TYPE":
                v = (att.get("ATTR_VALUE") or "").strip()
                if v:
                    rel_type = v
                    break

        #Logische Ableitung
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

        if rel_type not in _REL_TYPES:
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
    """
    Liest alle JSON in lbs_dir, konvertiert sie zu Template-Graphen und schreibt JSONL
    """
    print("LBS-Directory:", lbs_dir)
    pattern = os.path.join(lbs_dir, "*.json")
    files = glob.glob(pattern)
    print("Glob-Pattern:", pattern)
    print("Found files:", len(files))
    for p in files:
        print("  -", os.path.basename(p))

    graphs: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "rb") as f:
            raw = f.read()
        lbs_json = json.loads(raw.decode("utf-8"))
        graphs.append(json_zu_template(lbs_json))

    print("Built:", len(graphs), "Template-Graphs (Soll)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print("JSONL written to:", out_path)


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    lbs_dir = os.path.join(BASE, "data", "lbs_templates")
    out_path = os.path.join(BASE, "data", "lbs_soll_graphs.jsonl")
    build_all_templates(lbs_dir, out_path)
