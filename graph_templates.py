import os
import glob
import json
from typing import Dict, Any, List, Optional, Tuple

# Wir nutzen die Helfer aus graph_converter, damit Format identisch ist
from graph_converter import make_node, make_edge, classify_pattern


# -----------------------------
# Hilfsfunktionen zu LBS-IDs
# -----------------------------

def _core_type_and_id(role_id: str) -> Tuple[Optional[str], Optional[str]]:
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
    for d in role_obj.get("T_DESCR", []):
        if d.get("LANG") in ("D", "DE"):
            return d.get("DESCR", "")
    return ""


def _infer_malo_direction(role_id: str, descr_de: str = "") -> Optional[str]:
    rid = (role_id or "").upper()
    txt = (descr_de or "").upper()
    if "EIN" in rid or "EINSP" in rid or "EINSPEIS" in txt or "ERZEUG" in txt:
        return "generation"
    if "BZG" in rid or "BEZUG" in txt or "AUSSPEIS" in txt:
        return "consumption"
    return None


def _infer_melo_function(role_id: str, descr_de: str = "") -> str:
    rid = (role_id or "").upper()
    txt = (descr_de or "").lower()
    if "_HS_" in rid or "hinterschalt" in txt:
        return "H"
    if "differenz" in txt or "saldo" in txt or "_DIF" in rid:
        return "D"
    if "speicher" in txt or "_SPEI" in rid:
        return "S"
    return "N"


# -----------------------------
# 1 LBS-JSON -> 1 Template-Graph
# -----------------------------

def lbsjson_to_template_graph(lbs_json: Dict[str, Any],
                              graph_id: Optional[str] = None) -> Dict[str, Any]:
    # 1) Meta zu T_ROLE aufbauen
    role_meta: Dict[str, Dict[str, Any]] = {}
    for r in lbs_json.get("T_ROLE", []):
        rid = r.get("ID", "")
        descr = _descr_de(r)
        level = r.get("LBS_OBJECT_LEVEL", 0)
        code = r.get("LBS_OBJECT_CODE", 0)
        attrs = {a.get("ATTR_CATEGORY"): a.get("ATTR_VALUE")
                 for a in r.get("T_ATTR", [])}
        role_meta[rid] = {
            "id": rid,
            "descr": descr,
            "level": level,
            "code": code,
            "attrs": attrs,
        }

    # 2) Erste MC / Variante wählen
    mc_list = lbs_json.get("T_MC", [])
    if not mc_list:
        raise ValueError("LBS-JSON ohne T_MC!")
    mc = mc_list[0]
    lbs_code = mc.get("LBS_CODE")
    var_list = mc.get("T_VAR", [])
    if not var_list:
        raise ValueError("LBS-JSON ohne T_VAR in T_MC!")
    var = var_list[0]   # TODO: ggf. irgendwann nach PRIORITY wählen

    # 3) Knoten (Kernobjekte) aufbauen
    nodes_by_id: Dict[str, Dict[str, Any]] = {}

    for r in var.get("T_ROLE", []):
        role_id = r.get("ROLE_ID", "")
        ntype, core_id = _core_type_and_id(role_id)
        if not ntype or not core_id:
            continue

        meta = role_meta.get(core_id, role_meta.get(role_id, {}))
        level = int(meta.get("level", 0))
        code = meta.get("code", 0)
        descr = meta.get("descr", "")

        # reine Hilfsrollen (Level 0, Code 0) ignorieren
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

    nodes: List[Dict[str, Any]] = list(nodes_by_id.values())

    # 4) Kanten mit REL_TYPE
    edges: List[Dict[str, str]] = []
    for rr in var.get("T_ROLEREL", []):
        o1 = rr.get("OBJECT_ID_1", "")
        o2 = rr.get("OBJECT_ID_2", "")
        t1, c1 = _core_type_and_id(o1)
        t2, c2 = _core_type_and_id(o2)
        if not t1 or not t2 or c1 not in nodes_by_id or c2 not in nodes_by_id:
            continue

        rel_type = None
        for att in rr.get("T_OBJRAT", []):
            if att.get("ATTR_CATEGORY") == "REL_TYPE":
                v = (att.get("ATTR_VALUE") or "").strip()
                if v:
                    rel_type = v
                    break

        # heuristische Fallbacks, falls REL_TYPE fehlt
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

        if rel_type not in ("MEMA", "MENE", "MEME", "METR"):
            continue

        edges.append(make_edge(c1, c2, rel_type))

    # 5) Pattern & Graph-Attribute
    malo_count = sum(1 for n in nodes if n["type"] == "MaLo")
    melo_count = sum(1 for n in nodes if n["type"] == "MeLo")
    pattern = classify_pattern(malo_count, melo_count)

    if graph_id is None:
        graph_id = f"catalog-{lbs_code}"

    return {
        "graph_id": graph_id,
        "label": str(lbs_code) if lbs_code is not None else None,
        "nodes": nodes,
        "edges": edges,
        "graph_attrs": {
            "pattern": pattern,
            "malo_count": malo_count,
            "melo_count": melo_count,
            "is_template": True,
            "lbs_code": str(lbs_code) if lbs_code is not None else None,
        }
    }


# -----------------------------
# Alle LBS-Dateien verarbeiten
# -----------------------------

def build_all_templates(lbs_dir: str, out_path: str) -> None:
    """
    Liest alle *.json in lbs_dir ein, konvertiert sie zu Template-Graphen
    und schreibt sie als JSONL (ein Graph pro Zeile).
    """
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
        # BDEW-Exports sind cp1252-kodiert
        lbs_json = json.loads(raw.decode("cp1252"))
        g = lbsjson_to_template_graph(lbs_json)
        graphs.append(g)

    print("Gebaut:", len(graphs), "Template-Graphen")

    with open(out_path, "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print("JSONL geschrieben nach:", out_path)


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))

    # genau dein Ordner aus dem Screenshot:
    lbs_dir = os.path.join(BASE, "data", "lbs_templates")

    # Output-Datei – ich würde sie in data/ legen:
    out_path = os.path.join(BASE, "data", "lbs_templates.jsonl")

    build_all_templates(lbs_dir, out_path)
