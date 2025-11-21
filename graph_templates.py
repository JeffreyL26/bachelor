import json
from typing import Dict, Any, List, Tuple

# --- Helper: ID -> (type, core_id) ------------------------------------------

def _core_type_and_id(role_id: str) -> Tuple[str | None, str | None]:
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

def _infer_malo_direction(role_id: str, descr_de: str = "") -> str | None:
    rid = (role_id or "").upper()
    txt = (descr_de or "").upper()
    if "EIN" in rid or "EINSP" in rid or "EINSPEIS" in txt or "ERZEUG" in txt:
        return "generation"
    if "BZG" in rid or "BEZUG" in txt or "AUSSPEIS" in txt or "AUSSPEISUNG" in txt:
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

def make_node(node_id: str, ntype: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": node_id, "type": ntype, "attrs": attrs}

def make_edge(src: str, dst: str, rel: str) -> Dict[str, str]:
    return {"src": src, "dst": dst, "rel": rel}

def classify_pattern(malo_count: int, melo_count: int) -> str:
    # vorsichtig: hier semantisch konsistent bleiben!
    if malo_count == 1 and melo_count == 1:
        return "1:1"
    if malo_count == 1 and melo_count == 2:
        return "1:2"
    if malo_count == 2 and melo_count == 1:
        return "2:1"
    if malo_count == 2 and melo_count == 2:
        return "2:2"
    return f"{malo_count}:{melo_count}"

# --- Kernfunktion: BDEW-LBS -> Template-Graph -------------------------------

def lbsjson_to_template_graph(lbs_json: Dict[str, Any],
                              graph_id: str | None = None) -> Dict[str, Any]:
    """
    Formt ein BDEW-LBS-JSON (T_ROLE / T_MC / T_VAR / T_ROLEREL)
    in das gleiche Graph-Format wie build_graphs() für Ist-Bündel:
    {
      "graph_id": ...,
      "label": <LBS_CODE>,
      "nodes": [...],
      "edges": [...],
      "graph_attrs": {...}
    }
    """
    # 1) Meta zu T_ROLE sammeln
    role_meta: Dict[str, Dict[str, Any]] = {}
    for r in lbs_json.get("T_ROLE", []):
        rid = r.get("ID", "")
        descr = _descr_de(r)
        level = r.get("LBS_OBJECT_LEVEL", 0)
        code  = r.get("LBS_OBJECT_CODE", 0)
        attrs = {a.get("ATTR_CATEGORY"): a.get("ATTR_VALUE")
                 for a in r.get("T_ATTR", [])}
        role_meta[rid] = {
            "id": rid,
            "descr": descr,
            "level": level,
            "code": code,
            "attrs": attrs,
        }

    # 2) Wir nehmen zunächst die erste MC-/Variante
    mc_list = lbs_json.get("T_MC", [])
    if not mc_list:
        raise ValueError("LBS-JSON ohne T_MC!")
    mc = mc_list[0]
    lbs_code = mc.get("LBS_CODE", None)
    var_list = mc.get("T_VAR", [])
    if not var_list:
        raise ValueError("LBS-JSON ohne T_VAR in T_MC!")
    var = var_list[0]

    # 3) Knoten aufbauen: alle Rollen aus T_VAR, die Kernobjekte sind
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    for r in var.get("T_ROLE", []):
        role_id = r.get("ROLE_ID", "")
        ntype, core_id = _core_type_and_id(role_id)
        if not ntype or not core_id:
            continue

        # Versuche Meta zuerst über Kern-ID, sonst über Original-ID
        meta = role_meta.get(core_id, role_meta.get(role_id, {}))
        level = meta.get("level", 0)
        code  = meta.get("code", 0)
        descr = meta.get("descr", "")

        # Option: reine Hilfsrollen (Level 0, Code 0) überspringen
        if level == 0 and code == 0:
            continue

        attrs: Dict[str, Any] = {"level": int(level)}

        if ntype == "MaLo":
            direction = _infer_malo_direction(core_id, descr)
            if direction:
                attrs["direction"] = direction

        elif ntype == "MeLo":
            fn = _infer_melo_function(core_id, descr)
            attrs["function"] = fn
            # optional "dynamic": 1 bei Hinterschaltung
            attrs["dynamic"] = 1 if fn == "H" else 0

        # TR / NeLo: zunächst keine Spezialattribute
        if core_id not in nodes_by_id:
            nodes_by_id[core_id] = make_node(core_id, ntype, attrs)

    nodes: List[Dict[str, Any]] = list(nodes_by_id.values())

    # 4) Kanten aufbauen: nur T_ROLEREL mit REL_TYPE
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
                val = (att.get("ATTR_VALUE") or "").strip()
                if val:
                    rel_type = val
                    break

        # ggf. heuristische Deduktion, falls REL_TYPE fehlt
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
                continue  # uninteressante Beziehung

        # Nur die relevanten Relationstypen aufnehmen
        if rel_type not in ("MEMA", "MENE", "MEME", "METR"):
            continue

        edges.append(make_edge(c1, c2, rel_type))

    # 5) Pattern und Graph-Attribute
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
            "is_template": True,
            "lbs_code": str(lbs_code) if lbs_code is not None else None,
        }
    }
