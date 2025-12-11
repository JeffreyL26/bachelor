# Imports für Dateien finden, einlesen und der ganze Kram
import os
import glob
import json
from typing import Dict, Any, List, Optional, Tuple

#TODO: Kommentare und Dokumentation revamp - Ziel 15.12

# Wir nutzen die Helfer aus graph_converter, damit Format identisch ist
from graph_converter import make_node, make_edge, classify_pattern


# ------------------------------
# SEMANTIK EXTRAHIEREN
# ------------------------------

def _core_type_and_id(role_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Nimmt eine LBS-Rolle und übersetzt sie in Knotentyp und einheitliche Knoten-ID.
    :param role_id: String der ID, "Hüllen" wie ANA_BZG oder G_BZG, die aber domänenspezifisch MaLo, MeLo, usw. sind
    :return:
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
    # z.B. Anlagerolle "Anlage MeLo ..." auf MeLo-ID mappen
    if rid.startswith("AME_"):
        return "MeLo", "ME_" + role_id.split("AME_", 1)[1]
    if rid.startswith("AMA_"):
        return "MaLo", "MA_" + role_id.split("AMA_", 1)[1]
    return None, None


def _descr_de(role_obj: Dict[str, Any]) -> str:
    """
    Der LBS-Rolleneintrag hat mehrsprachige Beschreibungen. Die Funktion holt die deutsche.
    :param role_obj: Der LBS-Rolleneintrag
    :return: deutsche Beschreibung, falls vorhanden, andernfalls leerer String
    """
    for d in role_obj.get("T_DESCR", []):
        if d.get("LANG") in ("D", "DE"):
            return d.get("DESCR", "")
    return ""


def _infer_malo_direction(role_id: str, descr_de: str = "") -> Optional[str]:
    """
    Erkennt, ob MaLo Erzeuger oder Verbraucher ist. Landet dann in den Knoten-Features in @lbsjson_to_template_graph.
    Zwei Graphen können die gleiche Anzahl MaLo haben, aber andere Funktionstypen.
    :param role_id: MaLo-ID
    :param descr_de: Deutsche Beschreibung der Rolle
    :return: Klare Kategorie ("generation" oder "consumption"), falls erkannt
    """
    rid = (role_id or "").upper()
    txt = (descr_de or "").upper()
    if "EIN" in rid or "EINSP" in rid or "EINSPEIS" in txt or "ERZEUG" in txt:
        return "generation"
    if "BZG" in rid or "BEZUG" in txt or "AUSSPEIS" in txt:
        return "consumption"
    return None


def _infer_melo_function(role_id: str, descr_de: str = "") -> str:
    """
    Erkennt MeLo-Rolle (Hinterschaltung, Differenzmessung, Speicher, Nichts spezielles). Landet dann in den Knoten-Features
    (attrs[function] in @lbsjson_to_template_graph).
    Zwei Graphen können die gleiche Anzahl MeLo haben, aber andere Funktionstypen.
    :param role_id: MeLo-ID
    :param descr_de: Deutsche Beschreibung der Rolle
    :return: Klare Funktion ("H", "D", "S" oder "N"), falls erkannt
    """
    rid = (role_id or "").upper()
    txt = (descr_de or "").lower()
    if "_HS_" in rid or "hinterschalt" in txt:
        return "H"
    if "differenz" in txt or "saldo" in txt or "_DIF" in rid:
        return "D"
    if "speicher" in txt or "_SPEI" in rid:
        return "S"
    return "N"


# ------------------------------
# GRAPH AUFBAUEN
# ------------------------------

def lbsjson_to_template_graph(lbs_json: Dict[str, Any],
                              graph_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Erstellt aus einem LBS-Export aus dem SAP-System (Soll-Konzept) einen "Goldstandard"-Graphen.
    :param lbs_json: LBS-Export als Python-Dict
    :param graph_id: ID des Graphen, falls vorhanden, sonst generiert
    :return: Template-Graph, der einem Soll-Konstrukt entspricht (LBS-Code und dazugehörige Variante)
    """

    # T_ROLE für den Eintrag in der LBS-Rollentabelle - Nachschlagewerk erstellen
    # Marker, ob Kernobjekt oder Zusatzrolle
    # Beispiel T_ROLE-Eintrag:
    # {
    #     "ID": "ME_xyz",                                       INFO: Wäre ein Kernobjekt. AME_xyz wäre eine Anlage (technische Rolle), wie diese MeLo in einer bestimmten Variante genutzt wird
    #     "LBS_OBJECT_LEVEL": 1,
    #     "LBS_OBJECT_CODE": 10,
    #     "T_DESCR": [...],
    #     "T_ATTR": [...]
    # }
    # INFO Teil 2: Semantisch beschreiben beide dieselbe fachliche Einheit. Damit keine zwei Knoten gebaut werden, gibt es dazu Level- und Code-Flags

    # Was weiß ich über diese Rolle aus der globalen Tabelle?
    role_meta: Dict[str, Dict[str, Any]] = {}
    for r in lbs_json.get("T_ROLE", []):
        rid = r.get("ID", "")                                   # Rollen-ID, wie z.B. ME_xyz
        descr = _descr_de(r)                                    # deutsche Beschreibung
        level = r.get("LBS_OBJECT_LEVEL", 0)                    # 0 ist das, was nicht als Knoten gebraucht wird
        code = r.get("LBS_OBJECT_CODE", 0)                      # wenn 0,0 reines Hilfsobjekt (wird später ignoriert)
        attrs = {a.get("ATTR_CATEGORY"): a.get("ATTR_VALUE")    # Attribute ale Map
                 for a in r.get("T_ATTR", [])}
        role_meta[rid] = {
            "id": rid,
            "descr": descr,
            "level": level,
            "code": code,
            "attrs": attrs,
        }


    # Messkonfiguration (MC)
    mc_list = lbs_json.get("T_MC", [])
    if not mc_list:
        raise ValueError("LBS-JSON ohne Messkonzeptkonfiguration (MC)!")
    mc = mc_list[0]
    lbs_code = mc.get("LBS_CODE")                               # Konfigurationstyp (jede MC sollte einen haben)
    var_list = mc.get("T_VAR", [])                              # Varianten dessen (z.B. unterschiedliche Ausprägungen)
    if not var_list:
        raise ValueError("LBS-JSON ohne T_VAR in Messkonzeptkonfiguration (MC)!")
    var = var_list[0]   # TODO: ggf. irgendwann nach PRIORITY wählen, derzeit einfach erste Variante


    # Knoten/Kernobjekte aufbauen
    nodes_by_id: Dict[str, Dict[str, Any]] = {}

    for r in var.get("T_ROLE", []):                             # Innerhalb einer Variante gibt es auch eine T_ROLE Liste, hier: rollenbezogen
        role_id = r.get("ROLE_ID", "")                          # z.B. AME_xyz, ME_xyz
        # Mapping LBS-Rollen auf Knotentyp und Knoten-ID
        ntype, core_id = _core_type_and_id(role_id)
        if not ntype or not core_id:
            continue

        # Metadaten aus Rollentabelle aus Kernrolle, alternativ Rollen-ID, ansonsten leer
        meta = role_meta.get(core_id, role_meta.get(role_id, {}))
        level = int(meta.get("level", 0))
        code = meta.get("code", 0)
        descr = meta.get("descr", "")

        # keine Hilfsrolle, kein Kernobjekt → Knoten brauchen wir nicht
        if level == 0 and code == 0:
            continue

        # Knoten Attribute
        attrs: Dict[str, Any] = {"level": level}                # Jeder Knoten bekommt Level
        if ntype == "MaLo":                                     # MaLo-Knoten aus vorheriger Funktion
            direction = _infer_malo_direction(core_id, descr)
            if direction:
                attrs["direction"] = direction
        elif ntype == "MeLo":                                   # MeLo-Knoten aus vorheriger Funktion
            fn = _infer_melo_function(core_id, descr)
            attrs["function"] = fn
            # Wenn Hinterschaltung, hängt hinter anderer MeLo/Schaltstelle. Verhalten dynamischer und nicht immer gleich.
            # TODO: im Messkonzept besonders flexible / geschaltete / nachgelagerte Verbrauchs-/Erzeugungs-Situationen
            attrs["dynamic"] = 1 if fn == "H" else 0

        # Pro Kernobjekt GENAU ein Knoten im Graph
        if core_id not in nodes_by_id:
            nodes_by_id[core_id] = make_node(core_id, ntype, attrs)

    # Liste aller Knoten im LBS-Konstrukt
    nodes: List[Dict[str, Any]] = list(nodes_by_id.values())

    # Kanten durch REL_TYPE
    edges: List[Dict[str, str]] = []
    for rr in var.get("T_ROLEREL", []):                         # Tabelle der Beziehungen zwischen Rollen
        # Rollen-IDs der beiden Knoten mit Beziehung
        t1, c1 = _core_type_and_id(rr.get("OBJECT_ID_1", ""))   # Typen wie MeLo und IDs
        t2, c2 = _core_type_and_id(rr.get("OBJECT_ID_2", ""))
        if not t1 or not t2 or c1 not in nodes_by_id or c2 not in nodes_by_id:
            continue

        rel_type = None
        for att in rr.get("T_OBJRAT", []):                      # T_OBJRAT ist die Attributliste zur Beziehung
            if att.get("ATTR_CATEGORY") == "REL_TYPE":
                v = (att.get("ATTR_VALUE") or "").strip()
                if v:
                    rel_type = v
                    break

        # Fallbacks, falls REL_TYPE fehlt
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

        # 4 Beziehungen, die relevant sind: Darauf reduzieren
        # NOTIZ: MEME sind Hinterzählungsketten
        if rel_type not in ("MEMA", "MENE", "MEME", "METR"):
            continue

        # Beispiel:
        # {
        #     "src": c1,                                        Quellknoten
        #     "dst": c2,                                        Zielknoten
        #     "rel": rel_type                                   Beziehung
        # }
        edges.append(make_edge(c1, c2, rel_type))

    # Self-Loops herausfiltern, es könnten nämlich c1 = c2-Beziehungen entstehen (keine echte Beziehung "MeLo hängt mit sich selbst zusammen" → trivial)
    edges = [
        e for e in edges
        if not (e["rel"] == "MEME" and e["src"] == e["dst"])
    ]

    # Pattern aufbauen (1:1, 1:2 , ...)
    malo_count = sum(1 for n in nodes if n["type"] == "MaLo")
    melo_count = sum(1 for n in nodes if n["type"] == "MeLo")
    pattern = classify_pattern(malo_count, melo_count)

    # Wenn kein explizites graph_id, dann einfach nach LBS-Code
    if graph_id is None:
        graph_id = f"catalog-{lbs_code}"

    # Rückgabe fertiger Graph
    return {
        "graph_id": graph_id,
        "label": str(lbs_code) if lbs_code is not None else None,   # LBS-Code als String, für DGMC Klassen-ID
        "nodes": nodes,
        "edges": edges,
        "graph_attrs": {
            "pattern": pattern,
            "malo_count": malo_count,
            "melo_count": melo_count,
            "is_template": True,                                    # Soll-Zustand Flag
            "lbs_code": str(lbs_code) if lbs_code is not None else None,
        }
    }


# ------------------------------
# GRAPH-TEMPLATE FERTIGSTELEN
# ------------------------------

def build_all_templates(lbs_dir: str, out_path: str) -> None:
    """
    Liest alle JSON in lbs_dir ein, konvertiert sie zu Template-Graphen
    und schreibt sie als JSONL (ein Graph pro Zeile).
    :param lbs_dir: Directory der Templates
    :param out_path: Ausgabepfad
    """
    print("LBS-Verzeichnis:", lbs_dir)

    # Alle Dateien suchen, die auf json enden
    pattern = os.path.join(lbs_dir, "*.json")
    files = glob.glob(pattern)
    print("Glob-Pattern:", pattern)
    print("Gefundene Dateien:", len(files))
    for p in files:
        print("  -", os.path.basename(p))

    # Liste, in die alle Template-Graphen kommen
    graphs: List[Dict[str, Any]] = []

    for path in files:
        with open(path, "rb") as f:
            raw = f.read()
        # BDEW-Exports sind cp1252-kodiert, deshalb rb (read binary) → Bytes in String
        lbs_json = json.loads(raw.decode("cp1252"))
        # Graph bauen und ins Dict
        g = lbsjson_to_template_graph(lbs_json)
        graphs.append(g)

    print("Gebaut:", len(graphs), "Template-Graphen (Soll)")

    with open(out_path, "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print("JSONL geschrieben nach:", out_path)


if __name__ == "__main__":
    # Pfade
    BASE = os.path.dirname(os.path.abspath(__file__))
    lbs_dir = os.path.join(BASE, "data", "lbs_templates")
    out_path = os.path.join(BASE, "data", "lbs_soll_graphs.jsonl")

    build_all_templates(lbs_dir, out_path)
