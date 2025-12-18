import os                                       # Dateipfade
import re                                       # Reguläre Ausdrücke
import json                                     # Für JSON-Formate
import time                                     # Testzwecke
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Any       # Typ-Hints für Struktur

import pandas as pd                             # Für das Excel-Zeug
import networkx as nx                           # Aus Beziehungen Graphkomponenten bestimmen

# Fehlermeldungen aus den Excel-Sachen ignorieren
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


# ------------------------------
# FORMATIEREN
# ------------------------------
def _word_unifier(s: str) -> str:
    """
    Entfernt unnütze Leerzeichen, wandelt in lower case um, ersetzt Leerraumfolgen
    durch Unterstriche, wandelt Umlaute in ASCII um.
    RegEx: Entfernt Symbole
    :param s: Originaler Spaltenname
    :return: Bereinigte Version (maschinenfreundlich)
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

def column_unifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Macht dasselbe wie in @_canon, nur für den gesamten DataFrame

    Beispiel:
    ["Marktlokation ID", "Ausspeisung / Einspeisung", "Messlokation"]
    ["marktlokation_id", "ausspeisungeinspeisung", "messlokation"]
    :param df: Einen Datensatz
    :return: Bereinigter Datensatz
    """
    df = df.copy()
    df.columns = [_word_unifier(c) for c in df.columns]
    return df


# ------------------------------
# ROHDATEN AUS EXCEL EINLESEN
# ------------------------------
def table_loader(base_directory: str) -> Dict[str, pd.DataFrame]:
    """
    Lädt und verarbeitet mehrere Excel-Dateien als DataFrames in einem Dictionary.
    Spalten in jedem geladenen DataFrame werden bereinigt.

    Beispiel:
    {
    "malo": <DataFrame>,
    "melo": <DataFrame>,
    "pod_rel": <DataFrame>,
    "meter": <DataFrame>,
    ... <optional>
    }

    :param base_directory: Verzeichnispfad
    :type base_directory: str
    :return: Dictionary, in dem die Schlüssel die Datensatznamen und die Werte die DataFrames sind
    :rtype: Dict[str, pd.DataFrame]
    """
    def csv_reader(name):
        enc = "cp1252" if "MALO" in name.upper() else "utf-8-sig"  # Die CSVs sind nicht einheitlich exportiert, wir wollen Umlaute und ß behalten
        path = os.path.join(base_directory, name)
        return column_unifier(pd.read_csv(path, sep=";", encoding=enc, dtype=str))

    tables = {}

    # Sets von René
    tables["malo"] = csv_reader("data/training_data/SDF_MALO.csv")
    tables["melo"] = csv_reader("data/training_data/SDF_MELO.csv")
    tables["pod_rel"] = csv_reader("data/training_data/SDF_POD_REL.csv")
    tables["meter"] = csv_reader("data/training_data/SDF_METER.csv")

    # Excel einlesen - 20s langsamer als CSV
    # def rd(name):
    #     path = os.path.join(base_dir, name)
    #     return normalize_columns(pd.read_excel(path))
    #
    # tables = {}
    #
    # tables["malo"] = rd("SDF_MALO.xlsx")
    # tables["melo"] = rd("SDF_MELO.xlsx")
    # tables["pod_rel"] = rd("SDF_POD_REL.xlsx")
    # tables["meter"] = rd("SDF_METER.xlsx")

    # Optional
    #for opt in ["SDF_MSB.xlsx", "Mapping_StdFormat.xlsx",
    #            "SDF_DIST_Netbill.xlsx", "SDF_SUPP_Liefbill.xlsx", "SDF_MOP.xlsx"]:
    #    p = os.path.join(base_dir, opt)
    #    if os.path.exists(p):
    #        try:
    #            tables[opt.split(".")[0].lower()] = normalize_columns(pd.read_excel(p))
    #        except Exception:
    #            pass

    return tables


# ------------------------------
# VEREINHEITLICHEN
# ------------------------------
def canonicalize(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Verarbeitet und standardisiert Tabellendaten, indem Spalten umbenannt oder neue Spalten
    hinzugefügt werden.
    Jeder Schlüssel im Eingabe dictionary entspricht einer bestimmten Kategorie („malo“,
    „melo“, ...) und wird validiert, umbenannt und in ein kanonisches Format konvertiert.
    Wenn erforderliche Spalten nicht existieren, löst die Funktion einen Fehler aus.
    Verarbeitet mehrere Kategorien, darunter „malo“, „melo“, „pod_rel“, „meter“ sowie optional
    die Kategorie „msb“.

    :param tables: Dictionary mit String-Schlüssel und DataFrame-Values, die standardisiert werden sollen.
    :type tables: Dict[str, pd.DataFrame]
    :return: Standardisiertes Input-Dictionary
    :rtype: Dict[str, pd.DataFrame]
    :raises ValueError: Wenn eine erwartete Spalte fehlt oder nicht gemappt werden kann.
    """
    t = {}

    # MaLo
    malo = tables["malo"].copy()

    # Mapping auf eindeutige ID und Lieferrichtung (Ein-/Ausspeisung)
    # Nach möglichen Spalten, in denen MaLo sein könnte, suchen
    malo_id_cols = [c for c in malo.columns if "marktlokation" in c and "id" in c] + \
                   [c for c in malo.columns if c in ("malo_tranche", "maloid", "malo_id")]
    if not malo_id_cols:
        raise ValueError("MALO: keine Spalte für MaLo-ID gefunden!")
    # Erste gefundene Spalte wird als MaLo-ID verwende
    malo.rename(columns={malo_id_cols[0]: "malo_id"}, inplace=True)

    dir_cols = [c for c in malo.columns if "ausspeisung" in c and "einspeisung" in c] + \
               [c for c in malo.columns if c in ("lieferrichtung", "richtung", "direction_code")]
    if dir_cols:
        malo.rename(columns={dir_cols[0]: "direction_code"}, inplace=True)
    else:
        malo["direction_code"] = None

    # Speichern im Ergebnis-Dictionary
    t["malo"] = malo

    # MeLo
    melo = tables["melo"].copy()

    # Nach möglichen Spalten, in denen MeLo sein könnte, suchen
    melo_id_cols = [c for c in melo.columns if c in ("messlokation", "melo", "melo_id")]
    if not melo_id_cols:
        raise ValueError("MELO: keine Spalte für MeLo-ID gefunden!")
    melo.rename(columns={melo_id_cols[0]: "melo_id"}, inplace=True)

    # Sucht nach möglichen Spannungsebene-Spalten
    # Falls keine: voltage_level mit None
    volt_cols = [c for c in melo.columns if "spannungsebene" in c or c == "voltage_level"]
    if volt_cols:
        melo.rename(columns={volt_cols[0]: "voltage_level"}, inplace=True)
    else:
        melo["voltage_level"] = None

    # Speichern im Ergebnis-Dictionary
    t["melo"] = melo

    # POD_REL (Relation MaLo <-> MeLo, Tabelle, die verknüpft)
    pr = tables["pod_rel"].copy()

    # MaLo
    pr_malo_cols = [c for c in pr.columns if c in ("malo_tranche", "malo_id", "malo")]
    # MeLo
    pr_melo_cols = [c for c in pr.columns if c in ("melo", "messlokation", "melo_id")]
    if not pr_malo_cols or not pr_melo_cols:
        raise ValueError("POD_REL: Spalten für MaLo/MeLo nicht gefunden!")
    pr.rename(columns={pr_malo_cols[0]: "malo_id", pr_melo_cols[0]: "melo_id"}, inplace=True)
    t["pod_rel"] = pr

    # METER (TR per MeLo)
    me = tables["meter"].copy()

    # Kandidaten MeLo-ID in der ZÄHLERTABELLE
    # Kandidaten für Zähler-ID
    # Pro MeLo: Welche TRs (Zähler, Messgeräte) hängen an MeLo dran?
    me_melo_cols = [c for c in me.columns if c in ("meldepunktbez_gers", "melo", "melo_id", "messlokation")]
    me_ser_cols  = [c for c in me.columns if c in ("serialnummer", "serial", "zaehlernummer", "zaehler_id")]
    if not me_melo_cols or not me_ser_cols:
        raise ValueError("METER: Spalten für MeLo/Serialnummer nicht gefunden!")

    # Für jede MeLo einen TR-Knoten pro Zähler mit METR-Kante
    me.rename(columns={me_melo_cols[0]: "melo_id", me_ser_cols[0]: "tr_id"}, inplace=True)
    t["meter"] = me

    # MSB optional
    # MeLo-Spalten und iln (internationale Lokationsnummer des MSB) finden
    # Welche MSB welche MeLo betreut

    # if "sdf_msb" in tables:
    #    msb = tables["sdf_msb"].copy()
    #    msb_melo_cols = [c for c in msb.columns if c in ("messlokation", "melo_id", "melo")]
    #    msb_iln_cols  = [c for c in msb.columns if "iln" in c]
    #    if msb_melo_cols:
    #        msb.rename(columns={msb_melo_cols[0]: "melo_id"}, inplace=True)
    #    if msb_iln_cols:
    #        msb.rename(columns={msb_iln_cols[0]: "msb_iln"}, inplace=True)
    #    t["msb"] = msb

    return t


# ------------------------------
# FACHLICHE CODES → ROLLEN
# ------------------------------
def map_direction(direction_code: Any) -> str:
    """
    Mapping von Einspeisung und Ausspeisung auf consumption und generation.

    Beispiel:
    {"id": <MaLo-ID>, "type": "MaLo", "attrs": {"direction": "consumption"}}

    """
    if direction_code is None:
        return None
    s = str(direction_code).strip().upper()             # Standardisieren
    if s in ("Z07", "BEZUG", "VERBRAUCH", "V"):
        return "consumption"
    if s in ("Z06", "EINSPEISUNG", "ERZEUGUNG", "E"):
        return "generation"
    return None


# ------------------------------
# BIPARTITER GRAPH & KOMPONENTEN
# ------------------------------
def component_builder(dataframe_dict: Dict[str, pd.DataFrame]) -> List[Tuple[set, set]]:
    """
    Baut bipartite Graphen aus der pod_rel-Tabelle (MaLo–MeLo-Beziehungen) und zerlegt ihn dann in seine
    zusammenhängenden Komponenten.
    Die Knotenmenge ist in zwei disjunkte Klassen MaLo und MeLo aufgeteilt, und Kanten verlaufen nur zwischen diesen Klassen.
    Aus dem Graphen werden zusammenhängende Komponenten extrahiert und in Mengen von MaLo- und MeLo-Identifikatoren
    aufgeteilt.

    :param dataframe_dict:  Ein Dictionary, das DataFrames enthält, wobei einer der Schlüssel "pod_rel" ist. Dieser
            DataFrame muss die Spalten "malo_id" und "melo_id" enthalten, die Beziehungen zwischen
            MaLo- und MeLo-Entitäten darstellen.
    :type dataframe_dict: Dict[str, pd.DataFrame]

    :return: Eine Liste von Tupeln, wobei jedes Tupel aus zwei Mengen besteht. Die erste Menge enthält
            MaLo-Identifikatoren und die zweite Menge MeLo-Identifikatoren, die zur gleichen
            zusammenhängenden Komponente gehören.
    :rtype: List[Tuple[set, set]]
    """

    # Relationstabelle ohne Zeilen, in denen eine der beiden IDs fehlt
    # Spalten MaLo und MeLo
    pr = dataframe_dict["pod_rel"][["malo_id", "melo_id"]].dropna().astype(str)

    # Ungerichteter Graph, wir prüfen erst auf allgemeine Verbundenheit
    G = nx.Graph()
    for _, row in pr.iterrows():
        malo = f"MaLo::{row.malo_id}"
        melo = f"MeLo::{row.melo_id}"
        G.add_node(malo, kind="MaLo", id=row.malo_id)
        G.add_node(melo, kind="MeLo", id=row.melo_id)
        G.add_edge(malo, melo)

    components = []

    # Zusammenhängende Teilgraphen - Knotennamen
    # Jede Komponente entspricht alle MaLo/MeLo, die über Kanten direkt oder indirekt verbunden sind
    # Bsp. Ist-Bündel:
    # [
    #     ({"MaLoA"}, {"MeLo1"}),           [1:1]
    #     ({"MaLoB"}, {"MeLo2", "MeLo3"}),  [1:2]
    #     ({"MaLoC", "MaLoD"}, {"MeLo4"}),  [2:1]
    # ]
    for comp_nodes in nx.connected_components(G):
        malos = {G.nodes[n]["id"] for n in comp_nodes if G.nodes[n]["kind"] == "MaLo"}
        melos = {G.nodes[n]["id"] for n in comp_nodes if G.nodes[n]["kind"] == "MeLo"}
        if malos or melos:
            components.append((malos, melos))   # Komponenten-Tupel
    return components

# Gibt an wie viele MaLo oder MeLo in einer Komponente sind
def classify_pattern(malo_count: int, melo_count: int) -> str:
    if malo_count == 1 and melo_count == 1:
        return "1:1"
    if malo_count == 1 and melo_count == 2:
        return "2:1"
    if malo_count == 2 and melo_count == 1:
        return "1:2"
    if malo_count == 2 and melo_count == 2:
        return "2:2"
    return f"{melo_count}:{malo_count}"


# ------------------------------
# GRAPH AUFBAUEN
# ------------------------------
def make_node(node_id: str, ntype: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baut einheitlich Knotenobjekte (Dictionary).
    Bsp.:
    {
    "id":   node_id,
    "type": ntype,
    "attrs": attrs
    }

    :param node_id: Fachliche ID (MeLo, MaLo oder TR)
    :param ntype: Knotentyp als String
    :param attrs: Attribut-Dictionary
    :return: Dictionary
    """
    return {"id": node_id, "type": ntype, "attrs": attrs}

def make_edge(src: str, dst: str, rel: str) -> Dict[str, str]:
    """
    Baut einheitlich Kantenobjekte (Dictionary).

    :param src: Quellknoten-ID
    :param dst: Zielknoten-ID
    :param rel: Beziehung (MEMA, METR, ...)
    :return: Dictionary
    """
    return {"src": src, "dst": dst, "rel": rel}

def build_graphs(t: Dict[str, pd.DataFrame],
                 restrict_to: Tuple[int, int] = None) -> List[Dict[str, Any]]:
    """
    Baut Graphen für jede verbundene Komponente aus MaLo/MeLo im Stil der LBS.
    Jede Komponente ist ein Ist-Bündel.

    :param t: Dictionary mit Input-Daten für Graph-Konstruktion:
              - 'malo': MaLo DataFrame
              - 'melo': MeLo DataFrame
              - 'pod_rel': MeLo - MaLo-Beziehung DataFrame
              - 'meter': Meter DataFrame
    :param restrict_to: Zwei Integer mit maximale Anzahl für MaLo bzw. MeLo
    :return: Liste aus Dictionaries, jeweils einen Graph repräsentierend (Metadata, Knoten, Kanten, Attribute)
    :rtype: List[Dict[str, Any]]
    """
    malo_df = t["malo"].copy()
    melo_df = t["melo"].copy()
    pr_df   = t["pod_rel"][["malo_id", "melo_id"]].dropna().astype(str).copy() # Auf die wichitgsten beiden Spalten reduziert
    meter_df = t["meter"].copy()

    # Setzt MaLo- und MeLo-ID als Index und erzeugt Dictionary
    # Bsp.:
    # {
    #     "MaLo1": {"direction_code": "Z07", ...},
    #     "MaLo2": {"direction_code": "Z06", ...},
    #     ...
    # }
    malo_info = malo_df.set_index("malo_id").to_dict(orient="index")
    melo_info = melo_df.set_index("melo_id").to_dict(orient="index")

    # TR je MeLo sammeln
    # Bsp.:
    # {
    #     "MeLoA": ["Zähler1", "Zähler2"],
    #     "MeLoB": ["Zähler3"],
    #     ...
    # }
    meter_by_melo = defaultdict(list)                                   # Für jeden neuen Key eine leere List
    if "melo_id" in meter_df.columns and "tr_id" in meter_df.columns:   # Nur, wenn beide Spalten existieren
        for _, r in meter_df[["melo_id", "tr_id"]].dropna().astype(str).iterrows(): # Iteration über diese Spalten
            meter_by_melo[r.melo_id].append(r.tr_id)                    # Für jede MeLo hängt eine (oder mehrere) TR-IDs dran

    # Bsp.:
    # [
    #   ({"MaLo1"}, {"MeLoA"}),              # 1:1
    #   ({"MaLo2"}, {"MeLoB", "MeLoC"}),     # 1:2
    #   ({"MaLo3", "MaLo4"}, {"MeLoD"}),     # 2:1
    #   ...
    # ]
    comps = component_builder(t)
    graphs = []

    for malos, melos in comps:
        m_count, e_count = len(malos), len(melos)

        # Wenn Komponente zu groß, dann überspringen
        if restrict_to and (m_count > restrict_to[0] or e_count > restrict_to[1]):
            continue

        nodes = []
        edges = []

        # MaLo-Knoten hinzufügen
        # Bsp.:
        # {
        #     "id": mid,
        #     "type": "MaLo",
        #     "attrs": {"direction": "consumption" / "generation" / None}
        # }
        for mid in sorted(malos):
            dinfo = malo_info.get(mid, {})
            direction = map_direction(dinfo.get("direction_code"))
            nodes.append(make_node(mid, "MaLo", {"direction": direction}))

        # MeLo-Knoten und TR pro Knoten hinzufügen
        # Bsp.:
        # {
        #     "id": eid,
        #     "type": "MeLo",
        #     "attrs": {"voltage_level": "NS" / "MS" / None}
        # }
        for eid in sorted(melos):
            einfo = melo_info.get(eid, {})
            nodes.append(make_node(eid, "MeLo", {"voltage_level": einfo.get("voltage_level")}))
            # Liste aller TR-IDs, die zu dieser MeLo gehören (ggf. leere Liste)
            for tr in meter_by_melo.get(eid, []):
                nodes.append(make_node(tr, "TR", {}))
                edges.append(make_edge(eid, tr, "METR"))

        # Kanten zwischen MeLo und MaLo setzen (nur Paare der aktuellen Komponente)
        comp_pairs = set((str(r.malo_id), str(r.melo_id)) for _, r in pr_df.iterrows()
                         if r.malo_id in malos and r.melo_id in melos)
        for (malo_id, melo_id) in comp_pairs:
            edges.append(make_edge(melo_id, malo_id, "MEMA"))

        # Graph-Metadaten
        pattern = classify_pattern(m_count, e_count)                    # Pattern-Funktion (Relation MaLo - MeLo) - Label für Bündelformat
        graph_id = f"comp:{sorted(list(malos))}|{sorted(list(melos))}"  # Eindeutige ID pro Graph
        graphs.append({
            "graph_id": graph_id,
            "label": None,  # kein LBS-Code, reines Ist-Bündel
            "nodes": nodes,
            "edges": edges,
            "graph_attrs": {"pattern": pattern, "malo_count": m_count, "melo_count": e_count}
        })

    #Bsp.:
    # [
    #     {"graph_id": "...", "nodes": [...], "edges": [...], "graph_attrs": {...}},
    #     {...},
    #     ...
    # ]
    return graphs


# ------------------------------
# VALIDIERUNG
# ------------------------------
def relation_validation(t: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Validiert die Beziehungen zwischen MaLo und MeLo basierend auf dem gegebenen Mapping und den eingebundenen DataFrames.
    Checkt fehlende Referenzen und Multiplizität der Beziehungen.

    :param t: Ergebnis von canonicalize, also ein Dictionary mit:
        - "pod_rel" Beziehungen-DataFrame
        - "malo" MaLoID-DataFrame
        - "melo" MeLoID-DataFrame
    :type t: Dict[str, pd.DataFrame]
    :return: Dictionary mit entweder:
        - "missing_malo_refs": Liste von kaputten MaLo-ID-Referenzen
        - "missing_melo_refs": Liste von kaputten MeLo-ID-Referenzen
        - "counts": Zählstatistiken über Bündel:
    :rtype: Dict[str, Any]
    """
    pr = t["pod_rel"][["malo_id", "melo_id"]].dropna().astype(str)          # Relationstabelle ohne Zeilen, in denen eine der beiden IDs fehlt
    malo_ids = set(t["malo"]["malo_id"].astype(str))
    melo_ids = set(t["melo"]["melo_id"].astype(str))
    missing_malo = [(r.malo_id, r.melo_id) for _, r in pr.iterrows() if r.malo_id not in malo_ids]
    missing_melo = [(r.malo_id, r.melo_id) for _, r in pr.iterrows() if r.melo_id not in melo_ids]

    # Multiplizitäten (MaLo - MeLo)
    m_per_malo = pr.groupby("malo_id")["melo_id"].nunique()
    malo_2_1 = int((m_per_malo == 2).sum())
    malo_1_1 = int((m_per_malo == 1).sum())
    malo_gt2 = int((m_per_malo > 2).sum())

    m_per_melo = pr.groupby("melo_id")["malo_id"].nunique()
    melo_1_2 = int((m_per_melo == 2).sum())
    melo_1_1 = int((m_per_melo == 1).sum())
    melo_gt2 = int((m_per_melo > 2).sum())

    return {
        "missing_malo_refs": missing_malo,
        "missing_melo_refs": missing_melo,
        "counts": {
            "malo->melo": {
                "1": malo_1_1,
                "2": malo_2_1,
                ">2": malo_gt2
            },
            "melo->malo": {
                "1": melo_1_1,
                "2": melo_1_2,
                ">2": melo_gt2
            }
        }
    }


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    start_time = time.time()
    BASE = os.path.dirname(os.path.abspath(__file__))
    tables = table_loader(BASE)
    tables = canonicalize(tables)
    report = relation_validation(tables)

    # Fokus zunächst auf kleine Komponenten (<=2 MaLo, <=2 MeLo)
    graphs = build_graphs(tables, restrict_to=(2, 2))
    #graphs = build_graphs(tables, restrict_to=(4, 3))

    print("Ist-Graphen Check:", json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Gebaut: {len(graphs)} Graphen (Pattern-Counts):")
    from collections import Counter
    print(Counter(g["graph_attrs"]["pattern"] for g in graphs))
    print("Prozess ausgeführt in %s Sekunden" % (time.time() - start_time))
    # Optional: JSONL export
    with open(os.path.join(BASE, "data", "ist_graphs.jsonl"), "w", encoding="utf-8") as f:
         for g in graphs:
             f.write(json.dumps(g, ensure_ascii=False) + "\n")
