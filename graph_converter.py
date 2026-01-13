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

#TODO: Brauchen wir classify_pattern?

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

# ------------------------------
# ROHDATEN AUS EXCEL/CSV EINLESEN (NEUER DATENSATZ)
# ------------------------------
def table_loader_new(base_directory: str) -> Dict[str, pd.DataFrame]:
    """Lädt den neuen Bündel/MCID-Datensatz (CSV) aus data/training_data/.

    Erwartete Kern-Dateien (required):
      - MALO.csv
      - MELO.csv
      - PODREL.csv
      - METER.csv

    Optionale Dateien (werden geladen, falls vorhanden):
      - TR.csv
      - BNDL2MC.csv
      - NELO.csv

    Die Spaltennamen werden wie bei den SDF-Tabellen via column_unifier() bereinigt.
    """
    def csv_reader(rel_path: str) -> pd.DataFrame:
        # CSV-Exports sind nicht immer einheitlich kodiert; wir versuchen robust zu lesen.
        enc_candidates = ["cp1252", "utf-8-sig", "utf-8"]
        path = os.path.join(base_directory, rel_path)
        last_err = None
        for enc in enc_candidates:
            try:
                return column_unifier(pd.read_csv(path, sep=";", encoding=enc, dtype=str))
            except Exception as e:
                last_err = e
        raise last_err  # pragma: no cover

    base = "data/training_data"
    required = {
        "malo": f"{base}/MALO.csv",
        "melo": f"{base}/MELO.csv",
        "pod_rel": f"{base}/PODREL.csv",
        "meter": f"{base}/METER.csv",
    }

    tables: Dict[str, pd.DataFrame] = {}
    for key, rel in required.items():
        path = os.path.join(base_directory, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required CSV for new dataset: {rel}")
        tables[key] = csv_reader(rel)

    optional = {
        "tr": f"{base}/TR.csv",
        "bndl2mc": f"{base}/BNDL2MC.csv",
        "nelo": f"{base}/NELO.csv",
    }
    for key, rel in optional.items():
        path = os.path.join(base_directory, rel)
        if os.path.exists(path):
            tables[key] = csv_reader(rel)

    return tables


def table_loader_all(base_directory: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Lädt **beide** Datensätze (SDF + neuer Bündel/MCID-Datensatz)."""
    return {
        "sdf": table_loader(base_directory),
        "new": table_loader_new(base_directory),
    }

def canonicalize(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Standardisiert Tabellendaten in ein kanonisches Format.

    Pflicht-Keys in `tables`:
      - malo, melo, pod_rel, meter

    Optionale Keys (wenn vorhanden werden sie ebenfalls kanonisiert):
      - tr, bndl2mc, nelo

    Ziel: Beide Datensätze (SDF und neuer Bündel/MCID-Datensatz) sollen möglichst dieselbe
    kanonische Struktur nutzen, damit dieselben Downstream-Methoden (component_builder,
    build_graphs, Baseline/Pipeline) darauf arbeiten können.
    """

    def _clean_id_series(s: pd.Series) -> pd.Series:
        # robuste Normalisierung: str, strip, entferne "nan"/leere
        s = s.astype(str).str.strip()
        s = s.replace({"nan": "", "None": ""})
        return s

    t: Dict[str, pd.DataFrame] = {}

    # ------------------------------
    # MaLo
    # ------------------------------
    malo = tables["malo"].copy()
    malo_id_cols = [c for c in malo.columns if "marktlokation" in c and "id" in c] +                    [c for c in malo.columns if c in ("malo_tranche", "maloid", "malo_id", "id_der_marktlokation__tranche")]
    if not malo_id_cols:
        raise ValueError("MALO: keine Spalte für MaLo-ID gefunden!")
    malo.rename(columns={malo_id_cols[0]: "malo_id"}, inplace=True)

    dir_cols = [c for c in malo.columns if "ausspeisung" in c and "einspeisung" in c] +                [c for c in malo.columns if c in ("lieferrichtung", "richtung", "direction_code", "ausspeisung__einspeisung_lieferrichtung")]
    if dir_cols:
        malo.rename(columns={dir_cols[0]: "direction_code"}, inplace=True)
    else:
        malo["direction_code"] = None

    malo["malo_id"] = _clean_id_series(malo["malo_id"])
    malo = malo[malo["malo_id"] != ""]
    malo = malo.drop_duplicates(subset=["malo_id"], keep="last")
    t["malo"] = malo

    # ------------------------------
    # MeLo
    # ------------------------------
    melo = tables["melo"].copy()
    melo_id_cols = [c for c in melo.columns if c in ("messlokation", "melo", "melo_id")]
    if not melo_id_cols:
        raise ValueError("MELO: keine Spalte für MeLo-ID gefunden!")
    melo.rename(columns={melo_id_cols[0]: "melo_id"}, inplace=True)

    melo["melo_id"] = _clean_id_series(melo["melo_id"])
    melo = melo[melo["melo_id"] != ""]
    melo = melo.drop_duplicates(subset=["melo_id"], keep="last")
    t["melo"] = melo

    # ------------------------------
    # POD_REL (MaLo <-> MeLo)
    # ------------------------------
    pr = tables["pod_rel"].copy()
    pr_malo_cols = [c for c in pr.columns if c in ("malo_tranche", "malo_id", "malo", "malo_tranche")] +                    [c for c in pr.columns if "malo" in c and "tranche" in c]
    pr_melo_cols = [c for c in pr.columns if c in ("melo", "messlokation", "melo_id")]
    if not pr_malo_cols or not pr_melo_cols:
        raise ValueError("POD_REL: Spalten für MaLo/MeLo nicht gefunden!")
    pr.rename(columns={pr_malo_cols[0]: "malo_id", pr_melo_cols[0]: "melo_id"}, inplace=True)

    # Optional: NeLo-Spalte (Netzlokation) – in manchen PODREL-Exports vorhanden (z.B. Spalte "NELO")
    pr_nelo_cols = [c for c in pr.columns if c in ("nelo", "nelo_id") or ("nelo" in c)]
    if pr_nelo_cols:
        pr.rename(columns={pr_nelo_cols[0]: "nelo_id"}, inplace=True)
        pr["nelo_id"] = _clean_id_series(pr["nelo_id"])

    pr["malo_id"] = _clean_id_series(pr["malo_id"])
    pr["melo_id"] = _clean_id_series(pr["melo_id"])
    # POD_REL kann zwei Arten von Beziehungen enthalten:
    #   (1) MaLo <-> MeLo  (für Bündel/Komponenten)
    #   (2) MeLo <-> NeLo  (Spalte "NELO" kann befüllt sein, MaLo ggf. leer)
    # Für (1) brauchen wir MaLo+MeLo, für (2) mindestens MeLo+NeLo.
    if "nelo_id" in pr.columns:
        pr = pr[(pr["melo_id"] != "") & ((pr["malo_id"] != "") | (pr["nelo_id"] != ""))]
    else:
        pr = pr[(pr["malo_id"] != "") & (pr["melo_id"] != "")]
    t["pod_rel"] = pr

    # ------------------------------
    # METER (Zählerdaten / Geräte je MeLo; **keine** Technische Ressource)
    # ------------------------------
    # Hinweis:
    # - In unseren CSVs steht hier u.a. "Serialnummer", "Merkmal Zählertyp", "Einbaudatum" usw.
    # - Das sind fachlich Zähler-/Geräteinformationen und nicht gleichbedeutend mit einer
    #   "Technischen Ressource (TR)" im LBS-Sinne.
    me = tables["meter"].copy()
    me_melo_cols = [c for c in me.columns if c in ("meldepunktbez_gers", "melo", "melo_id", "messlokation")]
    me_ser_cols  = [c for c in me.columns if c in ("serialnummer", "serial", "zaehlernummer", "zaehler_id")]
    if not me_melo_cols or not me_ser_cols:
        raise ValueError("METER: Spalten für MeLo/Serialnummer nicht gefunden!")
    me.rename(columns={me_melo_cols[0]: "melo_id", me_ser_cols[0]: "meter_id"}, inplace=True)

    # Richtung/Art des Zählers (falls vorhanden): z.B. ERZ (Einrichtungszähler), ZRZ (Zweirichtungszähler)
    # In manchen Exports steht hier "Energierichtung".
    meter_dir_cols = [c for c in me.columns if ("energierichtung" in c) or ("energy" in c and "direction" in c)]
    if meter_dir_cols:
        me.rename(columns={meter_dir_cols[0]: "meter_direction_code"}, inplace=True)
    else:
        me["meter_direction_code"] = None

    me["melo_id"] = _clean_id_series(me["melo_id"])
    me["meter_id"] = _clean_id_series(me["meter_id"])
    me = me[(me["melo_id"] != "") & (me["meter_id"] != "")]
    t["meter"] = me

# ------------------------------
    # TR (optional) – für Checks/Analysen (nicht zwingend für Graphbau)
    # ------------------------------
    if "tr" in tables:
        tr = tables["tr"].copy()

        # Pflicht: TR-ID + referenzierte MaLo(Anlage)
        tr_id_cols = [c for c in tr.columns if c in ("externeid", "tr_id", "tr", "id")]
        malo_cols  = [c for c in tr.columns if c in ("malo_anlage", "malo", "malo_id", "marktlokation", "marktlokation_id")]

        if tr_id_cols:
            tr.rename(columns={tr_id_cols[0]: "tr_id"}, inplace=True)
        else:
            tr["tr_id"] = None

        if malo_cols:
            tr.rename(columns={malo_cols[0]: "malo_id"}, inplace=True)
        else:
            tr["malo_id"] = None

        # Optional: Art der technischen Ressource (wird später für TR-Richtung verwendet)
        # Erwartete Codes (laut Vorgabe):
        #   Z17 => consumption
        #   Z50 => generation
        #   Z56 => storage => both
        tr_type_cols = [c for c in tr.columns if c in ("art_der_technischen_ressource", "tr_type_code", "art_technische_ressource")]
        if tr_type_cols:
            tr.rename(columns={tr_type_cols[0]: "tr_type_code"}, inplace=True)
        else:
            tr["tr_type_code"] = None

        # Normalisieren
        if "tr_id" in tr.columns:
            tr["tr_id"] = _clean_id_series(tr["tr_id"])
        if "malo_id" in tr.columns:
            tr["malo_id"] = _clean_id_series(tr["malo_id"])
        if "tr_type_code" in tr.columns:
            tr["tr_type_code"] = _clean_id_series(tr["tr_type_code"])

        t["tr"] = tr

    # ------------------------------
    # BNDL2MC (optional) – bundle_id/mcid Metadaten
    # ------------------------------
    if "bndl2mc" in tables:
        bm = tables["bndl2mc"].copy()
        bundle_cols = [c for c in bm.columns if c in ("buendel", "bndel", "bundle", "bundle_id")]
        malo_cols   = [c for c in bm.columns if c in ("marktlokation", "malo", "malo_id")]
        melo_cols   = [c for c in bm.columns if c in ("messlokation", "melo", "melo_id")]
        mcid_cols   = [c for c in bm.columns if c in ("mcid",)]
        if bundle_cols:
            bm.rename(columns={bundle_cols[0]: "bundle_id"}, inplace=True)
        else:
            bm["bundle_id"] = None
        if malo_cols:
            bm.rename(columns={malo_cols[0]: "malo_id"}, inplace=True)
        else:
            bm["malo_id"] = None
        if melo_cols:
            bm.rename(columns={melo_cols[0]: "melo_id"}, inplace=True)
        else:
            bm["melo_id"] = None
        if mcid_cols:
            bm.rename(columns={mcid_cols[0]: "mcid"}, inplace=True)
        else:
            bm["mcid"] = None

        for col in ("bundle_id", "malo_id", "melo_id", "mcid"):
            if col in bm.columns:
                bm[col] = _clean_id_series(bm[col])
        bm = bm[(bm["bundle_id"] != "") & ((bm["malo_id"] != "") | (bm["melo_id"] != ""))]
        t["bndl2mc"] = bm

    # ------------------------------
    # NELO (optional) – derzeit nur IDs; Relations fehlen noch
    # ------------------------------
    if "nelo" in tables:
        ne = tables["nelo"].copy()
        nelo_id_cols = [c for c in ne.columns if c in ("id", "nelo", "nelo_id", "ne_id")]
        if nelo_id_cols:
            ne.rename(columns={nelo_id_cols[0]: "nelo_id"}, inplace=True)
        else:
            ne["nelo_id"] = None
        if "nelo_id" in ne.columns:
            ne["nelo_id"] = _clean_id_series(ne["nelo_id"])
            ne = ne[ne["nelo_id"] != ""]
        t["nelo"] = ne

    return t
def map_direction(direction_code: Any) -> str:
    """
    Mapping von Einspeisung und Ausspeisung auf consumption und generation.

    Beispiel:
    {"id": <MaLo-ID>, "type": "MaLo", "attrs": {"direction": "consumption"}}

    """
    if direction_code is None:
        return None

    # Standardisieren
    s = str(direction_code).strip().upper()
    if not s or s == "NAN":
        return None

    # SDF / Standard-Codes
    if s in ("Z07", "BEZUG", "VERBRAUCH", "V"):
        return "consumption"
    if s in ("Z06", "EINSPEISUNG", "ERZEUGUNG", "E"):
        return "generation"

    # Neuer Datensatz: booleanartige Markierung ("X" in 'Ausspeisung')
    # Laut Datenlieferung bedeutet "X" hier: generation.
    if s in ("X", "JA", "TRUE", "1"):
        return "generation"

    return None


def map_tr_direction(tr_type_code: Any) -> str:
    """Mapping der TR-Art (Art der technischen Ressource) auf Richtung.

    Erwartete Codes (TR.csv):
      - Z17 => consumption
      - Z50 => generation
      - Z56 => storage => both

    Rückgabe:
      - "consumption" | "generation" | "both" | None
    """
    if tr_type_code is None:
        return None

    s = str(tr_type_code).strip().upper()
    if not s or s == "NAN":
        return None

    if s == "Z17":
        return "consumption"
    if s == "Z50":
        return "generation"
    if s == "Z56":
        return "both"

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

    # Durch NeLo-Relationen können in POD_REL Zeilen ohne MaLo vorkommen.
    # Für die Komponentenbildung (Bündel) berücksichtigen wir **nur** echte MaLo–MeLo-Kanten.
    pr["malo_id"] = pr["malo_id"].astype(str).str.strip()
    pr["melo_id"] = pr["melo_id"].astype(str).str.strip()
    pr = pr[(pr["malo_id"] != "") & (pr["melo_id"] != "") & (pr["malo_id"].str.lower() != "nan") & (pr["melo_id"].str.lower() != "nan")]

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
        return "1:2"
    if malo_count == 2 and melo_count == 1:
        return "2:1"
    if malo_count == 2 and melo_count == 2:
        return "2:2"
    return f"{malo_count}:{melo_count}"


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
                 restrict_to: Tuple[int, int] = None,
                 dataset_tag: str = None) -> List[Dict[str, Any]]:
    """Baut Ist-Graphen aus den kanonisierten Tabellen.

    Knoten:
      - MaLo (attrs: direction)
      - MeLo (attrs: direction)
      - TR   (attrs: direction optional; **nur** aus TR.csv, falls vorhanden)

    Kanten:
      - MEMA: MeLo -> MaLo
      - METR: MeLo -> TR  (TR aus TR.csv, über MaLo(Anlage) -> POD_REL -> MeLo)

    Hinweis:
      - Die Tabellen METER/SDF_METER enthalten Zähler-/Geräteinformationen (z.B. Serialnummer).
        Diese werden hier **nicht** als TR modelliert, aber sie werden (falls vorhanden) genutzt,
        um die MeLo-Richtung (z.B. ERZ/ZRZ) plausibel abzuleiten.
      - bundle_id wird nur als graph_id genutzt, wenn BNDL2MC für die Komponente eindeutig ist.
    """
    malo_df = t["malo"].copy()
    pr_df   = t["pod_rel"][["malo_id", "melo_id"]].dropna().astype(str).copy()

    # Durch NeLo-Relationen können in POD_REL Zeilen ohne MaLo vorkommen.
    # Für MaLo–MeLo-Adjazenz und MEMA-Kanten nutzen wir nur Zeilen mit beiden IDs.
    pr_df["malo_id"] = pr_df["malo_id"].astype(str).str.strip()
    pr_df["melo_id"] = pr_df["melo_id"].astype(str).str.strip()
    pr_df = pr_df[(pr_df["malo_id"] != "") & (pr_df["melo_id"] != "") & (pr_df["malo_id"].str.lower() != "nan") & (pr_df["melo_id"].str.lower() != "nan")]
    tr_df = t.get("tr")  # optional

    bndl_df = t.get("bndl2mc")  # optional

    # Index-Infos
    malo_info = malo_df.set_index("malo_id").to_dict(orient="index")

    # Adjazenz: MeLo -> {MaLo}
    malos_by_melo = defaultdict(set)
    # Adjazenz: MaLo -> {MeLo}
    melos_by_malo = defaultdict(set)
    for _, r in pr_df.iterrows():
        malos_by_melo[str(r.melo_id)].add(str(r.malo_id))
        melos_by_malo[str(r.malo_id)].add(str(r.melo_id))


    # Adjazenz: MeLo -> {NeLo} (nur falls POD_REL eine NeLo-Spalte hat)
    nelos_by_melo = defaultdict(set)
    if "nelo_id" in t["pod_rel"].columns:
        pr_nelo_df = t["pod_rel"][["melo_id", "nelo_id"]].dropna().astype(str).copy()
        pr_nelo_df["melo_id"] = pr_nelo_df["melo_id"].astype(str).str.strip()
        pr_nelo_df["nelo_id"] = pr_nelo_df["nelo_id"].astype(str).str.strip()
        pr_nelo_df = pr_nelo_df[(pr_nelo_df["melo_id"] != "") & (pr_nelo_df["nelo_id"] != "") & (pr_nelo_df["nelo_id"].str.lower() != "nan")]

        for melo_id, grp in pr_nelo_df.groupby("melo_id")["nelo_id"]:
            nelos_by_melo[melo_id] = set(grp.tolist())

    # METER: Zähler-Richtung pro MeLo (falls vorhanden)
    # - SDF: "meter_direction_code" ist typischerweise ERZ oder ZRZ
    # - NEW: Spalte kann leer sein, dann ignorieren
    meter_codes_by_melo: Dict[str, set] = defaultdict(set)
    meter_df = t.get("meter")
    if isinstance(meter_df, pd.DataFrame) and not meter_df.empty and "melo_id" in meter_df.columns:
        dir_col = "meter_direction_code" if "meter_direction_code" in meter_df.columns else None
        if dir_col is not None:
            tmp = meter_df[["melo_id", dir_col]].copy()
            tmp["melo_id"] = tmp["melo_id"].astype(str).str.strip()
            tmp[dir_col] = tmp[dir_col].astype(str).str.strip().str.upper()
            tmp = tmp[(tmp["melo_id"] != "") & (tmp[dir_col] != "") & (tmp[dir_col] != "NAN")]
            for melo_id, grp in tmp.groupby("melo_id")[dir_col]:
                meter_codes_by_melo[melo_id] = set(grp.tolist())

    # TR je MeLo sammeln (dedupliziert)    # TR aus TR.csv (optional): MaLo(Anlage) -> TR, und später via POD_REL auf MeLo projizieren.
    trs_by_malo = defaultdict(list)
    tr_rep_attrs_by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(tr_df, pd.DataFrame) and not tr_df.empty and {"tr_id", "malo_id"}.issubset(set(tr_df.columns)):
        tr_pairs = tr_df[["tr_id", "malo_id"]].dropna().astype(str).copy()
        tr_pairs["tr_id"] = tr_pairs["tr_id"].astype(str).str.strip()
        tr_pairs["malo_id"] = tr_pairs["malo_id"].astype(str).str.strip()
        tr_pairs = tr_pairs[(tr_pairs["tr_id"] != "") & (tr_pairs["malo_id"] != "")]

        # deterministisch: je MaLo die zugehörigen TR-IDs sortiert deduplizieren
        for malo_id, grp in tr_pairs.groupby("malo_id")["tr_id"]:
            uniq = sorted({x for x in grp.tolist() if x and x.lower() != "nan"})
            trs_by_malo[malo_id] = uniq

        # optionale Zusatz-Attribute (alles außer tr_id/malo_id) pro TR-ID (first occurrence)
        extra_cols = [c for c in tr_df.columns if c not in ("tr_id", "malo_id")]
        if extra_cols:
            rep = tr_df.dropna(subset=["tr_id"]).copy()
            rep["tr_id"] = rep["tr_id"].astype(str).str.strip()
            rep = rep[rep["tr_id"] != ""].drop_duplicates(subset=["tr_id"], keep="first")
            # to_dict speichert NaN als nan-String teils; wir lassen downstream entscheiden
            tr_rep_attrs_by_id = rep.set_index("tr_id")[extra_cols].to_dict(orient="index")

    comps = component_builder(t)
    graphs: List[Dict[str, Any]] = []

    for malos, melos in comps:
        m_count, e_count = len(malos), len(melos)

        # optionaler Größenfilter (Debug)
        if restrict_to:
            max_m, max_e = restrict_to
            if m_count > max_m or e_count > max_e:
                continue

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        comp_nelos: set = set()  # NeLo-IDs innerhalb dieser Komponente


        # Richtung pro MaLo (für Ableitung TR-direction)
        malo_dir = {}
        for mid in malos:
            dinfo = malo_info.get(mid, {})
            d = map_direction(dinfo.get("direction_code"))

            # Neuer Datensatz: in MALO.csv ist häufig nur "X" (Ausspeisung) gesetzt;
            # der Rest ist leer. Laut Vorgabe interpretieren wir "leer" als consumption.
            if d is None and (dataset_tag or "").lower() == "new":
                d = "consumption"

            malo_dir[mid] = d

        # --- MaLo-Knoten ---
        for mid in sorted(malos):
            nodes.append(make_node(mid, "MaLo", {"direction": malo_dir.get(mid)}))

        # --- MeLo-Knoten ---
        seen_tr = set()
        seen_metr = set()  # dedupliziert METR-Kanten (aus TR.csv)

        for eid in sorted(melos):
            # 1) Primär: Richtung über verbundene MaLo(s) ableiten
            dir_set = set()
            for mid in malos_by_melo.get(eid, set()):
                if mid in malos:
                    d = malo_dir.get(mid)
                    if d:
                        dir_set.add(d)

            if len(dir_set) == 1:
                melo_direction = next(iter(dir_set))
            elif len(dir_set) > 1:
                melo_direction = "both"
            else:
                melo_direction = None

            # 2) Optional: Zähler-Richtung aus METER (v.a. SDF: ERZ/ZRZ)
            #    - ZRZ (oder ZWR): bidirektional → both
            #    - ERZ: einrichtungszähler → Richtung bleibt wie aus MaLo abgeleitet
            m_codes = meter_codes_by_melo.get(eid, set())
            if m_codes:
                if any(c in ("ZRZ", "ZWR") for c in m_codes):
                    melo_direction = "both"

            nodes.append(make_node(eid, "MeLo", {"direction": melo_direction}))
#Zusätzliche TR aus TR.csv (via MaLo(Anlage) -> POD_REL -> MeLo)
        # Semantik: TRs aus TR.csv werden als TR-Knoten modelliert und per METR
        # an diejenigen MeLo gehängt, die zur referenzierten MaLo-Anlage verbunden sind.
        #
        # WICHTIG: TR-Richtung wird **nicht** mehr aus der MaLo-Richtung abgeleitet,
        # sondern aus der TR-Art ("Art der technischen Ressource"):
        #   Z17 => consumption, Z50 => generation, Z56 => storage => both
        # Fallback (nur wenn TR-Art fehlt/unbekannt): MaLo-basierte Richtung.
        if trs_by_malo:
            for malo_id in sorted(malos):
                malo_based_dir = malo_dir.get(malo_id)
                for raw_tr_id in trs_by_malo.get(malo_id, []):
                    tr_id = raw_tr_id

                    # TR-Art aus TR.csv (falls vorhanden) → Richtung
                    rep_attrs = tr_rep_attrs_by_id.get(raw_tr_id, {})
                    tr_dir = map_tr_direction(rep_attrs.get("tr_type_code"))
                    if tr_dir is None:
                        tr_dir = malo_based_dir

                    if tr_id not in seen_tr:
                        seen_tr.add(tr_id)
                        # Wir setzen beides: direction (kanonisch) und tr_direction (Alias),
                        # damit Ist- und Template-Graphen einheitlich bleiben.
                        attrs = {
                            "direction": tr_dir,
                            "source": "tr_table",
                            "malo_ref": malo_id,
                        }
                        # Zusatzattribute aus TR.csv (falls vorhanden) übernehmen
                        for k, v in rep_attrs.items():
                            if k in ("tr_id", "malo_id"):
                                continue
                            attrs[k] = v
                        nodes.append(make_node(tr_id, "TR", attrs))

                    # über POD_REL alle MeLo finden, die mit dieser MaLo verbunden sind (innerhalb der Komponente)
                    for melo_id in sorted(melos_by_malo.get(malo_id, set())):
                        if melo_id in melos:
                            key = (melo_id, tr_id)
                            if key not in seen_metr:
                                seen_metr.add(key)
                                edges.append(make_edge(melo_id, tr_id, "METR"))


# --- NeLo-Knoten + MENE-Kanten (MeLo -> NeLo) ---
        # Zuordnung NeLo -> MeLo erfolgt über die Spalte "NELO" in PODREL (falls vorhanden).
        # NeLo besitzt im Ist-Datensatz keine weiteren Attribute (nur ID).
        seen_mene = set()
        if nelos_by_melo:
            for melo_id in sorted(melos):
                for nelo_id in sorted(nelos_by_melo.get(melo_id, set())):
                    if not nelo_id:
                        continue
                    comp_nelos.add(nelo_id)
                    key = (melo_id, nelo_id)
                    if key not in seen_mene:
                        seen_mene.add(key)
                        edges.append(make_edge(melo_id, nelo_id, "MENE"))

        for nelo_id in sorted(comp_nelos):
            nodes.append(make_node(nelo_id, "NeLo", {}))

        # --- MEMA-Kanten (MeLo -> MaLo) ---
        # Effizient: über die bereits aufgebaute Adjazenz (malos_by_melo) iterieren (O(E) statt O(E * #Comps))
        seen_mema = set()
        for melo_id in sorted(melos):
            for malo_id in sorted(malos_by_melo.get(melo_id, set())):
                if malo_id in malos:
                    key = (melo_id, malo_id)
                    if key not in seen_mema:
                        seen_mema.add(key)
                        edges.append(make_edge(melo_id, malo_id, "MEMA"))

# --- graph_id: bundle_id wenn eindeutig ---
        graph_id = f"comp:{sorted(malos)}|{sorted(melos)}"
        bundle_id = None
        if isinstance(bndl_df, pd.DataFrame) and not bndl_df.empty:
            # bevorzugt: nur Rows, die wirklich innerhalb der Komponente liegen
            subset = bndl_df[
                (bndl_df.get("malo_id").astype(str).isin(list(malos)) if "malo_id" in bndl_df.columns else False) &
                (bndl_df.get("melo_id").astype(str).isin(list(melos)) if "melo_id" in bndl_df.columns else False)
            ] if ("malo_id" in bndl_df.columns and "melo_id" in bndl_df.columns) else pd.DataFrame()

            if subset.empty:
                # fallback (weniger streng): OR-Filter
                subset = bndl_df[
                    (bndl_df.get("malo_id").astype(str).isin(list(malos)) if "malo_id" in bndl_df.columns else False) |
                    (bndl_df.get("melo_id").astype(str).isin(list(melos)) if "melo_id" in bndl_df.columns else False)
                ] if ("malo_id" in bndl_df.columns or "melo_id" in bndl_df.columns) else pd.DataFrame()

            if not subset.empty and "bundle_id" in subset.columns:
                bundles = [b for b in subset["bundle_id"].dropna().astype(str).unique().tolist() if b and b.lower() != "nan"]
                if len(bundles) == 1:
                    bundle_id = bundles[0]
                    graph_id = f"bundle:{bundle_id}"

        # --- Graph-Metadaten / Graph-Features ---

        tr_count = len(seen_tr)
        nelo_count = len(comp_nelos)

        edge_type_counts = defaultdict(int)
        for e in edges:
            edge_type_counts[e.get("rel")] += 1

        graph_attrs = {
            "malo_count": m_count,
            "melo_count": e_count,
            "tr_count": tr_count,
            "nelo_count": nelo_count,
            "edge_type_counts": dict(edge_type_counts),
        }
        if dataset_tag:
            graph_attrs["dataset"] = dataset_tag
        if bundle_id is not None:
            graph_attrs["bundle_id"] = bundle_id

        graphs.append({
            "graph_id": graph_id,
            "nodes": nodes,
            "edges": edges,
            "graph_attrs": graph_attrs
        })

    return graphs


# ------------------------------
# TR-CHECKS (NEUER DATENSATZ)
# ------------------------------
def check_tr_only_with_malo_anlage(t: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Prüft TR.csv (falls vorhanden) auf TR, die *nur* über MaLo-Anlage referenziert werden.

    Hintergrund:
      - In TR.csv existiert typischerweise eine Spalte wie "MaLo (Anlage)".
      - Wenn diese MaLo-IDs **nicht** in POD_REL auftauchen, sind diese TRs im aktuellen
        Graphschema (Anbindung der TR an MeLo via MaLo(Anlage) -> POD_REL) *nicht* eindeutig platzierbar.

    Rückgabe:
      - Counts + Beispiel-IDs, damit du entscheiden kannst, ob du zusätzlich TR-MaLo-Kanten
        modellieren musst (oder ob die TR-Anbindung über POD_REL ausreicht).
    """
    if "tr" not in t:
        return {"available": False, "reason": "no TR table loaded"}

    tr = t["tr"].copy()
    if tr.empty or "tr_id" not in tr.columns:
        return {"available": True, "total_tr": 0}

    # nur Rows mit tr_id
    tr_ids = tr["tr_id"].dropna().astype(str)
    tr_ids = tr_ids[tr_ids.str.strip() != ""]
    total_tr = int(tr_ids.nunique())

    # MaLo-Referenz (Anlage)
    if "malo_id" not in tr.columns:
        return {"available": True, "total_tr": total_tr, "note": "TR table has no malo_id column after canonicalize"}

    tr2 = tr[["tr_id", "malo_id"]].dropna().astype(str)
    tr2["tr_id"] = tr2["tr_id"].str.strip()
    tr2["malo_id"] = tr2["malo_id"].str.strip()
    tr2 = tr2[(tr2["tr_id"] != "") & (tr2["malo_id"] != "")]

    # TR->MaLo Multiplizität
    malos_per_tr = tr2.groupby("tr_id")["malo_id"].nunique()
    tr_multi_malo = sorted(malos_per_tr[malos_per_tr > 1].index.tolist())

    # Orphan-MaLo (TR referenziert MaLo, die es in MALO nicht gibt)
    malo_ids = set(t.get("malo", pd.DataFrame()).get("malo_id", pd.Series(dtype=str)).astype(str).tolist())
    tr_malo_ids = set(tr2["malo_id"].tolist())
    orphan_malo_refs = sorted(list(tr_malo_ids - malo_ids)) if malo_ids else []

    # TR, deren MaLo-Anlage nicht in POD_REL auftaucht (also keine MeLo-Verknüpfung)
    pod_malo_ids = set(t.get("pod_rel", pd.DataFrame()).get("malo_id", pd.Series(dtype=str)).astype(str).tolist())
    malo_without_podrel = sorted(list(tr_malo_ids - pod_malo_ids)) if pod_malo_ids else sorted(list(tr_malo_ids))

    return {
        "available": True,
        "total_tr": total_tr,
        "rows_with_tr_and_malo": int(len(tr2)),
        "tr_with_multiple_malos": {
            "count": int(len(tr_multi_malo)),
            "examples": tr_multi_malo[:20],
        },
        "orphan_malo_refs": {
            "count": int(len(orphan_malo_refs)),
            "examples": orphan_malo_refs[:20],
        },
        "malo_refs_without_podrel": {
            "count": int(len(malo_without_podrel)),
            "examples": malo_without_podrel[:20],
        },
    }
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
    # Durch NeLo-Relationen können in POD_REL Zeilen ohne MaLo vorkommen.
    # Für diese Validierung betrachten wir nur echte MaLo–MeLo-Kanten.
    pr["malo_id"] = pr["malo_id"].astype(str).str.strip()
    pr["melo_id"] = pr["melo_id"].astype(str).str.strip()
    pr = pr[(pr["malo_id"] != "") & (pr["melo_id"] != "") & (pr["malo_id"].str.lower() != "nan") & (pr["melo_id"].str.lower() != "nan")]
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

    datasets = table_loader_all(BASE)

    all_graphs: List[Dict[str, Any]] = []

    for tag, raw_tables in datasets.items():
        tables = canonicalize(raw_tables)

        report = relation_validation(tables)
        print(f"\n[{tag}] Ist-Graphen Check:")
        print(json.dumps(report, indent=2, ensure_ascii=False))

        # Optional: TR-Analyse für neuen Datensatz
        if tag == "new":
            tr_report = check_tr_only_with_malo_anlage(tables)
            print(f"\n[{tag}] TR-Check (MaLo Anlage):")
            print(json.dumps(tr_report, indent=2, ensure_ascii=False))

        graphs = build_graphs(tables, restrict_to=None, dataset_tag=tag)
        all_graphs.extend(graphs)

        from collections import Counter
        print(f"[{tag}] Gebaut: {len(graphs)} Graphen")

        # Dataset-spezifischer Export
        #out_name = "ist_graphs.jsonl" if tag == "sdf" else "ist_graphs_pro.jsonl"
        #out_path = os.path.join(BASE, "data", out_name)
        #os.makedirs(os.path.dirname(out_path), exist_ok=True)
        #with open(out_path, "w", encoding="utf-8") as f:
        #    for g in graphs:
        #        f.write(json.dumps(g, ensure_ascii=False) + "\n")
        #print(f"[{tag}] Export: {out_path}")

    # Gesamtexport (beide Datensätze zusammen)
    out_all = os.path.join(BASE, "data", "ist_graphs_all.jsonl")
    with open(out_all, "w", encoding="utf-8") as f:
        for g in all_graphs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"\n[all] Export: {out_all}")

    print("Prozess ausgeführt in %s Sekunden" % (time.time() - start_time))
