# Deskriptive Analyse: Ist-Graphen vs. LBS-Soll-Templates

## Inputs

- Ist-Graphs: `data\ist_graphs_all.jsonl`
- Soll-Templates: `data\lbs_soll_graphs_pro.jsonl`
- Max graphs: _alle_
- Plots: an (matplotlib gefunden)
- pandas: gefunden
- graph_pipeline Import: ok

Geladen: **2521** Ist-Graphen, **15** Template-Graphen.

## Schema- & Qualitätschecks

- Issues total: 0
- ERROR: 0
- WARN: 0

### Häufigste Issue-Typen

Tabellen-Export: `analysis/descriptive/tables/issues_top.csv`

_Keine Daten._

## Inventar: Welche Felder sind enthalten?

### Ist: graph_attrs Keys (wie oft pro Graph vorhanden)

Tabellen-Export: `analysis/descriptive/tables/ist_graph_attrs_keys.csv`

| key              |   count | pct    |
|:-----------------|--------:|:-------|
| malo_count       |    2521 | 100.0% |
| melo_count       |    2521 | 100.0% |
| tr_count         |    2521 | 100.0% |
| nelo_count       |    2521 | 100.0% |
| edge_type_counts |    2521 | 100.0% |
| dataset          |    2521 | 100.0% |
| bundle_id        |      48 | 1.9%   |

### Soll: graph_attrs Keys (wie oft pro Graph vorhanden)

Tabellen-Export: `analysis/descriptive/tables/soll_graph_attrs_keys.csv`

| key                     |   count | pct    |
|:------------------------|--------:|:-------|
| malo_min                |      15 | 100.0% |
| melo_min                |      15 | 100.0% |
| tr_min                  |      15 | 100.0% |
| nelo_min                |      15 | 100.0% |
| malo_max                |      15 | 100.0% |
| melo_max                |      15 | 100.0% |
| tr_max                  |      15 | 100.0% |
| nelo_max                |      15 | 100.0% |
| malo_node_types         |      15 | 100.0% |
| melo_node_types         |      15 | 100.0% |
| tr_node_types           |      15 | 100.0% |
| nelo_node_types         |      15 | 100.0% |
| is_template             |      15 | 100.0% |
| lbs_code                |      15 | 100.0% |
| optionality_constraints |      15 | 100.0% |
| attachment_rules        |      15 | 100.0% |

### Ist: Node-Attribute Keys (wie oft über alle Nodes)

Tabellen-Export: `analysis/descriptive/tables/ist_node_attr_keys.csv`

| key           |   count |
|:--------------|--------:|
| direction     |    5385 |
| voltage_level |    2624 |

### Soll: Node-Attribute Keys (wie oft über alle Nodes)

Tabellen-Export: `analysis/descriptive/tables/soll_node_attr_keys.csv`

| key            |   count |
|:---------------|--------:|
| level          |     127 |
| object_code    |     127 |
| min_occurs     |     127 |
| max_occurs     |     127 |
| flexibility    |     127 |
| optional       |     127 |
| direction      |      75 |
| tr_direction   |      38 |
| function       |      28 |
| dynamic        |      28 |
| direction_hint |      28 |

## Struktur: Knotentypen & Kantentypen

### Knotentyp-Verteilung (gesamt)

Tabellen-Export: `analysis/descriptive/tables/node_type_distribution.csv`

| type   |   ist_count |   soll_count |
|:-------|------------:|-------------:|
| MaLo   |        2761 |           38 |
| MeLo   |        2624 |           28 |
| NeLo   |           0 |           15 |
| TR     |        2624 |           46 |

### Kantentyp-Verteilung (gesamt)

Tabellen-Export: `analysis/descriptive/tables/edge_type_distribution.csv`

| rel   |   ist_count |   soll_count |
|:------|------------:|-------------:|
| MEMA  |        2864 |           31 |
| MENE  |           0 |           12 |
| METR  |        2624 |           35 |

## Pro-Graph Kennzahlen

### Größenverteilungen (Nodes/Edges)

Tabellen-Export: `analysis/descriptive/tables/size_distributions.csv`

| kind   |   graphs |   nodes_min |   nodes_med |   nodes_max |   edges_min |   edges_med |   edges_max |
|:-------|---------:|------------:|------------:|------------:|------------:|------------:|------------:|
| ist    |     2521 |           3 |           3 |          11 |           2 |           2 |          10 |
| soll   |       15 |           3 |           9 |          13 |           0 |           5 |           9 |

Plots gespeichert unter `plots/` (siehe Dateien).

### Connectedness / Komponenten

Tabellen-Export: `analysis/descriptive/tables/connectivity_summary.csv`

| kind   |   graphs |   cc_count_mean |   cc_count_max |   largest_cc_mean |   largest_cc_min |
|:-------|---------:|----------------:|---------------:|------------------:|-----------------:|
| ist    |     2521 |           1     |              1 |             3.177 |                3 |
| soll   |       15 |           3.267 |              9 |             4.667 |                1 |

### Degree-Statistiken pro Knotentyp (undirected)

Tabellen-Export: `analysis/descriptive/tables/degree_stats_ist.csv`

| type   |   count |   deg_mean |   deg_min |   deg_max |
|:-------|--------:|-----------:|----------:|----------:|
| MaLo   |    2761 |      1.037 |         1 |         3 |
| MeLo   |    2624 |      2.091 |         2 |         4 |
| TR     |    2624 |      1     |         1 |         1 |

Tabellen-Export: `analysis/descriptive/tables/degree_stats_soll.csv`

| type   |   count |   deg_mean |   deg_min |   deg_max |
|:-------|--------:|-----------:|----------:|----------:|
| MaLo   |      38 |      0.816 |         0 |         2 |
| MeLo   |      28 |      2.786 |         0 |         6 |
| NeLo   |      15 |      0.8   |         0 |         1 |
| TR     |      46 |      0.761 |         0 |         1 |

## Attribut-Vollständigkeit & Value-Distributionen

### Coverage: zentrale Attribute (Ist)

Tabellen-Export: `analysis/descriptive/tables/attr_coverage_ist.csv`

| kind   | type   | attr           |   present |   total | present_pct   |
|:-------|:-------|:---------------|----------:|--------:|:--------------|
| ist    | MaLo   | direction      |      2657 |    2761 | 96.2%         |
| ist    | MaLo   | direction_hint |         0 |    2761 | 0.0%          |
| ist    | MeLo   | direction_hint |         0 |    2624 | 0.0%          |
| ist    | MeLo   | dynamic        |         0 |    2624 | 0.0%          |
| ist    | MeLo   | function       |         0 |    2624 | 0.0%          |
| ist    | MeLo   | melo_function  |         0 |    2624 | 0.0%          |
| ist    | MeLo   | voltage_level  |      2624 |    2624 | 100.0%        |
| ist    | TR     | direction      |      2378 |    2624 | 90.6%         |
| ist    | TR     | tr_direction   |         0 |    2624 | 0.0%          |

### Coverage: zentrale Attribute (Soll/Templates)

Tabellen-Export: `analysis/descriptive/tables/attr_coverage_soll.csv`

| kind   | type   | attr           |   present |   total | present_pct   |
|:-------|:-------|:---------------|----------:|--------:|:--------------|
| soll   | MaLo   | direction      |        37 |      38 | 97.4%         |
| soll   | MaLo   | direction_hint |         0 |      38 | 0.0%          |
| soll   | MeLo   | direction_hint |        28 |      28 | 100.0%        |
| soll   | MeLo   | dynamic        |        28 |      28 | 100.0%        |
| soll   | MeLo   | function       |        28 |      28 | 100.0%        |
| soll   | MeLo   | melo_function  |         0 |      28 | 0.0%          |
| soll   | MeLo   | voltage_level  |         0 |      28 | 0.0%          |
| soll   | TR     | direction      |        38 |      46 | 82.6%         |
| soll   | TR     | tr_direction   |        38 |      46 | 82.6%         |

### Top Values: Richtung (MaLo)

Tabellen-Export: `analysis/descriptive/tables/top_values_ist_MaLo_direction.csv`

| kind   | type   | attr      | value       |   count |
|:-------|:-------|:----------|:------------|--------:|
| ist    | MaLo   | direction | consumption |    2390 |
| ist    | MaLo   | direction | generation  |     267 |

Tabellen-Export: `analysis/descriptive/tables/top_values_soll_MaLo_direction.csv`

| kind   | type   | attr      | value       |   count |
|:-------|:-------|:----------|:------------|--------:|
| soll   | MaLo   | direction | consumption |      24 |
| soll   | MaLo   | direction | generation  |      13 |

### Top Values: Spannungsebene (MeLo.voltage_level)

Tabellen-Export: `analysis/descriptive/tables/top_values_ist_MeLo_voltage_level.csv`

| kind   | type   | attr          | value   |   count |
|:-------|:-------|:--------------|:--------|--------:|
| ist    | MeLo   | voltage_level | E06     |    2552 |
| ist    | MeLo   | voltage_level | 41      |      66 |
| ist    | MeLo   | voltage_level | E05     |       6 |

Tabellen-Export: `analysis/descriptive/tables/top_values_soll_MeLo_voltage_level.csv`

_Keine Daten._

### Top Values: MeLo-Funktion (Templates sollten function/melo_function tragen)

Tabellen-Export: `analysis/descriptive/tables/top_values_ist_MeLo_function.csv`

_Keine Daten._

Tabellen-Export: `analysis/descriptive/tables/top_values_ist_MeLo_melo_function.csv`

_Keine Daten._

Tabellen-Export: `analysis/descriptive/tables/top_values_soll_MeLo_function.csv`

| kind   | type   | attr     | value   |   count |
|:-------|:-------|:---------|:--------|--------:|
| soll   | MeLo   | function | N       |      16 |
| soll   | MeLo   | function | H       |       8 |
| soll   | MeLo   | function | D       |       4 |

Tabellen-Export: `analysis/descriptive/tables/top_values_soll_MeLo_melo_function.csv`

_Keine Daten._

### Top Values: TR Richtung (tr_direction)

Tabellen-Export: `analysis/descriptive/tables/top_values_ist_TR_tr_direction.csv`

_Keine Daten._

Tabellen-Export: `analysis/descriptive/tables/top_values_soll_TR_tr_direction.csv`

| kind   | type   | attr         | value                           |   count |
|:-------|:-------|:-------------|:--------------------------------|--------:|
| soll   | TR     | tr_direction | consumption                     |      17 |
| soll   | TR     | tr_direction | generation                      |      11 |
| soll   | TR     | tr_direction | consumption+generation(storage) |      10 |

## Template-spezifische Analyse (LBS-Schema)

Tabellen-Export: `analysis/descriptive/tables/template_optionality_summary.csv`

|      lbs_code | graph_id              |   nodes_total |   MaLo_nodes |   MeLo_nodes |   TR_nodes |   NeLo_nodes |   optional_nodes |   min0_nodes |   flexible_nodes |   attachment_rules_n |   constraints_n |
|--------------:|:----------------------|--------------:|-------------:|-------------:|-----------:|-------------:|-----------------:|-------------:|-----------------:|---------------------:|----------------:|
| 9992000000018 | catalog-9992000000018 |             3 |            1 |            0 |          1 |            1 |                2 |            2 |                0 |                    0 |               0 |
| 9992000000026 | catalog-9992000000026 |             4 |            1 |            1 |          1 |            1 |                2 |            2 |                0 |                    0 |               0 |
| 9992000000034 | catalog-9992000000034 |             6 |            2 |            1 |          2 |            1 |                4 |            4 |                0 |                    0 |               0 |
| 9992000000042 | catalog-9992000000042 |             7 |            2 |            1 |          3 |            1 |                4 |            4 |                0 |                    0 |               0 |
| 9992000000068 | catalog-9992000000068 |             7 |            2 |            2 |          2 |            1 |                3 |            3 |                0 |                    0 |               2 |
| 9992000000076 | catalog-9992000000076 |            10 |            3 |            2 |          4 |            1 |                5 |            5 |                0 |                    0 |               2 |
| 9992000000084 | catalog-9992000000084 |            12 |            4 |            3 |          4 |            1 |                6 |            6 |                0 |                    6 |               2 |
| 9992000000109 | catalog-9992000000109 |             4 |            1 |            1 |          1 |            1 |                2 |            2 |                0 |                    0 |               1 |
| 9992000000117 | catalog-9992000000117 |             9 |            2 |            2 |          4 |            1 |                6 |            6 |                0 |                    7 |               0 |
| 9992000000125 | catalog-9992000000125 |            11 |            3 |            3 |          4 |            1 |                7 |            7 |                0 |                    1 |               1 |
| 9992000000133 | catalog-9992000000133 |             9 |            3 |            2 |          3 |            1 |                5 |            5 |                0 |                    0 |               1 |
| 9992000000159 | catalog-9992000000159 |            10 |            3 |            2 |          4 |            1 |                5 |            5 |                0 |                    0 |               1 |
| 9992000000167 | catalog-9992000000167 |            13 |            4 |            3 |          5 |            1 |                7 |            7 |                0 |                    5 |               2 |
| 9992000000175 | catalog-9992000000175 |            10 |            3 |            2 |          4 |            1 |                5 |            5 |                0 |                    0 |               0 |
| 9992000000183 | catalog-9992000000183 |            12 |            4 |            3 |          4 |            1 |                6 |            6 |                0 |                    0 |               1 |

## Ist vs Soll: Bounds-Coverage (Kandidatenfilter ohne pattern)

Diese Tabelle zeigt pro Template, wie viele Ist-Graphen die Min/Max-Bounds erfüllen.

Tabellen-Export: `analysis/descriptive/tables/template_coverage.csv`

|      lbs_code | templates_graph_id    |   malo_min |   malo_max |   melo_min |   melo_max |   tr_min | tr_max   |   nelo_min |   nelo_max |   ist_graphs_fitting_total |   ist_fitting_sdf |   ist_fitting_new |
|--------------:|:----------------------|-----------:|-----------:|-----------:|-----------:|---------:|:---------|-----------:|-----------:|---------------------------:|------------------:|------------------:|
| 9992000000034 | catalog-9992000000034 |          1 |          2 |          1 |          1 |        0 |          |          0 |          1 |                       2419 |              2389 |                30 |
| 9992000000026 | catalog-9992000000026 |          1 |          1 |          1 |          1 |        0 |          |          0 |          1 |                       2303 |              2293 |                10 |
| 9992000000117 | catalog-9992000000117 |          2 |        nan |          1 |        nan |        0 |          |          0 |        nan |                        218 |               180 |                38 |
| 9992000000042 | catalog-9992000000042 |          2 |          2 |          1 |          1 |        0 |          |          0 |          1 |                        116 |                96 |                20 |
| 9992000000068 | catalog-9992000000068 |          2 |        nan |          2 |        nan |        0 |          |          0 |          1 |                        101 |                83 |                18 |
| 9992000000125 | catalog-9992000000125 |          2 |        nan |          2 |        nan |        0 |          |          0 |        nan |                        101 |                83 |                18 |
| 9992000000133 | catalog-9992000000133 |          2 |        nan |          2 |        nan |        0 |          |          0 |          1 |                        101 |                83 |                18 |
| 9992000000076 | catalog-9992000000076 |          3 |        nan |          2 |        nan |        0 |          |          0 |          1 |                         21 |                 3 |                18 |
| 9992000000159 | catalog-9992000000159 |          3 |          3 |          2 |          2 |        0 |          |          0 |          1 |                         20 |                 2 |                18 |
| 9992000000175 | catalog-9992000000175 |          3 |          3 |          2 |          2 |        0 |          |          0 |          1 |                         20 |                 2 |                18 |
| 9992000000084 | catalog-9992000000084 |          3 |        nan |          3 |        nan |        0 |          |          0 |          1 |                          1 |                 1 |               nan |
| 9992000000167 | catalog-9992000000167 |          3 |        nan |          3 |        nan |        0 |          |          0 |          1 |                          1 |                 1 |               nan |
| 9992000000018 | catalog-9992000000018 |          1 |          1 |          0 |          0 |        0 |          |          0 |        nan |                          0 |               nan |               nan |
| 9992000000109 | catalog-9992000000109 |          1 |          1 |          2 |        nan |        0 |          |          0 |        nan |                          0 |               nan |               nan |
| 9992000000183 | catalog-9992000000183 |          3 |          4 |          3 |          3 |        0 |          |          0 |          1 |                          0 |               nan |               nan |

## Abgleich mit Feature-Encoding (graph_pipeline.py)

### Unknown-Raten für die verwendeten Feature-Blöcke

Tabellen-Export: `analysis/descriptive/tables/encoder_unknown_rates.csv`

| kind   | metric               | count              |
|:-------|:---------------------|:-------------------|
| ist    | nodes_total          | 8009               |
| ist    | edges_total          | 5488               |
| ist    | dir_unknown_rate     | 6.5% (350/5385)    |
| ist    | melo_fn_unknown_rate | 100.0% (2624/2624) |
| ist    | volt_unknown_rate    | 2.5% (66/2624)     |
| soll   | nodes_total          | 127                |
| soll   | edges_total          | 78                 |
| soll   | dir_unknown_rate     | 10.7% (9/84)       |
| soll   | melo_fn_unknown_rate | 0.0% (0/28)        |
| soll   | volt_unknown_rate    | 100.0% (28/28)     |

### Welche Felder gehen aktuell ins Modell ein?

Der aktuelle Feature-Encoder (graph_pipeline.py) reduziert die reichhaltigen Node-Attrs auf wenige One-Hot-Blöcke. Das ist wichtig, um zu beurteilen, ob im Graph unnötige Daten mitgeschleppt werden oder ob entscheidende Informationen fehlen.

- **Knotentyp**: `node['type']` → One-Hot über {MaLo, MeLo, TR, NeLo}
- **Richtung** (nur MaLo/TR): `attrs['direction']` bzw. TR: `attrs['tr_direction']`; Fallback MaLo: `attrs['direction_hint']` → {consumption, generation, both, unknown}
- **MeLo-Funktion** (nur MeLo): `attrs['function']` oder `attrs['melo_function']` → {N, H, D, S, unknown}
- **Spannungsebene** (nur MeLo): `attrs['voltage_level']` → {E05, E06, unknown}
- **Kantentyp**: `edge['rel']` → One-Hot über {MEMA, METR, MENE, MEME, unknown}
- Alle anderen Node-Attribute (z.B. `min_occurs/max_occurs/flexibility/optional/object_code/level/dynamic/attachment_rules`) werden aktuell **nicht** als ML-Feature kodiert.

### Welche Node-Attribute sind im Datensatz vorhanden, werden aber vom Encoder ignoriert?

Ist-Graphen (Top ignorierte Keys):

Tabellen-Export: `analysis/descriptive/tables/ist_unused_node_attrs.csv`

_Keine Daten._

Template-Graphen (Top ignorierte Keys):

Tabellen-Export: `analysis/descriptive/tables/soll_unused_node_attrs.csv`

| key         |   count |
|:------------|--------:|
| level       |     127 |
| object_code |     127 |
| min_occurs  |     127 |
| max_occurs  |     127 |
| flexibility |     127 |
| optional    |     127 |
| dynamic     |      28 |

## Automatische Beobachtungen (Differenzen & mögliche Ursachen)

- Templates enthalten Knotentyp(e) ['NeLo'], die in Ist-Graphen nicht vorkommen (typisch: NeLo). Ursache: Ist-Converter modelliert diese Relationen/Knoten derzeit nicht.
- Templates enthalten Kanten-Typ(e) ['MENE'], die in Ist-Graphen fehlen. Ursache: Ist-Converter baut aktuell nur MEMA/METR.
- MeLo-Funktion ist in Ist-Graphen fast immer 'unknown' (Feature-Block wird vom Modell kaum genutzt). Ursache: Ist-Graphen tragen i.d.R. kein function/melo_function.
- Spannungsebene ist in Template-Graphen fast immer 'unknown' (Feature-Block wird vom Modell kaum genutzt). Ursache: Templates enthalten i.d.R. keine voltage_level-Angabe.
