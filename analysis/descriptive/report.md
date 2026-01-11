# Deskriptive Analyse: Ist-Graphen vs. LBS-Soll-Templates

## Inputs

- Ist-Graphs: `data\ist_graphs_all.jsonl`
- Soll-Templates: `data\lbs_soll_graphs_pro.jsonl`
- Max graphs: _alle_
- Plots: an (matplotlib gefunden)
- pandas: gefunden
- graph_pipeline Import: ok
- Strukturmetriken: nur valide Kanten (src/dst referenzieren existierende Nodes).

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

| key                           |   count |
|:------------------------------|--------:|
| direction                     |    2797 |
| voltage_level                 |    2624 |
| source                        |      36 |
| malo_ref                      |      36 |
| anlage                        |      36 |
| bisdatum                      |      36 |
| abdatum                       |      36 |
| art_der_technischen_ressource |      36 |
| verbrauchsart                 |      36 |
| wrmenutzung                   |      36 |
| ref_zur_tr                    |      36 |

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
| TR     |          36 |           46 |

### Kantentyp-Verteilung (gesamt; nur valide Kanten)

Tabellen-Export: `analysis/descriptive/tables/edge_type_distribution.csv`

| rel   |   ist_count |   soll_count |
|:------|------------:|-------------:|
| MEMA  |        2864 |           31 |
| MENE  |           0 |           12 |
| METR  |          47 |           35 |

## Pro-Graph Kennzahlen

### Per-Graph-Tabelle (Sample)

Hinweis: Vollständige Tabellen liegen als CSV unter `tables/`.

Tabellen-Export: `analysis/descriptive/tables/per_graph_ist.csv`

| kind   | graph_id                                                                                                       |   node_count |   edge_count |   edges_total |   edges_with_srcdst |   edges_with_rel |   edges_valid_endpoints |   edges_valid_typed |   edges_missing_srcdst |   edges_missing_rel |   edges_missing_refs |   edges_self_loops_valid |   node_count_unique |   edge_pairs_undirected_unique |   density_undirected_simple |   edges_per_node_valid_endpoints |   edges_per_node_valid_typed |   avg_degree_undirected |   deg_min_undirected |   deg_med_undirected |   deg_max_undirected |   isolated_nodes_undirected |   isolated_pct_undirected |   cc_count_undirected |   cc_largest_undirected |   avg_out_degree_directed |   avg_in_degree_directed |   out_degree_max_directed |   in_degree_max_directed |   scc_count_directed |   scc_largest_directed | dataset   |   malo_count |   melo_count |   tr_count |   nelo_count |   edge_MEMA |   edge_METR |   edge_MENE |   edge_MEME |
|:-------|:---------------------------------------------------------------------------------------------------------------|-------------:|-------------:|--------------:|--------------------:|-----------------:|------------------------:|--------------------:|-----------------------:|--------------------:|---------------------:|-------------------------:|--------------------:|-------------------------------:|----------------------------:|---------------------------------:|-----------------------------:|------------------------:|---------------------:|---------------------:|---------------------:|----------------------------:|--------------------------:|----------------------:|------------------------:|--------------------------:|-------------------------:|--------------------------:|-------------------------:|---------------------:|-----------------------:|:----------|-------------:|-------------:|-----------:|-------------:|------------:|------------:|------------:|------------:|
| ist    | comp:['50414974078']|['DE0071947684600000000000000000341']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414794286', '50415648028']|['DE0071947684600000000000000000738']                                      |            3 |            2 |             2 |                   2 |                2 |                       2 |                   2 |                      0 |                   0 |                    0 |                        0 |                   3 |                              2 |                    0.666667 |                         0.666667 |                     0.666667 |                 1.33333 |                    1 |                    1 |                    2 |                           0 |                         0 |                     1 |                       3 |                  0.666667 |                 0.666667 |                         2 |                        1 |                    3 |                      1 | sdf       |            2 |            1 |          0 |            0 |           2 |           0 |           0 |           0 |
| ist    | comp:['50414989928']|['DE0071947684600000000000000000530']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414960150', '50415720941']|['DE0071947684600000000000000000183']                                      |            3 |            2 |             2 |                   2 |                2 |                       2 |                   2 |                      0 |                   0 |                    0 |                        0 |                   3 |                              2 |                    0.666667 |                         0.666667 |                     0.666667 |                 1.33333 |                    1 |                    1 |                    2 |                           0 |                         0 |                     1 |                       3 |                  0.666667 |                 0.666667 |                         2 |                        1 |                    3 |                      1 | sdf       |            2 |            1 |          0 |            0 |           2 |           0 |           0 |           0 |
| ist    | comp:['50414943297']|['DE0071947684600000000000000002702']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414931664']|['DE0071947684600000000000000002357']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414791662']|['DE0071947684600000000000000000708']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414951084']|['DE0071947684600000000000000002920']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414914602']|['DE0071947684600000000000000002112']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414840120']|['DE0071947684600000000000000001247']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414863156', '50415025416']|['DE0071947684600000000000000001508', 'DE0071947684600000000000000002800'] |            4 |            3 |             3 |                   3 |                3 |                       3 |                   3 |                      0 |                   0 |                    0 |                        0 |                   4 |                              3 |                    0.5      |                         0.75     |                     0.75     |                 1.5     |                    1 |                    2 |                    2 |                           0 |                         0 |                     1 |                       4 |                  0.75     |                 0.75     |                         2 |                        2 |                    4 |                      1 | sdf       |            2 |            2 |          0 |            0 |           3 |           0 |           0 |           0 |
| ist    | comp:['50414854204']|['DE0071947684600000000000000001409']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414958189']|['DE0071947684600000000000000000161']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414948966', '50415607404']|['DE0071947684600000000000000002878']                                      |            3 |            2 |             2 |                   2 |                2 |                       2 |                   2 |                      0 |                   0 |                    0 |                        0 |                   3 |                              2 |                    0.666667 |                         0.666667 |                     0.666667 |                 1.33333 |                    1 |                    1 |                    2 |                           0 |                         0 |                     1 |                       3 |                  0.666667 |                 0.666667 |                         2 |                        1 |                    3 |                      1 | sdf       |            2 |            1 |          0 |            0 |           2 |           0 |           0 |           0 |
| ist    | comp:['50414970430']|['DE0071947684600000000000000000301']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414839462']|['DE0071947684600000000000000001240']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414998622']|['DE0071947684600000000000000000631']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50415014807']|['DE0071947684600000000000000002560']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50415527321']|['DE0003967684600000000000053025783']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414880449']|['DE0071947684600000000000000001706']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414846706']|['DE0071947684600000000000000001320']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414987104']|['DE0071947684600000000000000000492']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50414887114']|['DE0071947684600000000000000001779']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |
| ist    | comp:['50415608204', '50415654158']|['DE0003967684600000000000053032357']                                      |            3 |            2 |             2 |                   2 |                2 |                       2 |                   2 |                      0 |                   0 |                    0 |                        0 |                   3 |                              2 |                    0.666667 |                         0.666667 |                     0.666667 |                 1.33333 |                    1 |                    1 |                    2 |                           0 |                         0 |                     1 |                       3 |                  0.666667 |                 0.666667 |                         2 |                        1 |                    3 |                      1 | sdf       |            2 |            1 |          0 |            0 |           2 |           0 |           0 |           0 |
| ist    | comp:['50414807758']|['DE0071947684600000000000000000887']                                                     |            2 |            1 |             1 |                   1 |                1 |                       1 |                   1 |                      0 |                   0 |                    0 |                        0 |                   2 |                              1 |                    1        |                         0.5      |                     0.5      |                 1       |                    1 |                    1 |                    1 |                           0 |                         0 |                     1 |                       2 |                  0.5      |                 0.5      |                         1 |                        1 |                    2 |                      1 | sdf       |            1 |            1 |          0 |            0 |           1 |           0 |           0 |           0 |

_Hinweis: Tabelle gekürzt auf 25 Zeilen (insgesamt 2521)._

Tabellen-Export: `analysis/descriptive/tables/per_graph_soll.csv`

| kind   | graph_id              |   node_count |   edge_count |   edges_total |   edges_with_srcdst |   edges_with_rel |   edges_valid_endpoints |   edges_valid_typed |   edges_missing_srcdst |   edges_missing_rel |   edges_missing_refs |   edges_self_loops_valid |   node_count_unique |   edge_pairs_undirected_unique |   density_undirected_simple |   edges_per_node_valid_endpoints |   edges_per_node_valid_typed |   avg_degree_undirected |   deg_min_undirected |   deg_med_undirected |   deg_max_undirected |   isolated_nodes_undirected |   isolated_pct_undirected |   cc_count_undirected |   cc_largest_undirected |   avg_out_degree_directed |   avg_in_degree_directed |   out_degree_max_directed |   in_degree_max_directed |   scc_count_directed |   scc_largest_directed |      lbs_code |   malo_min |   malo_max |   malo_node_types |   melo_min |   melo_max |   melo_node_types |   tr_min | tr_max   |   tr_node_types |   nelo_min |   nelo_max |   nelo_node_types |   attachment_rules_n |   constraints_n |   edge_MEMA |   edge_METR |   edge_MENE |   edge_MEME |
|:-------|:----------------------|-------------:|-------------:|--------------:|--------------------:|-----------------:|------------------------:|--------------------:|-----------------------:|--------------------:|---------------------:|-------------------------:|--------------------:|-------------------------------:|----------------------------:|---------------------------------:|-----------------------------:|------------------------:|---------------------:|---------------------:|---------------------:|----------------------------:|--------------------------:|----------------------:|------------------------:|--------------------------:|-------------------------:|--------------------------:|-------------------------:|---------------------:|-----------------------:|--------------:|-----------:|-----------:|------------------:|-----------:|-----------:|------------------:|---------:|:---------|----------------:|-----------:|-----------:|------------------:|---------------------:|----------------:|------------:|------------:|------------:|------------:|
| soll   | catalog-9992000000018 |            3 |            0 |             0 |                   0 |                0 |                       0 |                   0 |                      0 |                   0 |                    0 |                        0 |                   3 |                              0 |                   0         |                         0        |                     0        |                0        |                    0 |                    0 |                    0 |                           3 |                 100       |                     3 |                       1 |                  0        |                 0        |                         0 |                        0 |                    3 |                      1 | 9992000000018 |          1 |          1 |                 1 |          0 |          0 |                 0 |        0 |          |               1 |          0 |        nan |                 1 |                    0 |               0 |           0 |           0 |           0 |           0 |
| soll   | catalog-9992000000026 |            4 |            3 |             3 |                   3 |                3 |                       3 |                   3 |                      0 |                   0 |                    0 |                        0 |                   4 |                              3 |                   0.5       |                         0.75     |                     0.75     |                1.5      |                    1 |                    1 |                    3 |                           0 |                   0       |                     1 |                       4 |                  0.75     |                 0.75     |                         3 |                        1 |                    4 |                      1 | 9992000000026 |          1 |          1 |                 1 |          1 |          1 |                 1 |        0 |          |               1 |          0 |          1 |                 1 |                    0 |               0 |           1 |           1 |           1 |           0 |
| soll   | catalog-9992000000034 |            6 |            5 |             5 |                   5 |                5 |                       5 |                   5 |                      0 |                   0 |                    0 |                        0 |                   6 |                              5 |                   0.333333  |                         0.833333 |                     0.833333 |                1.66667  |                    1 |                    1 |                    5 |                           0 |                   0       |                     1 |                       6 |                  0.833333 |                 0.833333 |                         5 |                        1 |                    6 |                      1 | 9992000000034 |          1 |          2 |                 2 |          1 |          1 |                 1 |        0 |          |               2 |          0 |          1 |                 1 |                    0 |               0 |           2 |           2 |           1 |           0 |
| soll   | catalog-9992000000042 |            7 |            6 |             6 |                   6 |                6 |                       6 |                   6 |                      0 |                   0 |                    0 |                        0 |                   7 |                              6 |                   0.285714  |                         0.857143 |                     0.857143 |                1.71429  |                    1 |                    1 |                    6 |                           0 |                   0       |                     1 |                       7 |                  0.857143 |                 0.857143 |                         6 |                        1 |                    7 |                      1 | 9992000000042 |          2 |          2 |                 2 |          1 |          1 |                 1 |        0 |          |               3 |          0 |          1 |                 1 |                    0 |               0 |           2 |           3 |           1 |           0 |
| soll   | catalog-9992000000068 |            7 |            5 |             5 |                   5 |                5 |                       5 |                   5 |                      0 |                   0 |                    0 |                        0 |                   7 |                              5 |                   0.238095  |                         0.714286 |                     0.714286 |                1.42857  |                    1 |                    1 |                    3 |                           0 |                   0       |                     2 |                       4 |                  0.714286 |                 0.714286 |                         3 |                        1 |                    7 |                      1 | 9992000000068 |          2 |        nan |                 2 |          2 |        nan |                 2 |        0 |          |               2 |          0 |          1 |                 1 |                    0 |               2 |           2 |           2 |           1 |           0 |
| soll   | catalog-9992000000076 |           10 |            8 |             8 |                   8 |                8 |                       8 |                   8 |                      0 |                   0 |                    0 |                        0 |                  10 |                              8 |                   0.177778  |                         0.8      |                     0.8      |                1.6      |                    1 |                    1 |                    6 |                           0 |                   0       |                     2 |                       7 |                  0.8      |                 0.8      |                         6 |                        1 |                   10 |                      1 | 9992000000076 |          3 |        nan |                 3 |          2 |        nan |                 2 |        0 |          |               4 |          0 |          1 |                 1 |                    0 |               2 |           3 |           4 |           1 |           0 |
| soll   | catalog-9992000000084 |           12 |            3 |             3 |                   3 |                3 |                       3 |                   3 |                      0 |                   0 |                    0 |                        0 |                  12 |                              3 |                   0.0454545 |                         0.25     |                     0.25     |                0.5      |                    0 |                    0 |                    3 |                           8 |                  66.6667  |                     9 |                       4 |                  0.25     |                 0.25     |                         3 |                        1 |                   12 |                      1 | 9992000000084 |          3 |        nan |                 4 |          3 |        nan |                 3 |        0 |          |               4 |          0 |          1 |                 1 |                    6 |               2 |           1 |           1 |           1 |           0 |
| soll   | catalog-9992000000109 |            4 |            3 |             3 |                   3 |                3 |                       3 |                   3 |                      0 |                   0 |                    0 |                        0 |                   4 |                              3 |                   0.5       |                         0.75     |                     0.75     |                1.5      |                    1 |                    1 |                    3 |                           0 |                   0       |                     1 |                       4 |                  0.75     |                 0.75     |                         3 |                        1 |                    4 |                      1 | 9992000000109 |          1 |          1 |                 1 |          2 |        nan |                 1 |        0 |          |               1 |          0 |        nan |                 1 |                    0 |               1 |           1 |           1 |           1 |           0 |
| soll   | catalog-9992000000117 |            9 |            0 |             0 |                   0 |                0 |                       0 |                   0 |                      0 |                   0 |                    0 |                        0 |                   9 |                              0 |                   0         |                         0        |                     0        |                0        |                    0 |                    0 |                    0 |                           9 |                 100       |                     9 |                       1 |                  0        |                 0        |                         0 |                        0 |                    9 |                      1 | 9992000000117 |          2 |        nan |                 2 |          1 |        nan |                 2 |        0 |          |               4 |          0 |        nan |                 1 |                    7 |               0 |           0 |           0 |           0 |           0 |
| soll   | catalog-9992000000125 |           11 |            8 |             8 |                   8 |                8 |                       8 |                   8 |                      0 |                   0 |                    0 |                        0 |                  11 |                              8 |                   0.145455  |                         0.727273 |                     0.727273 |                1.45455  |                    0 |                    1 |                    4 |                           1 |                   9.09091 |                     3 |                       5 |                  0.727273 |                 0.727273 |                         4 |                        2 |                   11 |                      1 | 9992000000125 |          2 |        nan |                 3 |          2 |        nan |                 3 |        0 |          |               4 |          0 |        nan |                 1 |                    1 |               1 |           4 |           4 |           0 |           0 |
| soll   | catalog-9992000000133 |            9 |            7 |             7 |                   7 |                7 |                       7 |                   7 |                      0 |                   0 |                    0 |                        0 |                   9 |                              7 |                   0.194444  |                         0.777778 |                     0.777778 |                1.55556  |                    1 |                    1 |                    4 |                           0 |                   0       |                     2 |                       5 |                  0.777778 |                 0.777778 |                         4 |                        1 |                    9 |                      1 | 9992000000133 |          2 |        nan |                 3 |          2 |        nan |                 2 |        0 |          |               3 |          0 |          1 |                 1 |                    0 |               1 |           3 |           3 |           1 |           0 |
| soll   | catalog-9992000000159 |           10 |            8 |             8 |                   8 |                8 |                       8 |                   8 |                      0 |                   0 |                    0 |                        0 |                  10 |                              8 |                   0.177778  |                         0.8      |                     0.8      |                1.6      |                    1 |                    1 |                    5 |                           0 |                   0       |                     2 |                       6 |                  0.8      |                 0.8      |                         5 |                        1 |                   10 |                      1 | 9992000000159 |          3 |          3 |                 3 |          2 |          2 |                 2 |        0 |          |               4 |          0 |          1 |                 1 |                    0 |               1 |           3 |           4 |           1 |           0 |
| soll   | catalog-9992000000167 |           13 |            5 |             5 |                   5 |                5 |                       5 |                   5 |                      0 |                   0 |                    0 |                        0 |                  13 |                              5 |                   0.0641026 |                         0.384615 |                     0.384615 |                0.769231 |                    0 |                    1 |                    4 |                           6 |                  46.1538  |                     8 |                       5 |                  0.384615 |                 0.384615 |                         4 |                        1 |                   13 |                      1 | 9992000000167 |          3 |        nan |                 4 |          3 |        nan |                 3 |        0 |          |               5 |          0 |          1 |                 1 |                    5 |               2 |           2 |           2 |           1 |           0 |
| soll   | catalog-9992000000175 |           10 |            8 |             8 |                   8 |                8 |                       8 |                   8 |                      0 |                   0 |                    0 |                        0 |                  10 |                              8 |                   0.177778  |                         0.8      |                     0.8      |                1.6      |                    1 |                    1 |                    5 |                           0 |                   0       |                     2 |                       6 |                  0.8      |                 0.8      |                         5 |                        1 |                   10 |                      1 | 9992000000175 |          3 |          3 |                 3 |          2 |          2 |                 2 |        0 |          |               4 |          0 |          1 |                 1 |                    0 |               0 |           3 |           4 |           1 |           0 |
| soll   | catalog-9992000000183 |           12 |            9 |             9 |                   9 |                9 |                       9 |                   9 |                      0 |                   0 |                    0 |                        0 |                  12 |                              9 |                   0.136364  |                         0.75     |                     0.75     |                1.5      |                    1 |                    1 |                    4 |                           0 |                   0       |                     3 |                       5 |                  0.75     |                 0.75     |                         4 |                        1 |                   12 |                      1 | 9992000000183 |          3 |          4 |                 4 |          3 |          3 |                 3 |        0 |          |               4 |          0 |          1 |                 1 |                    0 |               1 |           4 |           4 |           1 |           0 |

### Größenverteilungen (Nodes/Edges, raw)

Tabellen-Export: `analysis/descriptive/tables/size_distributions.csv`

| kind   |   graphs |   nodes_min |   nodes_med |   nodes_max |   edges_min |   edges_med |   edges_max |
|:-------|---------:|------------:|------------:|------------:|------------:|------------:|------------:|
| ist    |     2521 |           2 |           2 |           8 |           1 |           1 |           9 |
| soll   |       15 |           3 |           9 |          13 |           0 |           5 |           9 |

Plots gespeichert unter `plots/` (siehe Dateien).

### Connectedness / Komponenten (undirected; valide Kanten)

Tabellen-Export: `analysis/descriptive/tables/connectivity_summary.csv`

| kind   |   graphs |   cc_count_mean |   cc_count_max |   largest_cc_mean |   largest_cc_min |
|:-------|---------:|----------------:|---------------:|------------------:|-----------------:|
| ist    |     2521 |           1     |              1 |             2.15  |                2 |
| soll   |       15 |           3.267 |              9 |             4.667 |                1 |

### Dichte / E/N / Degree / Isolierte Knoten (undirected; valide Kanten)

Tabellen-Export: `analysis/descriptive/tables/struct_metrics_summary.csv`

| kind   |   graphs |   density_mean |   density_med |   E_per_N_mean |   avg_deg_mean |   isolated_pct_mean |
|:-------|---------:|---------------:|--------------:|---------------:|---------------:|--------------------:|
| ist    |     2521 |       0.961373 |      1        |       0.520101 |        1.0402  |              0      |
| soll   |       15 |       0.19842  |      0.177778 |       0.612962 |        1.22592 |             21.4608 |

### Degree-Statistiken pro Knotentyp (undirected; unique neighbors)

Tabellen-Export: `analysis/descriptive/tables/degree_stats_ist.csv`

| type   |   count |   deg_mean |   deg_min |   deg_max |
|:-------|--------:|-----------:|----------:|----------:|
| MaLo   |    2761 |      1.037 |         1 |         3 |
| MeLo   |    2624 |      1.109 |         1 |         5 |
| TR     |      36 |      1.306 |         1 |         2 |

Tabellen-Export: `analysis/descriptive/tables/degree_stats_soll.csv`

| type   |   count |   deg_mean |   deg_min |   deg_max |
|:-------|--------:|-----------:|----------:|----------:|
| MaLo   |      38 |      0.816 |         0 |         2 |
| MeLo   |      28 |      2.786 |         0 |         6 |
| NeLo   |      15 |      0.8   |         0 |         1 |
| TR     |      46 |      0.761 |         0 |         1 |

### Degree-Statistiken pro Knotentyp (directed; in/out, unique neighbors)

Tabellen-Export: `analysis/descriptive/tables/degree_stats_directed_ist.csv`

| type   |   count |   out_mean |   out_max |   in_mean |   in_max |
|:-------|--------:|-----------:|----------:|----------:|---------:|
| MaLo   |    2761 |      0     |         0 |     1.037 |        3 |
| MeLo   |    2624 |      1.109 |         5 |     0     |        0 |
| TR     |      36 |      0     |         0 |     1.306 |        2 |

Tabellen-Export: `analysis/descriptive/tables/degree_stats_directed_soll.csv`

| type   |   count |   out_mean |   out_max |   in_mean |   in_max |
|:-------|--------:|-----------:|----------:|----------:|---------:|
| MaLo   |      38 |      0     |         0 |     0.816 |        2 |
| MeLo   |      28 |      2.786 |         6 |     0     |        0 |
| NeLo   |      15 |      0     |         0 |     0.8   |        1 |
| TR     |      46 |      0     |         0 |     0.761 |        1 |

### Degree-Verteilungen (Plots; undirected, unique neighbors)

Degree-Plots gespeichert unter `plots/` (ist_degree_*.png, soll_degree_*.png).

## Graph-Signaturen (Struktur-Fingerprints)

- Ist: 2521 Graphen, 15 einzigartige Signaturen
- Soll: 15 Graphen, 13 einzigartige Signaturen
- Signaturen basieren auf Node-Typ-Counts, Edge-Rel-Counts, CC-Anzahl und Degree-Min/Median/Max.

### Top Signaturen (Ist)

Tabellen-Export: `analysis/descriptive/tables/signatures_top_ist.csv`

| kind   | signature_id   |   count | pct    | example_graph_id                                                                                                                                                                                        | signature                                                                  |
|:-------|:---------------|--------:|:-------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------|
| ist    | 6efad90624     |    2293 | 90.96% | comp:['50414974078']|['DE0071947684600000000000000000341']                                                                                                                                              | NT:MaLo=1,MeLo=1,TR=0,NeLo=0|ET:MEMA=1,METR=0,MENE=0,MEME=0|CC:1|DEG:1,1,1 |
| ist    | 623cabe978     |     106 | 4.20%  | comp:['50414794286', '50415648028']|['DE0071947684600000000000000000738']                                                                                                                               | NT:MaLo=2,MeLo=1,TR=0,NeLo=0|ET:MEMA=2,METR=0,MENE=0,MEME=0|CC:1|DEG:1,1,2 |
| ist    | afbaf8ef4b     |      80 | 3.17%  | comp:['50414863156', '50415025416']|['DE0071947684600000000000000001508', 'DE0071947684600000000000000002800']                                                                                          | NT:MaLo=2,MeLo=2,TR=0,NeLo=0|ET:MEMA=3,METR=0,MENE=0,MEME=0|CC:1|DEG:1,2,2 |
| ist    | 2f374262a1     |      10 | 0.40%  | comp:['50414941895', '50414948312', '50415030134']|['DE0071947684600000000000000000031', 'DE0071947684600000000000000002652']                                                                           | NT:MaLo=3,MeLo=2,TR=0,NeLo=0|ET:MEMA=4,METR=0,MENE=0,MEME=0|CC:1|DEG:1,2,2 |
| ist    | d2ad259fa3     |       9 | 0.36%  | bundle:100002161955                                                                                                                                                                                     | NT:MaLo=1,MeLo=1,TR=1,NeLo=0|ET:MEMA=1,METR=1,MENE=0,MEME=0|CC:1|DEG:1,1,2 |
| ist    | 3d1cbbb0e7     |       9 | 0.36%  | bundle:100002616793                                                                                                                                                                                     | NT:MaLo=2,MeLo=1,TR=1,NeLo=0|ET:MEMA=2,METR=1,MENE=0,MEME=0|CC:1|DEG:1,1,3 |
| ist    | 05fcbe5520     |       6 | 0.24%  | bundle:100003045702                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=1,NeLo=0|ET:MEMA=4,METR=2,MENE=0,MEME=0|CC:1|DEG:1,2,3 |
| ist    | 8fb147d1d4     |       1 | 0.04%  | comp:['50414918844', '50415021117', '50415148002']|['DE0003967684600000000000053018206', 'DE0003967684600000000000053018207', 'DE0071947684600000000000000002172', 'DE0071947684600000000000000002747'] | NT:MaLo=3,MeLo=4,TR=0,NeLo=0|ET:MEMA=6,METR=0,MENE=0,MEME=0|CC:1|DEG:1,2,3 |
| ist    | a6332cf2ce     |       1 | 0.04%  | comp:['50415031497', '50415263470', '50415263636']|['DE0071947684600000000000000000022']                                                                                                                | NT:MaLo=3,MeLo=1,TR=0,NeLo=0|ET:MEMA=3,METR=0,MENE=0,MEME=0|CC:1|DEG:1,1,3 |
| ist    | 075026649f     |       1 | 0.04%  | bundle:100002149765                                                                                                                                                                                     | NT:MaLo=1,MeLo=1,TR=2,NeLo=0|ET:MEMA=1,METR=2,MENE=0,MEME=0|CC:1|DEG:1,1,3 |
| ist    | 9725f560e7     |       1 | 0.04%  | bundle:100003262307                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=2,NeLo=0|ET:MEMA=4,METR=3,MENE=0,MEME=0|CC:1|DEG:1,2,4 |
| ist    | 03bab450ba     |       1 | 0.04%  | bundle:100003266124                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=3,NeLo=0|ET:MEMA=4,METR=5,MENE=0,MEME=0|CC:1|DEG:1,2,5 |
| ist    | 8ab8d4cad9     |       1 | 0.04%  | bundle:100003262313                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=1,NeLo=0|ET:MEMA=4,METR=1,MENE=0,MEME=0|CC:1|DEG:1,1,3 |
| ist    | c0dc24afe2     |       1 | 0.04%  | bundle:100002627035                                                                                                                                                                                     | NT:MaLo=2,MeLo=1,TR=2,NeLo=0|ET:MEMA=2,METR=2,MENE=0,MEME=0|CC:1|DEG:1,1,4 |
| ist    | c508c43a0c     |       1 | 0.04%  | bundle:100003120593                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=2,NeLo=0|ET:MEMA=4,METR=4,MENE=0,MEME=0|CC:1|DEG:1,2,4 |

### Top Signaturen (Soll)

Tabellen-Export: `analysis/descriptive/tables/signatures_top_soll.csv`

| kind   | signature_id   |   count | pct    | example_graph_id      | signature                                                                  |
|:-------|:---------------|--------:|:-------|:----------------------|:---------------------------------------------------------------------------|
| soll   | 8239ada13d     |       2 | 13.33% | catalog-9992000000026 | NT:MaLo=1,MeLo=1,TR=1,NeLo=1|ET:MEMA=1,METR=1,MENE=1,MEME=0|CC:1|DEG:1,1,3 |
| soll   | 5004182594     |       2 | 13.33% | catalog-9992000000159 | NT:MaLo=3,MeLo=2,TR=4,NeLo=1|ET:MEMA=3,METR=4,MENE=1,MEME=0|CC:2|DEG:1,1,5 |
| soll   | 81e8cbfe21     |       1 | 6.67%  | catalog-9992000000018 | NT:MaLo=1,MeLo=0,TR=1,NeLo=1|ET:MEMA=0,METR=0,MENE=0,MEME=0|CC:3|DEG:0,0,0 |
| soll   | 02a54f62f5     |       1 | 6.67%  | catalog-9992000000034 | NT:MaLo=2,MeLo=1,TR=2,NeLo=1|ET:MEMA=2,METR=2,MENE=1,MEME=0|CC:1|DEG:1,1,5 |
| soll   | 66705fce9a     |       1 | 6.67%  | catalog-9992000000042 | NT:MaLo=2,MeLo=1,TR=3,NeLo=1|ET:MEMA=2,METR=3,MENE=1,MEME=0|CC:1|DEG:1,1,6 |
| soll   | 964f57642f     |       1 | 6.67%  | catalog-9992000000068 | NT:MaLo=2,MeLo=2,TR=2,NeLo=1|ET:MEMA=2,METR=2,MENE=1,MEME=0|CC:2|DEG:1,1,3 |
| soll   | c165773f7b     |       1 | 6.67%  | catalog-9992000000076 | NT:MaLo=3,MeLo=2,TR=4,NeLo=1|ET:MEMA=3,METR=4,MENE=1,MEME=0|CC:2|DEG:1,1,6 |
| soll   | 7e0219bbd1     |       1 | 6.67%  | catalog-9992000000084 | NT:MaLo=4,MeLo=3,TR=4,NeLo=1|ET:MEMA=1,METR=1,MENE=1,MEME=0|CC:9|DEG:0,0,3 |
| soll   | b4c55be275     |       1 | 6.67%  | catalog-9992000000117 | NT:MaLo=2,MeLo=2,TR=4,NeLo=1|ET:MEMA=0,METR=0,MENE=0,MEME=0|CC:9|DEG:0,0,0 |
| soll   | 9c71c92a55     |       1 | 6.67%  | catalog-9992000000125 | NT:MaLo=3,MeLo=3,TR=4,NeLo=1|ET:MEMA=4,METR=4,MENE=0,MEME=0|CC:3|DEG:0,1,4 |
| soll   | 2a4628d258     |       1 | 6.67%  | catalog-9992000000133 | NT:MaLo=3,MeLo=2,TR=3,NeLo=1|ET:MEMA=3,METR=3,MENE=1,MEME=0|CC:2|DEG:1,1,4 |
| soll   | 83bdb66257     |       1 | 6.67%  | catalog-9992000000167 | NT:MaLo=4,MeLo=3,TR=5,NeLo=1|ET:MEMA=2,METR=2,MENE=1,MEME=0|CC:8|DEG:0,1,4 |
| soll   | 115b288ed9     |       1 | 6.67%  | catalog-9992000000183 | NT:MaLo=4,MeLo=3,TR=4,NeLo=1|ET:MEMA=4,METR=4,MENE=1,MEME=0|CC:3|DEG:1,1,4 |

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
| ist    | TR     | direction      |         0 |      36 | 0.0%          |
| ist    | TR     | tr_direction   |         0 |      36 | 0.0%          |

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

### Topologie-Checkliste: erwartete vs vorhandene Relationen

Tabellen-Export: `analysis/descriptive/tables/template_topology_checklist.csv`

| kind   | graph_id              |      lbs_code |   MaLo |   MeLo |   TR |   NeLo |   edge_MEMA |   edge_METR |   edge_MENE |   edge_MEME | rel_types_present   | expects_MEMA   | missing_MEMA   | expects_METR   | missing_METR   | expects_MENE   | missing_MENE   | multi_MeLo   | has_MEME   |   attachment_rules_n |   edges_missing_refs |   edges_missing_srcdst |   edges_missing_rel |
|:-------|:----------------------|--------------:|-------:|-------:|-----:|-------:|------------:|------------:|------------:|------------:|:--------------------|:---------------|:---------------|:---------------|:---------------|:---------------|:---------------|:-------------|:-----------|---------------------:|---------------------:|-----------------------:|--------------------:|
| soll   | catalog-9992000000117 | 9992000000117 |      2 |      2 |    4 |      1 |           0 |           0 |           0 |           0 |                     | True           | True           | True           | True           | True           | True           | True         | False      |                    7 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000125 | 9992000000125 |      3 |      3 |    4 |      1 |           4 |           4 |           0 |           0 | MEMA,METR           | True           | False          | True           | False          | True           | True           | True         | False      |                    1 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000018 | 9992000000018 |      1 |      0 |    1 |      1 |           0 |           0 |           0 |           0 |                     | False          | False          | False          | False          | False          | False          | False        | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000026 | 9992000000026 |      1 |      1 |    1 |      1 |           1 |           1 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | False        | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000034 | 9992000000034 |      2 |      1 |    2 |      1 |           2 |           2 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | False        | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000042 | 9992000000042 |      2 |      1 |    3 |      1 |           2 |           3 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | False        | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000068 | 9992000000068 |      2 |      2 |    2 |      1 |           2 |           2 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | True         | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000076 | 9992000000076 |      3 |      2 |    4 |      1 |           3 |           4 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | True         | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000084 | 9992000000084 |      4 |      3 |    4 |      1 |           1 |           1 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | True         | False      |                    6 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000109 | 9992000000109 |      1 |      1 |    1 |      1 |           1 |           1 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | False        | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000133 | 9992000000133 |      3 |      2 |    3 |      1 |           3 |           3 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | True         | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000159 | 9992000000159 |      3 |      2 |    4 |      1 |           3 |           4 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | True         | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000167 | 9992000000167 |      4 |      3 |    5 |      1 |           2 |           2 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | True         | False      |                    5 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000175 | 9992000000175 |      3 |      2 |    4 |      1 |           3 |           4 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | True         | False      |                    0 |                    0 |                      0 |                   0 |
| soll   | catalog-9992000000183 | 9992000000183 |      4 |      3 |    4 |      1 |           4 |           4 |           1 |           0 | MEMA,MENE,METR      | True           | False          | True           | False          | True           | False          | True         | False      |                    0 |                    0 |                      0 |                   0 |

- Templates: total=15
- Templates: missing_MEMA=1 (nur wenn MeLo+MaLo vorhanden)
- Templates: missing_METR=1 (nur wenn MeLo+TR vorhanden)
- Templates: missing_MENE=2 (nur wenn MeLo+NeLo vorhanden)
- Templates: graphs_with_invalid_edge_refs=0

### Topologie-Checkliste: Ist-Anomalien (Sample)

- Ist: total=2521
- Ist: missing_MEMA=0 (nur wenn MeLo+MaLo vorhanden)
- Ist: missing_METR=0 (nur wenn MeLo+TR vorhanden)
- Ist: missing_MENE=0 (nur wenn MeLo+NeLo vorhanden)
- Ist: graphs_with_invalid_edge_refs=0

Tabellen-Export: `analysis/descriptive/tables/ist_topology_anomalies.csv`

_Keine Daten._

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

| kind   | metric                   | count              |
|:-------|:-------------------------|:-------------------|
| ist    | nodes_total              | 5421               |
| ist    | edges_total(valid typed) | 2911               |
| ist    | dir_unknown_rate         | 5.0% (140/2797)    |
| ist    | melo_fn_unknown_rate     | 100.0% (2624/2624) |
| ist    | volt_unknown_rate        | 2.5% (66/2624)     |
| soll   | nodes_total              | 127                |
| soll   | edges_total(valid typed) | 78                 |
| soll   | dir_unknown_rate         | 10.7% (9/84)       |
| soll   | melo_fn_unknown_rate     | 0.0% (0/28)        |
| soll   | volt_unknown_rate        | 100.0% (28/28)     |

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

| key                           |   count |
|:------------------------------|--------:|
| source                        |      36 |
| malo_ref                      |      36 |
| anlage                        |      36 |
| bisdatum                      |      36 |
| abdatum                       |      36 |
| art_der_technischen_ressource |      36 |
| verbrauchsart                 |      36 |
| wrmenutzung                   |      36 |
| ref_zur_tr                    |      36 |

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
- Topologie-Check: 1 Templates fehlen MEMA obwohl MeLo+MaLo vorhanden, 1 Templates fehlen METR obwohl MeLo+TR vorhanden. Prüfen: Absicht (attachment_rules/Unklarheit) vs. Mapping-Lücke.
