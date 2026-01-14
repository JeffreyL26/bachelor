# DESCRIPTIVE ANALYSIS

## Inputs

- Instance-Graphs: `data\ist_graphs_all.jsonl`
- Graph-Templates: `data\lbs_soll_graphs.jsonl`
- Graphs covered: _all_
- Plots: created (matplotlib installed)
- pandas: installed
- graph_pipeline Import: ok
- Structural metrics: only _valid_ edges (src/dst reference existing nodes).

Loaded: **2521** Instance-Graphs, **15** Graph-Templates.

## Quality-Check

- Issues total: 0
- ERRORS: 0
- WARNINGS: 0

Errors would include missing node/edge keys, nodes without ID, duplicate IDs, edges referencing non-existent nodes.

Warnings would include unknown node types/edge relations or wrong node counts.

### Most common issues

Full table stored in: `analysis/descriptive/tables/issues_top.csv`

_No data._

## Fields present in Instance-Graphs

Metadata fields that are consistently available across Instance-Graphs.

### Instances: graph_attrs Keys (How often per Graph?)

graph_attr is a metadata dictionary.

Full table stored in: `analysis/descriptive/tables/ist_graph_attrs_keys.csv`

| KEY              |   COUNT | %      |
|:-----------------|--------:|:-------|
| malo_count       |    2521 | 100.0% |
| melo_count       |    2521 | 100.0% |
| tr_count         |    2521 | 100.0% |
| nelo_count       |    2521 | 100.0% |
| edge_type_counts |    2521 | 100.0% |
| dataset          |    2521 | 100.0% |
| bundle_id        |      48 | 1.9%   |

### Templates: graph_attrs Keys (How often per Graph?)

graph_attr is a metadata dictionary.

Full table stored in: `analysis/descriptive/tables/soll_graph_attrs_keys.csv`

| KEY                     |   COUNT | %      |
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

### Instances: Node-Attribute Keys (How often per Node?)

Full table stored in: `analysis/descriptive/tables/ist_node_attr_keys.csv`

| KEY       |   COUNT |
|:----------|--------:|
| direction |    5421 |
| source    |      36 |

### Templates: Node-Attribute Keys (How often per Node?)

Full table stored in: `analysis/descriptive/tables/soll_node_attr_keys.csv`

| KEY         |   COUNT |
|:------------|--------:|
| level       |     127 |
| object_code |     127 |
| min_occurs  |     127 |
| max_occurs  |     127 |
| flexibility |     127 |
| optional    |     127 |
| direction   |     112 |
| function    |      28 |
| dynamic     |      28 |

## Structure: Node Types & Edge Types

### Node Types in Dataset (Full)

Full table stored in: `analysis/descriptive/tables/node_type_distribution.csv`

| TYPE   |   IN INSTANCES |   IN TEMPLATES |
|:-------|---------------:|---------------:|
| MaLo   |           2761 |             38 |
| MeLo   |           2624 |             28 |
| NeLo   |             12 |             15 |
| TR     |             36 |             46 |

### Edge Types in Dataset (Full)

Only valid edges are considered. Edges are considered valid, when they have a valid source and a valid destination.

Full table stored in: `analysis/descriptive/tables/edge_type_distribution.csv`

| RELATION   |   IN INSTANCES |   IN TEMPLATES |
|:-----------|---------------:|---------------:|
| MEMA       |           2864 |             31 |
| MENE       |             12 |             12 |
| METR       |             47 |             35 |

## Metrics per Graph

### Per-Graph-Table (Sample)

Full table is stored in `tables/`.

Full table stored in: `analysis/descriptive/tables/per_graph_ist.csv`

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

_Table has been shortened to 25 rows (originally 2521)._

Full table stored in: `analysis/descriptive/tables/per_graph_soll.csv`

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

### Dimensions (Nodes/Edges)

Full table stored in: `analysis/descriptive/tables/size_distributions.csv`

| kind   |   graphs |   nodes_min |   nodes_med |   nodes_max |   edges_min |   edges_med |   edges_max |
|:-------|---------:|------------:|------------:|------------:|------------:|------------:|------------:|
| ist    |     2521 |           2 |           2 |           9 |           1 |           1 |          10 |
| soll   |       15 |           3 |           9 |          13 |           0 |           5 |           9 |

Plots saved in `plots/` (siehe Dateien).

### Number and Size of Connected Components

This considers valid and undirected connections only to approximate message-parsing reachability.

Full table stored in: `analysis/descriptive/tables/connectivity_summary.csv`

| kind   |   graphs |   cc_count_mean |   cc_count_max |   largest_cc_mean |   largest_cc_min |
|:-------|---------:|----------------:|---------------:|------------------:|-----------------:|
| ist    |     2521 |           1     |              1 |             2.155 |                2 |
| soll   |       15 |           3.267 |              9 |             4.667 |                1 |

### Density, Edges per Node, Degree and Isolated Nodes

Insights on how structurally rich the given dataset is.

Density shows how many edges the graph has, compared to the maximum possible amount.

Edges per Node shows structural repetition.

Degree shows the number of nodes connected to a node.

Isolated nodes are nodes with degree 0.

Full table stored in: `analysis/descriptive/tables/struct_metrics_summary.csv`

| kind   |   graphs |   density_mean |   density_med |   E_per_N_mean |   avg_deg_mean |   isolated_pct_mean |
|:-------|---------:|---------------:|--------------:|---------------:|---------------:|--------------------:|
| ist    |     2521 |       0.960834 |      1        |       0.520334 |        1.04067 |              0      |
| soll   |       15 |       0.19842  |      0.177778 |       0.612962 |        1.22592 |             21.4608 |

### Degree Statistics per Node Type - Undirected

Full table stored in: `analysis/descriptive/tables/degree_stats_ist.csv`

| type   |   count |   deg_mean |   deg_min |   deg_max |
|:-------|--------:|-----------:|----------:|----------:|
| MaLo   |    2761 |      1.037 |         1 |         3 |
| MeLo   |    2624 |      1.114 |         1 |         5 |
| NeLo   |      12 |      1     |         1 |         1 |
| TR     |      36 |      1.306 |         1 |         2 |

Full table stored in: `analysis/descriptive/tables/degree_stats_soll.csv`

| type   |   count |   deg_mean |   deg_min |   deg_max |
|:-------|--------:|-----------:|----------:|----------:|
| MaLo   |      38 |      0.816 |         0 |         2 |
| MeLo   |      28 |      2.786 |         0 |         6 |
| NeLo   |      15 |      0.8   |         0 |         1 |
| TR     |      46 |      0.761 |         0 |         1 |

### Degree Statistics per Node Type - Directed

Full table stored in: `analysis/descriptive/tables/degree_stats_directed_ist.csv`

| type   |   count |   out_mean |   out_max |   in_mean |   in_max |
|:-------|--------:|-----------:|----------:|----------:|---------:|
| MaLo   |    2761 |      0     |         0 |     1.037 |        3 |
| MeLo   |    2624 |      1.114 |         5 |     0     |        0 |
| NeLo   |      12 |      0     |         0 |     1     |        1 |
| TR     |      36 |      0     |         0 |     1.306 |        2 |

Full table stored in: `analysis/descriptive/tables/degree_stats_directed_soll.csv`

| type   |   count |   out_mean |   out_max |   in_mean |   in_max |
|:-------|--------:|-----------:|----------:|----------:|---------:|
| MaLo   |      38 |      0     |         0 |     0.816 |        2 |
| MeLo   |      28 |      2.786 |         6 |     0     |        0 |
| NeLo   |      15 |      0     |         0 |     0.8   |        1 |
| TR     |      46 |      0     |         0 |     0.761 |        1 |

### Degree-Distribution (Plots; undirected, unique neighbors)

Degree-Plots saved in `plots/` (ist_degree_*.png, soll_degree_*.png).

## Graph Signatures - Structural Diversity in the Dataset

- Instances: 2521 Graphs, 19 unique signatures
- Templates: 15 Graphs, 13 unique signatures
- Signatures are based on node type counts, edge relation counts, number of connected components and degree-Min/Median/Max. 
Connected Components are a great indicator, whether the structure forms a single coherent graph or splits into disconnected subgraphs.

### Top Signatures (Instances)

Full table stored in: `analysis/descriptive/tables/signatures_top_ist.csv`

| kind   | signature_id   |   count | %      | example_graph_id                                                                                                                                                                                        | signature                                                                  |
|:-------|:---------------|--------:|:-------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------|
| ist    | 6efad90624     |    2293 | 90.96% | comp:['50414974078']|['DE0071947684600000000000000000341']                                                                                                                                              | NT:MaLo=1,MeLo=1,TR=0,NeLo=0|ET:MEMA=1,METR=0,MENE=0,MEME=0|CC:1|DEG:1,1,1 |
| ist    | 623cabe978     |     101 | 4.01%  | comp:['50414794286', '50415648028']|['DE0071947684600000000000000000738']                                                                                                                               | NT:MaLo=2,MeLo=1,TR=0,NeLo=0|ET:MEMA=2,METR=0,MENE=0,MEME=0|CC:1|DEG:1,1,2 |
| ist    | afbaf8ef4b     |      80 | 3.17%  | comp:['50414863156', '50415025416']|['DE0071947684600000000000000001508', 'DE0071947684600000000000000002800']                                                                                          | NT:MaLo=2,MeLo=2,TR=0,NeLo=0|ET:MEMA=3,METR=0,MENE=0,MEME=0|CC:1|DEG:1,2,2 |
| ist    | 2f374262a1     |       9 | 0.36%  | comp:['50414941895', '50414948312', '50415030134']|['DE0071947684600000000000000000031', 'DE0071947684600000000000000002652']                                                                           | NT:MaLo=3,MeLo=2,TR=0,NeLo=0|ET:MEMA=4,METR=0,MENE=0,MEME=0|CC:1|DEG:1,2,2 |
| ist    | d2ad259fa3     |       9 | 0.36%  | bundle:100002161955                                                                                                                                                                                     | NT:MaLo=1,MeLo=1,TR=1,NeLo=0|ET:MEMA=1,METR=1,MENE=0,MEME=0|CC:1|DEG:1,1,2 |
| ist    | 3d1cbbb0e7     |       6 | 0.24%  | bundle:100002616793                                                                                                                                                                                     | NT:MaLo=2,MeLo=1,TR=1,NeLo=0|ET:MEMA=2,METR=1,MENE=0,MEME=0|CC:1|DEG:1,1,3 |
| ist    | be0d0f937a     |       5 | 0.20%  | bundle:100003249169                                                                                                                                                                                     | NT:MaLo=2,MeLo=1,TR=0,NeLo=1|ET:MEMA=2,METR=0,MENE=1,MEME=0|CC:1|DEG:1,1,3 |
| ist    | 05fcbe5520     |       5 | 0.20%  | bundle:100002988406                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=1,NeLo=0|ET:MEMA=4,METR=2,MENE=0,MEME=0|CC:1|DEG:1,2,3 |
| ist    | 3b20757886     |       3 | 0.12%  | bundle:100002563743                                                                                                                                                                                     | NT:MaLo=2,MeLo=1,TR=1,NeLo=1|ET:MEMA=2,METR=1,MENE=1,MEME=0|CC:1|DEG:1,1,4 |
| ist    | 8fb147d1d4     |       1 | 0.04%  | comp:['50414918844', '50415021117', '50415148002']|['DE0003967684600000000000053018206', 'DE0003967684600000000000053018207', 'DE0071947684600000000000000002172', 'DE0071947684600000000000000002747'] | NT:MaLo=3,MeLo=4,TR=0,NeLo=0|ET:MEMA=6,METR=0,MENE=0,MEME=0|CC:1|DEG:1,2,3 |
| ist    | a6332cf2ce     |       1 | 0.04%  | comp:['50415031497', '50415263470', '50415263636']|['DE0071947684600000000000000000022']                                                                                                                | NT:MaLo=3,MeLo=1,TR=0,NeLo=0|ET:MEMA=3,METR=0,MENE=0,MEME=0|CC:1|DEG:1,1,3 |
| ist    | 075026649f     |       1 | 0.04%  | bundle:100002149765                                                                                                                                                                                     | NT:MaLo=1,MeLo=1,TR=2,NeLo=0|ET:MEMA=1,METR=2,MENE=0,MEME=0|CC:1|DEG:1,1,3 |
| ist    | 7e77b08f37     |       1 | 0.04%  | bundle:100003262307                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=2,NeLo=1|ET:MEMA=4,METR=3,MENE=1,MEME=0|CC:1|DEG:1,2,5 |
| ist    | 64ccf90809     |       1 | 0.04%  | bundle:100003266124                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=3,NeLo=1|ET:MEMA=4,METR=5,MENE=1,MEME=0|CC:1|DEG:1,2,5 |
| ist    | dca1595d75     |       1 | 0.04%  | bundle:100003045702                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=1,NeLo=1|ET:MEMA=4,METR=2,MENE=1,MEME=0|CC:1|DEG:1,2,4 |
| ist    | 8ab8d4cad9     |       1 | 0.04%  | bundle:100003262313                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=1,NeLo=0|ET:MEMA=4,METR=1,MENE=0,MEME=0|CC:1|DEG:1,1,3 |
| ist    | c0dc24afe2     |       1 | 0.04%  | bundle:100002627035                                                                                                                                                                                     | NT:MaLo=2,MeLo=1,TR=2,NeLo=0|ET:MEMA=2,METR=2,MENE=0,MEME=0|CC:1|DEG:1,1,4 |
| ist    | b0facc4164     |       1 | 0.04%  | bundle:100003124197                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=0,NeLo=1|ET:MEMA=4,METR=0,MENE=1,MEME=0|CC:1|DEG:1,1,3 |
| ist    | c508c43a0c     |       1 | 0.04%  | bundle:100003120593                                                                                                                                                                                     | NT:MaLo=3,MeLo=2,TR=2,NeLo=0|ET:MEMA=4,METR=4,MENE=0,MEME=0|CC:1|DEG:1,2,4 |

### Top Signatures (Templates)

Full table stored in: `analysis/descriptive/tables/signatures_top_soll.csv`

| kind   | signature_id   |   count | %      | example_graph_id      | signature                                                                  |
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

## Attributes & Value-Distribution

### Coverage: Main Attributes (Instances)

Full table stored in: `analysis/descriptive/tables/attr_coverage_ist.csv`

| kind   | type   | attr      |   present |   total | present %   |
|:-------|:-------|:----------|----------:|--------:|:------------|
| ist    | MaLo   | direction |      2761 |    2761 | 100.0%      |
| ist    | MeLo   | direction |      2624 |    2624 | 100.0%      |
| ist    | TR     | direction |        36 |      36 | 100.0%      |

### Coverage: Main Attributes (Templates)

Full table stored in: `analysis/descriptive/tables/attr_coverage_soll.csv`

| kind   | type   | attr      |   present |   total | present %   |
|:-------|:-------|:----------|----------:|--------:|:------------|
| soll   | MaLo   | direction |        38 |      38 | 100.0%      |
| soll   | MeLo   | direction |        28 |      28 | 100.0%      |
| soll   | TR     | direction |        46 |      46 | 100.0%      |

### Top Values: Direction (MaLo)

Full table stored in: `analysis/descriptive/tables/top_values_ist_MaLo_direction.csv`

| kind   | type   | attr      | value       |   count |
|:-------|:-------|:----------|:------------|--------:|
| ist    | MaLo   | direction | consumption |    2457 |
| ist    | MaLo   | direction | generation  |     304 |

Full table stored in: `analysis/descriptive/tables/top_values_soll_MaLo_direction.csv`

| kind   | type   | attr      | value       |   count |
|:-------|:-------|:----------|:------------|--------:|
| soll   | MaLo   | direction | consumption |      25 |
| soll   | MaLo   | direction | generation  |      13 |

### Top Values: Direction (MeLo)

Full table stored in: `analysis/descriptive/tables/top_values_ist_MeLo_direction.csv`

| kind   | type   | attr      | value       |   count |
|:-------|:-------|:----------|:------------|--------:|
| ist    | MeLo   | direction | consumption |    1597 |
| ist    | MeLo   | direction | both        |     922 |
| ist    | MeLo   | direction | generation  |     105 |

Full table stored in: `analysis/descriptive/tables/top_values_soll_MeLo_direction.csv`

| kind   | type   | attr      | value       |   count |
|:-------|:-------|:----------|:------------|--------:|
| soll   | MeLo   | direction | both        |      11 |
| soll   | MeLo   | direction | consumption |      10 |
| soll   | MeLo   | direction | generation  |       7 |

### Top Values: Direction (TR)

Full table stored in: `analysis/descriptive/tables/top_values_ist_TR_direction.csv`

| kind   | type   | attr      | value       |   count |
|:-------|:-------|:----------|:------------|--------:|
| ist    | TR     | direction | consumption |      25 |
| ist    | TR     | direction | both        |      11 |

Full table stored in: `analysis/descriptive/tables/top_values_soll_TR_direction.csv`

| kind   | type   | attr      | value       |   count |
|:-------|:-------|:----------|:------------|--------:|
| soll   | TR     | direction | consumption |      21 |
| soll   | TR     | direction | generation  |      13 |
| soll   | TR     | direction | both        |      12 |

## Template-specific Analysis (LBS-Scheme)

### Topology Checklist: Expected vs. Existing Relations

Full table stored in: `analysis/descriptive/tables/template_topology_checklist.csv`

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
- Templates: missing_MEMA=1 (only when MeLo+MaLo exists)
- Templates: missing_METR=1 (only when MeLo+TR exists)
- Templates: missing_MENE=2 (only when MeLo+NeLo exists)
- Templates: graphs_with_invalid_edge_refs=0

### Topology Checklist: Instance-Anomalies (Sample)

- Ist: total=2521
- Ist: missing_MEMA=0 (only when MeLo+MaLo exists)
- Ist: missing_METR=0 (only when MeLo+TR exists)
- Ist: missing_MENE=0 (only when MeLo+NeLo exists)
- Ist: graphs_with_invalid_edge_refs=0

Full table stored in: `analysis/descriptive/tables/ist_topology_anomalies.csv`

_No data._

Full table stored in: `analysis/descriptive/tables/template_optionality_summary.csv`

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

## Instances vs. Templates: Flexibility Compliance

Shows how many Instance-Graphs comply to the min/max-bounds given by the governing body BDEW (per Template).

Full table stored in: `analysis/descriptive/tables/template_coverage.csv`

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

## Feature Encoding Alignment (graph_pipeline.py)

### Unknown-rates for used features

Full table stored in: `analysis/descriptive/tables/encoder_unknown_rates.csv`

| kind   | metric                         | count         |
|:-------|:-------------------------------|:--------------|
| ist    | nodes_total                    | 5433          |
| ist    | edges_total(valid typed)       | 2923          |
| ist    | node_type_unknown_rate         | 0.0% (0/5433) |
| ist    | dir_unknown_rate(MaLo/MeLo/TR) | 0.0% (0/5421) |
| ist    | edge_rel_unknown_rate          | 0.0% (0/2923) |
| soll   | nodes_total                    | 127           |
| soll   | edges_total(valid typed)       | 78            |
| soll   | node_type_unknown_rate         | 0.0% (0/127)  |
| soll   | dir_unknown_rate(MaLo/MeLo/TR) | 0.0% (0/112)  |
| soll   | edge_rel_unknown_rate          | 0.0% (0/78)   |

### Which Features are Considered for DGMC?

The feature encoder (graph_pipeline.py) reduces node and edge information to compact one-hot blocks. This section verifies which raw fields can actually affect the model input, and which are currently unused.

- **Node type**: `node['type']` → one-hot over {MaLo, MeLo, TR, NeLo}
- **Direction** (MaLo/MeLo/TR): derived from `attrs['direction']` with fallbacks (TR: `tr_direction`, `tr_type_code` / `art_der_technischen_ressource`; MaLo/MeLo: `direction_hint`) → {consumption, generation, both, unknown}
- **Edge relation**: `edge['rel']` (or legacy keys like `type`/`edge_type`) → one-hot over {MEMA, METR, MENE, MEME, unknown}
- All other node/graph attributes are currently **not** encoded as ML features.

### Which Node-Attributes are within the Dataset, however unused in Encoding?

Instance-Graphs (Top unused Keys):

Full table stored in: `analysis/descriptive/tables/ist_unused_node_attrs.csv`

| key    |   count |
|:-------|--------:|
| source |      36 |

Template-Graphs (Top unused Keys):

Full table stored in: `analysis/descriptive/tables/soll_unused_node_attrs.csv`

| key         |   count |
|:------------|--------:|
| level       |     127 |
| object_code |     127 |
| min_occurs  |     127 |
| max_occurs  |     127 |
| flexibility |     127 |
| optional    |     127 |
| function    |      28 |
| dynamic     |      28 |

## Observations (Automatic) and Possible Causes

- Topology-Check: 1 Template(s) don't have MEMA although MeLo+MaLo exists, 1 Template(s) miss METR although MeLo+TR exists. Check if: Model choice (attachment_rules) vs. mapping-gap.
