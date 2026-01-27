# Automated Matching Analysis of Metering Constructs using GNN (DGMC) - PROJECT README

Install correct librariy versions to run:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install pykeops
pip install "git+https://github.com/rusty1s/deep-graph-matching-consensus.git"
```


- data/lbs_templates/
  JSON LBS template files (9992 … - Final.json).

- data/training_data/
  CSV exports for MaLo/MeLo/relations/TR/NeLo/label mappings

- Generated artifacts (created by scripts):
  - data/lbs_soll_graphs.jsonl – template graphs
  - data/ist_graphs_all.jsonl – instance graphs
  - data/synthetic_training_pairs.jsonl – synthetic DGMC training pairs
  - data/synthetic_training_pairs_control.jsonl – permutation-only control pairs
  - data/dgmc_partial.pt – DGMC checkpoint
  - data/dgmc_perm.pt – DGMC checkpoint trained on synthetic_training_pairs_control.py
  - runs/*.jsonl – matching outputs
  - analysis/descriptive/ – descriptive analysis report (markdown + CSV + plots)

---

## Code Files

### baseline_constraints.py
Rule-based baseline that scores an instance graph against each template using hard bounds and constraint penalties, returning top-k candidates. Evaluates hits@k using subset.
**Run (CLI):**
```bash
python baseline_constraints.py   --ist_path <path>   --templates_path <path>   --out_path <path>   --top_k <int>   --max_ist <int>   --max_templates <int>   --bndl2mc_path <path | ''>   --lbs_json_dir <path>
```
**Command Line Parameters:**
- --ist_path <path>
- --templates_path <path>
- --out_path <path>
- --top_k <int>
- --max_ist <int> (optional limit)
- --max_templates <int> (optional limit)
- --bndl2mc_path <path | ''> (set to empty string to disable evaluation)
- --lbs_json_dir <dir>

### baseline_graph_similarity.py
Baseline matcher using exact Graph Edit Distance (GED) and converts GED into a similarity score in [0, 1]. Produces top-k template candidates and evaluation on the labeled subset.
**Run (CLI):**
```bash
python baseline_graph_similarity.py   --ist_path <path>   --templates_path <path>   --out_path <path>   --top_k <int>   --max_ist <int>   --max_templates <int>   --directed   --timeout_s <float>   --lbs_json_dir <path>   --bndl2mc_path <path | ''>   --type_mismatch_cost <float>   --dir_mismatch_cost <float>   --dir_unknown_cost <float>
```
**CLI options:**
- --ist_path <path>
- --templates_path <path>
- --out_path <path>
- --top_k <int>
- --max_ist <int> (optional)
- --max_templates <int> (optional)
- --directed (boolean, default is undirected)
- --timeout_s <float> (optional per pair computation timeout)
- --lbs_json_dir <dir> (used to enrich template directions, where template directions might be zero, not the case as for final templates)
- --bndl2mc_path <path | ''> (set to empty string to disable evaluation)
- --type_mismatch_cost <float>
- --dir_mismatch_cost <float>
- --dir_unknown_cost <float>

### dgmc_dataset.py
Dataset + batching utilities to load synthetic JSONL graph pairs and produce PyG Batch objects plus correspondence tensors y. Main is merely a check if everything functions.
**Run:**
```bash
python dgmc_dataset.py
```

### dgmc_template_training.py
Trains DGMC on synthetic pairs created from templates, using an GINE encoder and  DGMC refinement (num_steps). Saves the best checkpoint in epochs as a model (by validation loss).
**Run (CLI):**
```bash
python dgmc_template_training.py   --pairs_path <path>   --epochs <int>   --batch_size <int>   --lr <float>   --weight_decay <float>   --seed <int>   --num_steps <int>   --k <-1 | int>=1>   --detach   --hidden_channels <int>   --out_channels <int>   --num_layers <int>   --dropout <float>   --train_frac <float in [0,1]>   --num_workers <int>   --save_path <path>
```
**CLI options:**
- --pairs_path <path>
- --epochs <int>
- --batch_size <int>
- --lr <float>
- --weight_decay <float>
- --seed <int>
- --num_steps <int> (DGMC refinement steps)
- --k <-1 | int>=1> (-1 = dense all-vs-all node matching, >=1 = sparse top-k, meaning DGMC oly keeps the top-k target candidates per source node, limits search space)
- --detach (boolean, default false. If true, DGMC wouldn't backpropagate)
- --hidden_channels <int>
- --out_channels <int>
- --num_layers <int>
- --dropout <float>
- --train_frac <float> (train split fraction [0,1])
- --num_workers <int>
- --save_path <path>

### dgmc_matcher.py
Matches each instance graph against all template graphs using a trained DGMC checkpoint, returns top-k templates per instance graph, and produces node-level correspondences for the best template. Evaluation on labeled subset.
**Run (CLI):**
```bash
python dgmc_matcher.py   --ist_path <path>   --templates_path <path>   --checkpoint <path>   --out_path <path>   --device <cpu | cuda | cuda:0 | ...>   --undirected   --directed   --top_templates <int>   --top_matches <int>   --score_mode <mean_rowmax | mean_rowmax_symmetric>   --max_ist <int | 0>   --max_templates <int | 0>   --override_num_steps <int>   --override_k <int>   --override_detach   --no_override_detach   --bndl2mc_path <path | ''>
```
**CLI options:**
- --ist_path <path>
- --templates_path <path>
- --checkpoint <path> (DGMC checkpoint)
- --out_path <path>
- --device <cpu | cuda | cuda:N | ...>
- --undirected (boolean, use undirected message passing, default)
- --directed (boolean, overrides --undirected)
- --top_templates <int> (graph-level top-k templates to return)
- --top_matches <int> (node-level top-k candidates per template node)
- --score_mode <mean_rowmax | mean_rowmax_symmetric> (how to rank templates from the DGMC similarity matrix)
- --max_ist <int | 0> (0 = all)
- --max_templates <int | 0> (0 = all)
- --override_num_steps <int> (-999 to keep checkpoint value, otherwise override)
- --override_k <int> (-999 to keep checkpoint value, otherwise override)
- --override_detach (boolean, force detach=True)
- --no_override_detach (boolean, force detach=False)
- --bndl2mc_path <path | ''> (set to empty string to disable evaluation)

### graph_converter.py
Loads CSV exports (SDF dataset + 2nd dataset with TR/NeLo) and converts them into instance graphs in the shared JSON graph format. Writes the combined output to data/ist_graphs_all.jsonl.
**Run:**
```bash
python graph_converter.py
```

### graph_pipeline.py
Converts JSON graphs into PyG Data objects with one-hot node features (type + direction) and one-hot edge features (relation type). Also provides helpers to load JSONL files as PyG objects for training/inference.
Not intended to be executed directly

### graph_templates.py
Converts raw LBS template JSON files into template graphs in JSONL format. Writes data/lbs_soll_graphs.jsonl, which is then used by baselines, pair generation, training, and matching.
**Run:**
```bash
python graph_templates.py
```

### pipeline_tester.py
Quick check that graph_pipeline.py can load and convert both instance and template JSONL files to PyG Data objects (prints tensor shapes metadata). 
**Run:**
```bash
python pipeline_tester.py
```

### synthetic_pair_builder.py
Generates synthetic training pairs from template graphs by applying controlled perturbations (node/edge/attribute dropout, extra nodes, attachment rules) and produces partial correspondences y for DGMC. Writes data/synthetic_training_pairs.jsonl.
**Run (no CLI, parameters in main):**
```bash
python synthetic_pair_builder.py
```

### synthetic_pair_builder_control.py
Builds a permutation-only  dataset where graph B is a node-permuted copy of graph A, producing a full 1:1 y. Writes data/synthetic_training_pairs_control.jsonl.
**Run (no CLI, parameters in main):**
```bash
python synthetic_pair_builder_control.py
```

### deskriptive_analyse.py
Creates a descriptive report comparing instance graphs vs template graphs (counts, distributions, hashes, optional plots) and writes a markdown summary plus CSV tables (and optionally figures) to an output directory.
**Run (CLI):**
```bash
python deskriptive_analyse.py   --ist <path>   --soll <path>   --out_dir <path>   --max_graphs <int>   --no-plots
```
**CLI options:**
- --ist <path>
- --soll <path>
- --out_dir <dir>
- --max_graphs <int> (optional)
- --no-plots (boolean, disables plotting)

---

Data files (CSV)

SDF_MALO.csv
1st dataset export containing Marktlokation (MaLo) Attributes used by graph_converter.py to create MaLo nodes and derive direction hints. 

SDF_MELO.csv
1st dataset export containing Messlokation (MeLo) attributes used by graph_converter.py to create MeLo nodes and link them to MaLo/TR where possible.

SDF_POD_REL.csv
1st dataset export describing MaLo/MeLo relationships used by graph_converter.py to connect MaLo and MeLo.

SDF_METER.csv
1st dataset export describing information used by graph_converter.py to build edges.

MALO.csv
2nd dataset export for Marktlokation (MaLo) used by graph_converter.py to create MaLo nodes and derive direction hints.

MELO.csv
2nd dataset export for Messlokation (MeLo) used by graph_converter.py to create MeLo nodes and link them to MaLo/TR where possible.

PODREL.csv
2nd dataset export describing MaLo/MeLo relationships used by graph_converter.py to connect MaLo and MeLo.

METER.csv
2nd dataset export describing information used by graph_converter.py to build edges.

TR.csv
2nd dataset export for Technische Ressourcen (TR) used by graph_converter.py to add TR nodes and connect them to MeLo/MaLo via metering relations.

NELO.csv
2nd dataset export for Netzlokation (NeLo) used by graph_converter.py to add NeLo.

BNDL2MC.csv
Label mapping from bundle to (MaLo, MeLo) to MCID used for evaluation in dgmc_matcher.py, baseline_constraints.py, and baseline_graph_similarity.py. MCID has values for LBS concepts. 

---
Data Files (LBS)
All JSON files in lbs_templates represent their corresponding standardized LBS consept in the BDEW codelist from 2024.
