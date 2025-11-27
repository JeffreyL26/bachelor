from graph_pipeline import load_jsonl_as_pyg

if __name__ == "__main__":
    ist_graphs = load_jsonl_as_pyg("ist_graphs_small.jsonl", undirected=True)
    print("Anzahl Ist-Graphen:", len(ist_graphs))
    print("Erster Graph:", ist_graphs[0])
    print("x-Shape:", ist_graphs[0].x.shape)
    print("edge_index-Shape:", ist_graphs[0].edge_index.shape)
    print("edge_attr-Shape:", ist_graphs[0].edge_attr.shape)
    print("graph_id:", ist_graphs[0].graph_id)
    print("graph_attrs:", ist_graphs[0].graph_attrs)