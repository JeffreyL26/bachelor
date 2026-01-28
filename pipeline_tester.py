from graph_pipeline import load_jsonl_as_pyg
import os


if __name__ == "__main__":
    """
    Konsolenausgabe zum Testen der Konvertierung in PyG-Objekte
    """

    BASE = os.path.dirname(os.path.abspath(__file__))

    ist_path = os.path.join(BASE, "data", "ist_graphs_all.jsonl")
    soll_path = os.path.join(BASE, "data", "lbs_soll_graphs.jsonl")

    print("Ist-Pfad:", ist_path)
    ist_graphs = load_jsonl_as_pyg(ist_path, undirected=True)  # symmetrische Verbindungen
    print("Anzahl Ist-Graphen:", len(ist_graphs))

    print()
    print("Soll-Pfad:", soll_path)
    soll_graphs = load_jsonl_as_pyg(soll_path, undirected=True)
    print("Anzahl Soll-Graphen:", len(soll_graphs))

    if ist_graphs:
        g0 = ist_graphs[0]
        print("\n--- Beispiel: 1. Ist-Graph ---")
        print("x-Shape:", g0.x.shape)
        print("edge_index-Shape:", g0.edge_index.shape)
        print("edge_attr-Shape:", g0.edge_attr.shape)
        print("Graph-ID:", getattr(g0, "graph_id", None))
        print("Graph-Label:", getattr(g0, "graph_label", None))
        print("Graph-Attrs Keys:", list(getattr(g0, "graph_attrs", {}).keys())[:20])
        print("Node-Types (erste 10):", getattr(g0, "node_types", [])[:10])

    if soll_graphs:
        t0 = soll_graphs[0]
        print("\n--- Beispiel: 1. Soll-Graph ---")
        print("x-Shape:", t0.x.shape)
        print("edge_index-Shape:", t0.edge_index.shape)
        print("edge_attr-Shape:", t0.edge_attr.shape)
        print("Graph-ID:", getattr(t0, "graph_id", None))
        print("Graph-Label:", getattr(t0, "graph_label", None))
        print("Graph-Attrs Keys:", list(getattr(t0, "graph_attrs", {}).keys())[:20])
        print("Node-Types (erste 10):", getattr(t0, "node_types", [])[:10])
