from graph_pipeline import load_jsonl_as_pyg
import os

if __name__ == "__main__":
    """
    Konsolenausgae zum Testen der Konvertierung in PyG-Objekte
    """
    BASE = os.path.dirname(os.path.abspath(__file__))
    ist_graphs = load_jsonl_as_pyg(os.path.join(BASE, "data", "ist_graphs.jsonl"), undirected=True) # Da symmetrische Verbindungen
    print("Anzahl Ist-Graphen:", len(ist_graphs))
    print("Erster Graph:", ist_graphs[0])
    print("x-Shape:", ist_graphs[0].x.shape)
    print("edge_index-Shape:", ist_graphs[0].edge_index.shape)
    print("edge_attr-Shape:", ist_graphs[0].edge_attr.shape)
    print("graph_id:", ist_graphs[0].graph_id)
    print("graph_attrs:", ist_graphs[0].graph_attrs)