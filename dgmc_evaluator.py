from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List

#TODO: Kommentare

MatchRec = Dict[str, Any]


def load_jsonl(path: str) -> List[MatchRec]:
    records: List[MatchRec] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize_overall(matches: List[MatchRec]) -> None:
    print("=== Overview ===")
    print("Anzahl Matches:", len(matches))

    patterns = Counter(m["ist_pattern"] for m in matches)
    print("Ist-Pattern-Verteilung:")
    for pat, cnt in sorted(patterns.items()):
        print(f"  Pattern {pat}: {cnt}")

    templates = Counter(m["best_template_label"] for m in matches)
    print("\nTop-Templates (Best-Match-Label):")
    for label, cnt in templates.most_common():
        print(f"  Template {label}: {cnt}")


def summarize_scores_by_pattern(matches: List[MatchRec]) -> None:
    scores_by_pattern: Dict[str, List[float]] = defaultdict(list)
    for m in matches:
        pat = m.get("ist_pattern", "unknown")
        scores_by_pattern[pat].append(float(m.get("best_score", 0.0)))

    print("\n=== Scores je Pattern ===")
    for pat, scores in sorted(scores_by_pattern.items()):
        if not scores:
            continue
        mu = mean(scores)
        sd = pstdev(scores) if len(scores) > 1 else 0.0
        print(f"Pattern {pat}: n={len(scores)}, "
              f"min={min(scores):.6f}, max={max(scores):.6f}, "
              f"mean={mu:.6f}, std={sd:.6f}")


def analyze_all_to_one_mapping(matches: List[MatchRec]) -> None:
    """
    Prüft, wie oft alle Ist-Knoten auf denselben Template-Knoten gemappt werden.
    Das ist z.B. bei 1:1-BEZG-Fällen sichtbar.
    """
    total = len(matches)
    all_to_one_count = 0

    all_to_one_by_pattern: Dict[str, int] = defaultdict(int)
    total_by_pattern: Dict[str, int] = defaultdict(int)

    for m in matches:
        pat = m.get("ist_pattern", "unknown")
        total_by_pattern[pat] += 1

        mapping = m.get("mapping", [])
        if not mapping:
            continue

        tgt_indices = {mp.get("tgt_index") for mp in mapping}
        if len(tgt_indices) == 1:
            all_to_one_count += 1
            all_to_one_by_pattern[pat] += 1

    print("\n=== Degenerates Matching (alle Ist-Knoten -> ein Template-Knoten) ===")
    print(f"Gesamt: {all_to_one_count}/{total} ({all_to_one_count/total*100:.1f}%)")
    for pat, cnt in sorted(all_to_one_by_pattern.items()):
        total_pat = total_by_pattern.get(pat, 0)
        ratio = cnt / total_pat * 100 if total_pat > 0 else 0.0
        print(f"  Pattern {pat}: {cnt}/{total_pat} ({ratio:.1f}%)")


def print_sample_matches(matches: List[MatchRec],
                         pattern_filter: str | None = None,
                         template_filter: str | None = None,
                         k: int = 5) -> None:
    """
    Gibt ein paar Beispielmatches aus, optional gefiltert nach Pattern und/oder Template-Label.
    """
    print("\n=== Beispiel-Matches ===")
    shown = 0
    for m in matches:
        if pattern_filter and m.get("ist_pattern") != pattern_filter:
            continue
        if template_filter and m.get("best_template_label") != template_filter:
            continue

        print("-" * 60)
        print(f"Ist-Graph: {m.get('ist_graph_id')}")
        print(f"  Pattern: {m.get('ist_pattern')}, "
              f"MaLo={m.get('ist_malo_count')}, MeLo={m.get('ist_melo_count')}")
        print(f"  Best Template: {m.get('best_template_label')} "
              f"(Graph-ID: {m.get('best_template_graph_id')}, "
              f"Pattern: {m.get('best_template_pattern')})")
        print(f"  Best Score: {m.get('best_score'):.6f}")
        print("  Mapping (erste 10 Knoten):")
        for mp in m.get("mapping", [])[:10]:
            print(f"    src[{mp['src_index']}]={mp['src_node_id']}  "
                  f"->  tgt[{mp['tgt_index']}]={mp['tgt_node_id']}  "
                  f"(score={mp['score']:.6f})")

        shown += 1
        if shown >= k:
            break

    if shown == 0:
        print("  (Keine Matches für diese Filter gefunden.)")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    matches_path = os.path.join(base, "data", "ist_dgmc_matches.jsonl")

    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {matches_path}")

    matches = load_jsonl(matches_path)

    summarize_overall(matches)
    summarize_scores_by_pattern(matches)
    analyze_all_to_one_mapping(matches)

    # Optional: ein paar Beispiele zeigen, z.B. für Pattern 1:1
    print_sample_matches(matches, pattern_filter="1:1", template_filter="9992000000175", k=3)


if __name__ == "__main__":
    main()
