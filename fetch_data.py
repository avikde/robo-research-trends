"""
Fetch robotics research papers from Semantic Scholar API and cache results locally.

Three separate query groups are used:
  hardware        - legged/manipulation/aerial robot papers (for build-vs-buy analysis)
  robotics_models - robot learning/policy papers (for train-from-scratch vs fine-tune)
  llm_models      - LLM and foundation model papers (for train-from-scratch vs fine-tune)

Usage:
    python fetch_data.py                        # Fetch all groups
    python fetch_data.py --groups hardware      # Fetch one group only
    python fetch_data.py --dry-run              # Print plan without hitting API
    python fetch_data.py --limit 300            # Papers per query (default: 500)
"""

import argparse
import json
import os
import time

from semanticscholar import SemanticScholar

QUERY_GROUPS = {
    "hardware": [
        ("legged_robots", "legged robot locomotion"),
        ("quadruped_robots", "quadruped robot"),
        ("bipedal_humanoid", "bipedal humanoid robot"),
        ("aerial_robots", "aerial robot UAV"),
        ("robot_manipulation", "robot manipulation arm"),
    ],
    "robotics_models": [
        ("robot_learning_policy", "robot learning policy"),
        ("imitation_learning_robot", "imitation learning robot"),
        ("rl_locomotion", "reinforcement learning robot locomotion"),
        ("robot_foundation_model", "robot foundation model"),
    ],
    "llm_models": [
        ("llm_pretraining", "large language model pretraining"),
        ("llm_finetuning", "large language model fine-tuning"),
        ("foundation_model_training", "foundation model training"),
        ("vision_language_model", "vision language model"),
    ],
}

FIELDS = ["title", "year", "referenceCount", "citationCount", "authors", "abstract"]

YEAR_RANGE = (2010, 2025)
DATA_DIR = "data"


def fetch_query(sch, group, label, query, limit, dry_run):
    out_path = os.path.join(DATA_DIR, f"{group}__{label}.json")

    if os.path.exists(out_path):
        print(f"  [skip] {label}: cache exists")
        return

    if dry_run:
        print(f"  [dry-run] group={group} label={label} query='{query}' limit={limit}")
        return

    print(f"  [fetch] {label}: '{query}' ...")
    results = sch.search_paper(query, fields=FIELDS, limit=limit)

    papers = []
    for paper in results:
        year = paper.year
        if year is None or not (YEAR_RANGE[0] <= year <= YEAR_RANGE[1]):
            continue
        papers.append(
            {
                "paperId": paper.paperId,
                "title": paper.title,
                "year": year,
                "referenceCount": paper.referenceCount,
                "citationCount": paper.citationCount,
                "authorCount": len(paper.authors) if paper.authors else None,
                "abstract": paper.abstract or "",
            }
        )

    print(f"    -> {len(papers)} papers in {YEAR_RANGE[0]}-{YEAR_RANGE[1]}")
    with open(out_path, "w") as f:
        json.dump({"group": group, "label": label, "query": query, "papers": papers}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(QUERY_GROUPS.keys()),
        default=list(QUERY_GROUPS.keys()),
        help="Which query groups to fetch (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    sch = SemanticScholar()

    for group in args.groups:
        print(f"\n=== Group: {group} ===")
        for label, query in QUERY_GROUPS[group]:
            fetch_query(sch, group, label, query, args.limit, args.dry_run)
            if not args.dry_run:
                time.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
