"""
Fetch robotics and LLM research papers from arXiv and cache results locally.

Three query groups:
  hardware        - cs.RO papers on legged/aerial/manipulation robots (build vs. buy)
  robotics_models - cs.RO papers on robot learning/policy (fine-tune vs. from scratch)
  llm_models      - cs.CL/cs.LG papers on LLMs and foundation models

Usage:
    python fetch_data.py                        # Fetch all groups
    python fetch_data.py --groups hardware      # Fetch one group only
    python fetch_data.py --dry-run              # Print plan without fetching
    python fetch_data.py --limit 500            # Papers per query (default: 500)
"""

import argparse
import json
import os
import time

import arxiv
from tqdm import tqdm

QUERY_GROUPS = {
    "hardware": [
        ("legged_robots",    'cat:cs.RO AND abs:"legged robot"'),
        ("quadruped_robots", 'cat:cs.RO AND abs:"quadruped"'),
        ("bipedal_humanoid", 'cat:cs.RO AND (abs:"bipedal" OR abs:"humanoid robot")'),
        ("aerial_robots",    'cat:cs.RO AND (abs:"aerial robot" OR abs:"quadrotor" OR abs:"UAV")'),
        ("robot_manipulation", 'cat:cs.RO AND abs:"robot manipulation"'),
    ],
    "robotics_models": [
        ("robot_learning",    'cat:cs.RO AND abs:"robot learning"'),
        ("imitation_learning", 'cat:cs.RO AND abs:"imitation learning"'),
        ("rl_robot",          'cat:cs.RO AND abs:"reinforcement learning"'),
        ("robot_foundation",  'cat:cs.RO AND (abs:"foundation model" OR abs:"pretrained" OR abs:"pre-trained")'),
    ],
    "llm_models": [
        ("language_model",    '(cat:cs.CL OR cat:cs.LG) AND abs:"language model"'),
        ("foundation_model",  '(cat:cs.CL OR cat:cs.LG) AND abs:"foundation model"'),
        ("llm_finetuning",    '(cat:cs.CL OR cat:cs.LG) AND abs:"fine-tuning"'),
        ("model_pretraining", '(cat:cs.CL OR cat:cs.LG) AND abs:"pretraining"'),
    ],
}

YEAR_RANGE = (2010, 2025)
DATA_DIR = "data"


def fetch_query(group, label, query, limit, dry_run):
    out_path = os.path.join(DATA_DIR, f"{group}__{label}.json")

    if os.path.exists(out_path):
        print(f"  [skip] {label}: cache exists")
        return

    if dry_run:
        print(f"  [dry-run] group={group} label={label} limit={limit}")
        print(f"    query: {query}")
        return

    print(f"  [fetch] {label} ...")
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    with tqdm(total=limit, unit="paper") as pbar:
        for result in client.results(search):
            pbar.update(1)
            year = result.published.year
            if not (YEAR_RANGE[0] <= year <= YEAR_RANGE[1]):
                continue
            papers.append(
                {
                    "paperId": result.entry_id,
                    "title": result.title,
                    "year": year,
                    "authorCount": len(result.authors),
                    "abstract": result.summary or "",
                    "categories": result.categories,
                }
            )

    print(f"    -> {len(papers)} papers in {YEAR_RANGE[0]}-{YEAR_RANGE[1]}")
    with open(out_path, "w") as f:
        json.dump({"group": group, "label": label, "query": query, "papers": papers}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=500, help="Max papers per query (default: 500)")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(QUERY_GROUPS.keys()),
        default=list(QUERY_GROUPS.keys()),
        help="Which query groups to fetch (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    for group in args.groups:
        print(f"\n=== Group: {group} ===")
        for label, query in QUERY_GROUPS[group]:
            fetch_query(group, label, query, args.limit, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
