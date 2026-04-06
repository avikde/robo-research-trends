"""
Classify paper abstracts using Gemini into exhaustive, mutually exclusive categories.

Hardware papers  -> commercial | custom | simulation | no_hardware
Model papers     -> pretrained | scratch | no_ml | unclear

Results are cached in data/ml_classifications.json (paperId -> label).
Re-running is safe: already-classified papers are skipped.

Usage:
    export GEMINI_API_KEY=...
    python classify_ml.py                           # classify all groups
    python classify_ml.py --groups hardware         # classify one group only
    python classify_ml.py --groups hardware --print # print abstracts for a group (no classification)
    python classify_ml.py --batch-size 20           # papers per Gemini call (default: 20)
"""

import argparse
import concurrent.futures
import glob
import json
import os
import time

from google import genai
from google.genai import types
from tqdm import tqdm

CALL_TIMEOUT = 15  # seconds before a hanging Gemini call is abandoned

DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "ml_classifications.json")
MODEL = "gemini-3.1-flash-lite-preview"
BATCH_SIZE = 20

HARDWARE_GROUPS = {"hardware"}
MODEL_GROUPS = {"robotics_models", "llm_models"}

HARDWARE_EXTRACT_PROMPT = """\
For each robotics paper, extract the robot platform(s) used in experiments, if any.
Look for any robot referred to by a specific name, product name, model number, or brand.
If no specific platform is named, describe what you can infer (e.g. "unnamed quadrotor", "custom bipedal robot").
If it is a simulation-only or purely theoretical paper, say "simulation" or "none".

Papers:
{papers_json}

Respond with a JSON array only, no markdown:
[{{"id": "<id>", "platform": "<platform name or description>"}}, ...]"""

HARDWARE_CLASSIFY_PROMPT = """\
Classify each robotics paper by its primary hardware approach, given the robot platform identified.

- "commercial"  : the platform is a product that can be purchased — identified by a brand name,
                  product name, or model number (even if you don't recognize it). Clues: the paper
                  mentions buying, ordering, or acquiring it; it has a version number or SKU-like name;
                  or it is referred to as a third-party or manufacturer's product.
- "custom"      : the research team designed, built, or assembled the robot hardware themselves.
                  Clues: "we designed", "we built", "novel hardware", "prototype", "custom".
- "simulation"  : all experiments are in simulation; no physical robot is used.
- "no_hardware" : purely theoretical, mathematical, or dataset/benchmark paper; no robot involved.

Papers:
{papers_json}

Respond with a JSON array only, no markdown:
[{{"id": "<id>", "label": "<category>"}}, ...]"""

MODELS_PROMPT = """\
Classify each ML paper by its primary model training approach. Choose exactly one category per paper:

- "pretrained"  : uses a pre-trained model, fine-tunes an existing model, applies transfer learning,
                  or builds on a foundation model / LLM / VLM
- "scratch"     : trains a neural network or model from scratch (random initialization)
- "no_ml"       : uses classical or non-ML methods only (e.g. trajectory optimization, MPC,
                  dynamic programming, SLAM, analytical control)
- "unclear"     : the abstract does not contain enough information to determine the approach

Papers:
{papers_json}

Respond with a JSON array only, no markdown:
[{{"id": "<id>", "label": "<category>"}}, ...]"""


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def load_papers_for_group(group_prefix):
    papers = []
    for path in glob.glob(os.path.join(DATA_DIR, f"{group_prefix}__*.json")):
        with open(path) as f:
            d = json.load(f)
        group = d["group"]
        for p in d["papers"]:
            papers.append({
                "id": p["paperId"],
                "title": p.get("title") or "",
                "abstract": (p.get("abstract") or "")[:1000],  # trim to save tokens
                "group": group,
            })
    return papers


def call_gemini(client, prompt_template, batch):
    papers_json = json.dumps(
        [{"id": p["id"], "title": p["title"], "abstract": p["abstract"]} for p in batch],
        indent=2,
    )
    prompt = prompt_template.format(papers_json=papers_json)

    def _call():
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        return future.result(timeout=CALL_TIMEOUT)


def call_with_retry(client, prompt_template, batch):
    for attempt in range(2):
        try:
            return call_gemini(client, prompt_template, batch)
        except concurrent.futures.TimeoutError:
            tqdm.write(f"    [warn] timed out after {CALL_TIMEOUT}s (attempt {attempt + 1})")
        except Exception as e:
            tqdm.write(f"    [warn] failed: {e} (attempt {attempt + 1})")
        if attempt == 0:
            time.sleep(5)
    return None


def classify_group(client, group, papers, cache, batch_size):
    is_hardware = group in HARDWARE_GROUPS

    uncached = [p for p in papers if p["id"] not in cache]
    if not uncached:
        print(f"  [{group}] all {len(papers)} papers already classified")
        return

    print(f"  [{group}] classifying {len(uncached)} papers ({len(papers) - len(uncached)} cached) ...")
    batches = [uncached[i:i + batch_size] for i in range(0, len(uncached), batch_size)]

    for batch in tqdm(batches, unit="batch"):
        if is_hardware:
            # Stage 1: extract platform names
            extracted = call_with_retry(client, HARDWARE_EXTRACT_PROMPT, batch)
            if extracted is None:
                continue
            # Augment batch with extracted platform for stage 2
            platform_by_id = {item["id"]: item.get("platform", "") for item in extracted}
            batch2 = [
                {**p, "abstract": f"Platform: {platform_by_id.get(p['id'], '')}. Abstract: {p['abstract']}"}
                for p in batch
            ]
            # Stage 2: classify using platform info
            results = call_with_retry(client, HARDWARE_CLASSIFY_PROMPT, batch2)
        else:
            results = call_with_retry(client, MODELS_PROMPT, batch)

        if results is None:
            continue

        for item in results:
            # store {"label": ..., "platform": ...} for hardware, plain label for models
            if is_hardware:
                platform_by_id = {item["id"]: item.get("platform", "") for item in (extracted or [])}
                cache[item["id"]] = {
                    "label": item["label"],
                    "platform": platform_by_id.get(item["id"], ""),
                }
            else:
                cache[item["id"]] = item["label"]

        save_cache(cache)
        time.sleep(0.5)


def print_abstracts(groups):
    cache = load_cache()
    for group in groups:
        papers = load_papers_for_group(group)
        if not papers:
            print(f"[{group}] no data found — run fetch_data.py first")
            continue
        print(f"\n{'=' * 60}")
        print(f"Group: {group}  ({len(papers)} papers)")
        print(f"{'=' * 60}")
        for p in papers:
            entry = cache.get(p["id"])
            if entry is None:
                label, platform = "unclassified", ""
            elif isinstance(entry, dict):
                label, platform = entry.get("label", "?"), entry.get("platform", "")
            else:
                label, platform = entry, ""
            platform_str = f"  platform: {platform}" if platform else ""
            print(f"\n[{label}]{platform_str}")
            print(f"  {p['title']}")
            print(p["abstract"] or "(no abstract)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["hardware", "robotics_models", "llm_models"],
        default=["hardware", "robotics_models", "llm_models"],
    )
    parser.add_argument("--print", action="store_true", help="Print abstracts for the given groups and exit")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.print:
        print_abstracts(args.groups)
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY environment variable before running.")

    client = genai.Client(api_key=api_key)
    cache = load_cache()

    for group in args.groups:
        papers = load_papers_for_group(group)
        if not papers:
            print(f"  [{group}] no data found — run fetch_data.py first")
            continue
        classify_group(client, group, papers, cache, args.batch_size)

    print(f"\nDone. {len(cache)} papers classified total.")
    print(f"Cache saved to {CACHE_FILE}")


if __name__ == "__main__":
    main()
