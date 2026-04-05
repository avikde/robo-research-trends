"""
Classify paper abstracts using Gemini into exhaustive, mutually exclusive categories.

Hardware papers  -> commercial | custom | simulation | no_hardware
Model papers     -> pretrained | scratch | no_ml | unclear

Results are cached in data/ml_classifications.json (paperId -> label).
Re-running is safe: already-classified papers are skipped.

Usage:
    export GEMINI_API_KEY=...
    python classify_ml.py                  # classify all groups
    python classify_ml.py --groups hardware
    python classify_ml.py --batch-size 20  # papers per Gemini call (default: 20)
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

HARDWARE_PROMPT = """\
Classify each robotics paper by its primary hardware approach. Choose exactly one category per paper:

- "commercial"   : uses a named off-the-shelf robot platform (e.g. Spot, ANYmal, Unitree A1/Go1/Go2/H1,
                   Franka/Panda, UR5/UR10, Kuka, Sawyer, Baxter, Kinova, Cassie, Digit, Atlas,
                   DJI, Crazyflie, or similar)
- "custom"       : the research team designed, built, or assembled their own robot hardware
- "simulation"   : all experiments are in simulation; no physical robot is used
- "no_hardware"  : purely theoretical, mathematical, or dataset/benchmark paper; no robot involved

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


def classify_group(client, group, papers, cache, batch_size):
    prompt = HARDWARE_PROMPT if group in HARDWARE_GROUPS else MODELS_PROMPT

    uncached = [p for p in papers if p["id"] not in cache]
    if not uncached:
        print(f"  [{group}] all {len(papers)} papers already classified")
        return

    print(f"  [{group}] classifying {len(uncached)} papers ({len(papers) - len(uncached)} cached) ...")
    batches = [uncached[i:i + batch_size] for i in range(0, len(uncached), batch_size)]

    for batch in tqdm(batches, unit="batch"):
        for attempt in range(2):
            try:
                results = call_gemini(client, prompt, batch)
                for item in results:
                    cache[item["id"]] = item["label"]
                break
            except concurrent.futures.TimeoutError:
                tqdm.write(f"    [warn] batch timed out after {CALL_TIMEOUT}s (attempt {attempt + 1})")
                if attempt == 0:
                    time.sleep(5)
            except Exception as e:
                tqdm.write(f"    [warn] batch failed: {e} (attempt {attempt + 1})")
                if attempt == 0:
                    time.sleep(5)
        save_cache(cache)
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["hardware", "robotics_models", "llm_models"],
        default=["hardware", "robotics_models", "llm_models"],
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

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
