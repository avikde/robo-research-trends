"""
Load classified paper data and plot trends over time.

Uses ML classifications from data/ml_classifications.json (produced by classify_ml.py)
when available, falling back to keyword matching otherwise.

Outputs:
    figures/hardware_trend.png         - hardware approach breakdown by year
    figures/robotics_models_trend.png  - model training approach for robot learning papers
    figures/llm_models_trend.png       - model training approach for LLM papers
    data/hardware_classified.csv
    data/robotics_models_classified.csv
    data/llm_models_classified.csv

Usage:
    python analyze.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "data"
FIGURES_DIR = "figures"
YEAR_RANGE = (2015, 2025)
ML_CACHE = os.path.join(DATA_DIR, "ml_classifications.json")

# ---------------------------------------------------------------------------
# Keyword fallbacks (used only when ML classifications are unavailable)
# ---------------------------------------------------------------------------

COMMERCIAL_PLATFORMS = [
    "spot", "boston dynamics", "anymal", "anybotics",
    "unitree", "go1", "go2", "b1", "h1", "a1 robot",
    "cassie", "digit", "atlas",
    "franka", "panda robot", "widowx", "widow x", "aloha", "trossen",
    "ur5", "ur10", "universal robots", "kuka", "sawyer", "baxter", "kinova",
    "dji", "crazyflie", "parrot drone",
]
CUSTOM_HARDWARE = [
    "custom robot", "novel robot", "we designed", "we built",
    "custom-built", "custom hardware", "novel hardware",
    "custom platform", "prototype robot", "our robot platform",
    "in-house robot", "we constructed", "we fabricated",
]
PRETRAINED_SIGNALS = [
    "fine-tun", "finetuning", "fine tuning", "pretrained", "pre-trained",
    "foundation model", "transfer learning", "zero-shot", "few-shot",
    "large language model", "vision-language model", "diffusion model",
    "gpt", "llama", "bert", "clip", "t5", "qwen", "gemini", "mistral",
]
SCRATCH_SIGNALS = [
    "trained from scratch", "train from scratch", "end-to-end train",
    "we train a", "we train our", "train our model", "from random initialization",
]


def contains_any(text, keywords):
    text = text.lower()
    return any(kw.lower() in text for kw in keywords)


def load_ml_cache():
    if os.path.exists(ML_CACHE):
        with open(ML_CACHE) as f:
            return json.load(f)
    return {}


def load_group(group_prefix):
    rows = []
    for path in glob.glob(os.path.join(DATA_DIR, f"{group_prefix}__*.json")):
        with open(path) as f:
            d = json.load(f)
        for p in d["papers"]:
            rows.append({
                "paperId": p.get("paperId") or "",
                "label": d["label"],
                "year": p["year"],
                "title": p.get("title") or "",
                "abstract": p.get("abstract") or "",
            })
    return pd.DataFrame(rows)


def apply_hw_labels(df, ml_cache):
    """Add a 'hw_label' column: commercial / custom / simulation / no_hardware."""
    def classify_row(row):
        ml = ml_cache.get(row["paperId"])
        if ml:
            return ml
        text = (row["title"] + " " + row["abstract"]).lower()
        if contains_any(text, COMMERCIAL_PLATFORMS):
            return "commercial"
        if contains_any(text, CUSTOM_HARDWARE):
            return "custom"
        return "no_hardware"
    df = df.copy()
    df["hw_label"] = df.apply(classify_row, axis=1)
    return df


def apply_model_labels(df, ml_cache):
    """Add a 'model_label' column: pretrained / scratch / no_ml / unclear."""
    def classify_row(row):
        ml = ml_cache.get(row["paperId"])
        if ml:
            return ml
        text = (row["title"] + " " + row["abstract"]).lower()
        if contains_any(text, PRETRAINED_SIGNALS):
            return "pretrained"
        if contains_any(text, SCRATCH_SIGNALS):
            return "scratch"
        return "unclear"
    df = df.copy()
    df["model_label"] = df.apply(classify_row, axis=1)
    return df


def yearly_pct(df, label_col, categories):
    """Return a DataFrame: year x category -> % of papers that year."""
    df = df[df["year"].between(*YEAR_RANGE)]
    counts = df.groupby(["year", label_col]).size().unstack(fill_value=0)
    # ensure all expected categories are present
    for cat in categories:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[categories]
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return pct


def plot_stacked(pct, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    pct.plot(kind="area", stacked=True, ax=ax, alpha=0.75)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of papers")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    ml_cache = load_ml_cache()

    using_ml = bool(ml_cache)
    print(f"ML classifications: {'loaded ' + str(len(ml_cache)) + ' entries' if using_ml else 'not found, using keyword fallback'}")

    # --- Hardware ---
    hw = load_group("hardware")
    if hw.empty:
        print("No hardware data found. Run: python fetch_data.py --groups hardware")
    else:
        hw = apply_hw_labels(hw, ml_cache)
        hw.to_csv(os.path.join(DATA_DIR, "hardware_classified.csv"), index=False)
        pct = yearly_pct(hw, "hw_label", ["commercial", "custom", "simulation", "no_hardware"])
        print(f"Hardware: {len(hw)} papers\n{pct.round(1).to_string()}\n")
        plot_stacked(
            pct,
            "Hardware approach over time\n(% of papers per year)",
            os.path.join(FIGURES_DIR, "hardware_trend.png"),
        )

    # --- Robotics models ---
    rm = load_group("robotics_models")
    if rm.empty:
        print("No robotics_models data found. Run: python fetch_data.py --groups robotics_models")
    else:
        rm = apply_model_labels(rm, ml_cache)
        rm.to_csv(os.path.join(DATA_DIR, "robotics_models_classified.csv"), index=False)
        pct = yearly_pct(rm, "model_label", ["pretrained", "scratch", "no_ml", "unclear"])
        print(f"Robotics models: {len(rm)} papers\n{pct.round(1).to_string()}\n")
        plot_stacked(
            pct,
            "Robot learning: model training approach over time\n(% of papers per year)",
            os.path.join(FIGURES_DIR, "robotics_models_trend.png"),
        )

    # --- LLM / foundation models ---
    lm = load_group("llm_models")
    if lm.empty:
        print("No llm_models data found. Run: python fetch_data.py --groups llm_models")
    else:
        lm = apply_model_labels(lm, ml_cache)
        lm.to_csv(os.path.join(DATA_DIR, "llm_models_classified.csv"), index=False)
        pct = yearly_pct(lm, "model_label", ["pretrained", "scratch", "no_ml", "unclear"])
        print(f"LLM models: {len(lm)} papers\n{pct.round(1).to_string()}\n")
        plot_stacked(
            pct,
            "LLM / foundation models: training approach over time\n(% of papers per year)",
            os.path.join(FIGURES_DIR, "llm_models_trend.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
