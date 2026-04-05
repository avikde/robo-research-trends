"""
Classify papers by build-vs-buy (hardware) and train-from-scratch vs fine-tune (models),
then plot trends over time.

Outputs:
    figures/hardware_trend.png         - % of papers mentioning commercial vs custom hardware
    figures/robotics_models_trend.png  - % of robot learning papers mentioning fine-tuning vs from-scratch
    figures/llm_models_trend.png       - % of LLM papers mentioning fine-tuning vs from-scratch
    data/hardware_classified.csv
    data/robotics_models_classified.csv
    data/llm_models_classified.csv

Usage:
    python analyze.py
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "data"
FIGURES_DIR = "figures"
YEAR_RANGE = (2010, 2025)

# ---------------------------------------------------------------------------
# Keyword lists for classification
# ---------------------------------------------------------------------------

# Named commercial robot platforms (case-insensitive substring match)
COMMERCIAL_PLATFORMS = [
    # Legged
    "spot", "boston dynamics", "anymal", "anybotics",
    "unitree", "go1", "go2", "b1", "h1", "a1 robot",
    "cassie", "digit", "atlas",
    # Manipulation
    "franka", "panda robot", "widowx", "widow x", "aloha", "trossen",
    "ur5", "ur10", "universal robots", "kuka", "sawyer", "baxter", "kinova",
    # Drones
    "dji", "crazyflie", "parrot drone",
]

# Language indicating researchers built their own hardware
CUSTOM_HARDWARE = [
    "custom robot", "novel robot", "we designed", "we built",
    "custom-built", "custom hardware", "novel hardware",
    "custom platform", "prototype robot", "our robot platform",
    "in-house robot", "we constructed", "we fabricated",
]

# Language indicating use of a pre-trained or fine-tuned model
PRETRAINED_SIGNALS = [
    "fine-tun", "finetuning", "fine tuning",
    "pretrained", "pre-trained", "pre trained",
    "foundation model", "transfer learning",
    "zero-shot", "few-shot",
    "large language model", "vision-language model", "vision language model",
    "diffusion model", "gpt", "llama", "bert", "clip", "t5",
    "qwen", "gemini", "mistral", "palm", "flamingo",
    "we finetune", "we fine-tune",
]

# Language indicating training a model from scratch
SCRATCH_SIGNALS = [
    "trained from scratch", "train from scratch", "training from scratch",
    "end-to-end train", "end-to-end learning",
    "we train a", "we train our", "train our model", "train our network",
    "from random initialization", "randomly initialized",
]


def contains_any(text, keywords):
    text = text.lower()
    return any(kw.lower() in text for kw in keywords)


def load_group(group_prefix):
    rows = []
    for path in glob.glob(os.path.join(DATA_DIR, f"{group_prefix}__*.json")):
        with open(path) as f:
            d = json.load(f)
        for p in d["papers"]:
            rows.append(
                {
                    "label": d["label"],
                    "year": p["year"],
                    "title": p.get("title") or "",
                    "abstract": p.get("abstract") or "",
                }
            )
    return pd.DataFrame(rows)


def classify_hardware(df):
    text = df["title"] + " " + df["abstract"]
    df = df.copy()
    df["commercial"] = text.apply(lambda t: contains_any(t, COMMERCIAL_PLATFORMS))
    df["custom"] = text.apply(lambda t: contains_any(t, CUSTOM_HARDWARE))
    return df


def classify_models(df):
    text = df["title"] + " " + df["abstract"]
    df = df.copy()
    df["pretrained"] = text.apply(lambda t: contains_any(t, PRETRAINED_SIGNALS))
    df["from_scratch"] = text.apply(lambda t: contains_any(t, SCRATCH_SIGNALS))
    return df


def yearly_pct(df, flag_col):
    """Return a Series: year -> % of papers with flag_col == True."""
    grp = df.groupby("year")
    return (grp[flag_col].sum() / grp[flag_col].count() * 100).rename(flag_col)


def plot_hardware(df, out_path):
    df = df[df["year"].between(*YEAR_RANGE)]
    years = sorted(df["year"].unique())

    pct_commercial = yearly_pct(df, "commercial")
    pct_custom = yearly_pct(df, "custom")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pct_commercial.index, pct_commercial.values, marker="o", label="Commercial platform mentioned")
    ax.plot(pct_custom.index, pct_custom.values, marker="s", linestyle="--", label="Custom/novel hardware mentioned")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of papers")
    ax.set_title("Hardware: commercial platforms vs. custom builds over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_models(df, title, out_path):
    df = df[df["year"].between(*YEAR_RANGE)]

    pct_pretrained = yearly_pct(df, "pretrained")
    pct_scratch = yearly_pct(df, "from_scratch")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pct_pretrained.index, pct_pretrained.values, marker="o", label="Pre-trained / fine-tuned")
    ax.plot(pct_scratch.index, pct_scratch.values, marker="s", linestyle="--", label="Trained from scratch")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of papers")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Hardware ---
    hw = load_group("hardware")
    if hw.empty:
        print("No hardware data found. Run: python fetch_data.py --groups hardware")
    else:
        print(f"Hardware: {len(hw)} papers loaded")
        hw = classify_hardware(hw)
        hw.to_csv(os.path.join(DATA_DIR, "hardware_classified.csv"), index=False)
        plot_hardware(hw, os.path.join(FIGURES_DIR, "hardware_trend.png"))

    # --- Robotics models ---
    rm = load_group("robotics_models")
    if rm.empty:
        print("No robotics_models data found. Run: python fetch_data.py --groups robotics_models")
    else:
        print(f"Robotics models: {len(rm)} papers loaded")
        rm = classify_models(rm)
        rm.to_csv(os.path.join(DATA_DIR, "robotics_models_classified.csv"), index=False)
        plot_models(
            rm,
            "Robot learning: pre-trained/fine-tuned vs. trained from scratch",
            os.path.join(FIGURES_DIR, "robotics_models_trend.png"),
        )

    # --- LLM / foundation models ---
    lm = load_group("llm_models")
    if lm.empty:
        print("No llm_models data found. Run: python fetch_data.py --groups llm_models")
    else:
        print(f"LLM models: {len(lm)} papers loaded")
        lm = classify_models(lm)
        lm.to_csv(os.path.join(DATA_DIR, "llm_models_classified.csv"), index=False)
        plot_models(
            lm,
            "LLM / foundation models: pre-trained/fine-tuned vs. trained from scratch",
            os.path.join(FIGURES_DIR, "llm_models_trend.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
