"""
Classify papers by build-vs-buy (hardware) and train-from-scratch vs fine-tune (models),
then plot trends over time, normalized by papers with empirical demonstrations.

Outputs:
    figures/hardware_trend.png         - % of empirical papers mentioning commercial vs custom hardware
    figures/robotics_models_trend.png  - % of empirical robot learning papers: fine-tuned vs from-scratch
    figures/llm_models_trend.png       - % of empirical LLM papers: fine-tuned vs from-scratch
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

# Language indicating an empirical paper with real experimental demonstrations
EMPIRICAL_SIGNALS = [
    "we demonstrate", "we show", "we evaluate", "we validate", "we test",
    "real robot", "physical robot", "real-world", "hardware experiment",
    "on a robot", "on the robot", "we deploy", "deployed on",
    "our experiments", "in experiments", "experiment shows",
    "results show", "results demonstrate", "we achieve",
    "benchmark", "evaluation shows",
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


def classify(df, extra_flags):
    text = (df["title"] + " " + df["abstract"]).str.lower()
    df = df.copy()
    df["empirical"] = text.apply(lambda t: contains_any(t, EMPIRICAL_SIGNALS))
    for col, keywords in extra_flags.items():
        df[col] = text.apply(lambda t: contains_any(t, keywords))
    return df


def yearly_pct_of_empirical(df, flag_col):
    """% of empirical papers per year where flag_col is True."""
    emp = df[df["empirical"]]
    grp = emp.groupby("year")
    return (grp[flag_col].sum() / grp[flag_col].count() * 100).rename(flag_col)


def plot_trend(series_list, labels, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    styles = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D")]
    for (line, marker), series, label in zip(styles, series_list, labels):
        ax.plot(series.index, series.values, linestyle=line, marker=marker, label=label)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
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
        hw = hw[hw["year"].between(*YEAR_RANGE)]
        hw = classify(hw, {"commercial": COMMERCIAL_PLATFORMS, "custom": CUSTOM_HARDWARE})
        hw.to_csv(os.path.join(DATA_DIR, "hardware_classified.csv"), index=False)
        n_empirical = hw[hw["empirical"]].groupby("year").size()
        print(f"Hardware: {len(hw)} papers, {hw['empirical'].sum()} empirical")
        plot_trend(
            [yearly_pct_of_empirical(hw, "commercial"), yearly_pct_of_empirical(hw, "custom")],
            ["Commercial platform", "Custom/novel hardware"],
            "Hardware: commercial platforms vs. custom builds\n(% of empirical papers per year)",
            "% of empirical papers",
            os.path.join(FIGURES_DIR, "hardware_trend.png"),
        )

    # --- Robotics models ---
    rm = load_group("robotics_models")
    if rm.empty:
        print("No robotics_models data found. Run: python fetch_data.py --groups robotics_models")
    else:
        rm = rm[rm["year"].between(*YEAR_RANGE)]
        rm = classify(rm, {"pretrained": PRETRAINED_SIGNALS, "from_scratch": SCRATCH_SIGNALS})
        rm.to_csv(os.path.join(DATA_DIR, "robotics_models_classified.csv"), index=False)
        print(f"Robotics models: {len(rm)} papers, {rm['empirical'].sum()} empirical")
        plot_trend(
            [yearly_pct_of_empirical(rm, "pretrained"), yearly_pct_of_empirical(rm, "from_scratch")],
            ["Pre-trained / fine-tuned", "Trained from scratch"],
            "Robot learning: pre-trained/fine-tuned vs. trained from scratch\n(% of empirical papers per year)",
            "% of empirical papers",
            os.path.join(FIGURES_DIR, "robotics_models_trend.png"),
        )

    # --- LLM / foundation models ---
    lm = load_group("llm_models")
    if lm.empty:
        print("No llm_models data found. Run: python fetch_data.py --groups llm_models")
    else:
        lm = lm[lm["year"].between(*YEAR_RANGE)]
        lm = classify(lm, {"pretrained": PRETRAINED_SIGNALS, "from_scratch": SCRATCH_SIGNALS})
        lm.to_csv(os.path.join(DATA_DIR, "llm_models_classified.csv"), index=False)
        print(f"LLM models: {len(lm)} papers, {lm['empirical'].sum()} empirical")
        plot_trend(
            [yearly_pct_of_empirical(lm, "pretrained"), yearly_pct_of_empirical(lm, "from_scratch")],
            ["Pre-trained / fine-tuned", "Trained from scratch"],
            "LLM / foundation models: pre-trained/fine-tuned vs. trained from scratch\n(% of empirical papers per year)",
            "% of empirical papers",
            os.path.join(FIGURES_DIR, "llm_models_trend.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
