# Robotics research trends investigation

This project analyzes trends in robotics research publications over time, using paper metadata and abstracts from the [arXiv API](https://arxiv.org/help/api/index). It measures two signals of increasing system complexity:

1. **Hardware: build vs. buy** — are papers using off-the-shelf commercial robot platforms (Spot, ANYmal, Unitree, Franka, etc.) or building custom hardware? Tracked across legged, aerial, and manipulation robot papers.

2. **Model training: fine-tune vs. from scratch** — are papers fine-tuning pre-trained models or training from scratch? Tracked separately for robot learning papers and LLM/foundation model papers.

The approach is keyword classification on paper abstracts and titles. Each paper is flagged (non-exclusively) for mentions of commercial platforms, custom-build language, pre-trained model language, or from-scratch training language. The plots show the percentage of papers matching each signal per year.

## Prerequisites

Python 3.9+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Replicating the results

### Step 1: Fetch data

```bash
python fetch_data.py
```

This queries arXiv for papers (2010–2025) across three groups:

| Group | Content | Used for |
|---|---|---|
| `hardware` | Legged, aerial, and manipulation robot papers | Build vs. buy trend |
| `robotics_models` | Robot learning and policy papers | Fine-tune vs. from-scratch (robotics) |
| `llm_models` | LLM and foundation model papers | Fine-tune vs. from-scratch (LLMs) |

Results are cached as JSON files in `data/`. The fetch takes roughly 5–10 minutes.

To fetch only one group:
```bash
python fetch_data.py --groups hardware
python fetch_data.py --groups robotics_models llm_models
```

To preview without hitting the API:
```bash
python fetch_data.py --dry-run
```

### Step 2: Classify with Gemini (recommended)

```bash
export GEMINI_API_KEY=your_key_here
python classify_ml.py
```

This sends paper abstracts to Gemini in batches and assigns each paper one exhaustive label:
- **Hardware papers**: `commercial` / `custom` / `simulation` / `no_hardware`
- **Model papers**: `pretrained` / `scratch` / `no_ml` / `unclear`

Results are cached in `data/ml_classifications.json`. Re-running skips already-classified papers. With ~500 papers per query this takes a few minutes.

To classify only specific groups:
```bash
python classify_ml.py --groups hardware
python classify_ml.py --groups robotics_models llm_models
```

To inspect the raw abstracts for a group before or after classifying:
```bash
python classify_ml.py --groups hardware --print
```

### Step 3: Analyze and plot

```bash
python analyze.py
```

Uses ML classifications if `data/ml_classifications.json` exists, otherwise falls back to keyword matching. Outputs stacked area charts where categories sum to 100% per year:

| File | Content |
|---|---|
| `figures/hardware_trend.png` | Hardware approach breakdown over time |
| `figures/robotics_models_trend.png` | Model training approach for robot learning papers |
| `figures/llm_models_trend.png` | Model training approach for LLM/foundation model papers |
| `data/hardware_classified.csv` | Per-paper classification for hardware group |
| `data/robotics_models_classified.csv` | Per-paper classification for robotics models group |
| `data/llm_models_classified.csv` | Per-paper classification for LLM models group |

## Notes

- Fetch cache: re-running `fetch_data.py` skips queries that already have a file in `data/`. Delete the relevant file to re-fetch.
- Classification cache: `data/ml_classifications.json` persists Gemini results. Delete it to re-classify from scratch.
- The arXiv API is free and requires no API key. The client enforces a 3-second delay between requests automatically.
- Search results are ranked by arXiv relevance — treat trends as indicative, not as exhaustive counts.
- arXiv coverage is thinner before ~2015 (fewer authors posted preprints then), so early data points should be interpreted with more caution.
