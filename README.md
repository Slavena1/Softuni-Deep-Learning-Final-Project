# Child, Learner, Machine: Benchmarking Real Language Models Against Human Mandarin Acquisition

Testing whether real LLMs (Claude, GPT, Qwen), queried directly about their confidence in Mandarin words, behave more like frequency-driven adult L2 learners or concreteness-driven child L1 learners — and whether more model capacity (dense NNs, pretrained transformer embeddings) actually improves prediction of word-acquisition difficulty in the first place. Builds on the SoftUni Machine Learning Final Project.

## Project structure

```

Softuni-Deep-Learning-Final-Project/
├── data/                      # generated artifacts (embeddings, cost log, LLM response cache) — tracked
│   └── llm_responses/         # cached per-word LLM API responses, one folder per provider/model
├── notebook/
│   └── analysis.ipynb         # the full analysis: baselines, neural nets, LLM API testing, results, discussion
├── src/
│   ├── data_prep.py           # loading + cleaning for all 5 raw sources, and build_dataset()
│   ├── features.py            # spectrum_score()
│   ├── baseline_models.py     # Ridge/Lasso/Random Forest baseline (from the ML project)
│   ├── neural_models.py       # dense NN training/eval, train/val splitting
│   ├── embeddings.py          # frozen MacBERT embedding extraction
│   ├── llm_api.py             # LLM API integration: caching, retries, cost tracking, provider callers
│   └── evaluation.py          # bootstrap CIs, stratified error analysis, spectrum score computation
├── requirements.txt
└── README.md

```

The notebook is the narrative and results layer. All data loading, modeling, and evaluation logic lives in `src/` as reusable, importable functions — it does not live inline in notebook cells.

## Data

Raw source data is downloaded automatically from a public Google Drive folder at the start of the notebook (via `gdown`) — no manual download or personal Drive access needed:

| File | Source |
|---|---|
| `wordbank_mandarin_items.csv` | Wordbank Beijing Mandarin CDI, https://wordbank.stanford.edu |
| `hsk1.csv` ... `hsk6.csv` | HSK vocabulary lists, https://github.com/plaktos/hsk_csv |
| `train-00000-of-00001.parquet` | Hu, M. et al. Chinese BabyLM Challenge corpus |
| `babylm_zh_frequencies.csv` | Precomputed word frequencies from the Chinese BabyLM corpus |
| `Concretenss_Ratings_of_9877_Two_Character_Chinese_Words.xlsx` | Xu & Li (2020) |
| `liu_2007_single_char.txt` | Liu et al. (2007), Chinese Single-character Word Database |

Public folder: https://drive.google.com/drive/folders/13qEDS3YRXskJjXPesqNmPn9uImHmopVl?usp=sharing

These raw files are excluded from version control (see `.gitignore`); generated artifacts that are expensive or costly to reproduce — MacBERT embeddings (`data/*_embeddings.npy`), the LLM API cost log (`data/cost_log.csv`), and cached LLM responses (`data/llm_responses/`) — are committed so reruns don't re-embed or re-charge for API calls.

## Reproducing the analysis

Open `notebook/analysis.ipynb` in Google Colab and run all cells top to bottom. The first cells clone this repo, install requirements, and download the raw data via `gdown` — no manual setup or personal Google Drive access required. API keys for Anthropic, OpenAI, and OpenRouter must be set as Colab secrets (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`) to reproduce Section 4.4; all other sections run without them.

## Methodology notes

Train/test splitting is performed on raw, unscaled features, matching the ML project's approach — scaling is fit only on training data. LLM API calls are cached to disk per word/model on first fetch (`data/llm_responses/`), so reruns cost nothing after the first successful pass; per-call cost is logged to `data/cost_log.csv` for transparency.

## Key finding

Added model capacity — a dense neural network, then pretrained transformer (MacBERT) embeddings — does not improve word-acquisition-difficulty prediction over the original three hand-picked features (frequency, concreteness, word length); for adult HSK data the added complexity reliably hurts, and for child AoA data both neural variants underperform a trivial mean-baseline. Given that, real LLMs (Claude, GPT, Qwen) were tested directly rather than via a proxy: all three lean toward the same frequency-driven confidence pattern as adult L2 learners, though the 200-word sample size limits how precisely their positions can be distinguished from one another or from the human data. See the notebook's Discussion and Limitations sections for the full picture, including which findings hold up under bootstrap resampling and which don't.
