# Data

## Reused from the ML project (same provenance, same loaders)
- Wordbank Mandarin AoA data
- HSK vocabulary lists (levels 1-6)
- Chinese BabyLM corpus frequencies (babylm_zh_frequencies.csv)
- Xu & Li (2021) two-character concreteness ratings
- Liu et al. (2007) single-character concreteness ratings

Raw data files above are NOT committed to this repo - see
../src/data_prep.py for the loader functions (ported directly from
the ML project, unchanged), and place the files locally at these
paths to run the notebook.

## New for this project
- llm_responses/ - cached structured responses from each LLM API
  tested, one subfolder per model. These ARE committed, so the
  notebook is reviewable with zero API cost.
- cost_log.csv - running log of every API call made. Also committed.
- Pretrained transformer: hfl/chinese-macbert-base -
  downloaded via transformers, not stored in this repo.
