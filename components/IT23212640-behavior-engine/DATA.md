# Data and model files not stored in this repository

The public datasets are **not committed**, because of their size and their licences. Download them yourself and place them in `data/raw/` with exactly these file names. Then `python scripts/check_setup.py` shows `[ OK ]` for each one.

| File in `data/raw/` | Dataset | Where to get it | Used for |
|---|---|---|---|
| `train.csv` | Jigsaw Toxic Comment Classification Challenge | Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data (download `archive.zip` / the competition files and extract) | Content-model training and validation; part held out for generator pools |
| `test.csv` | Jigsaw (as above) | same archive | Content-model test set and generator pools (with labels) |
| `test_labels.csv` | Jigsaw (as above) | same archive | Labels for `test.csv` (rows labelled −1 are dropped) |
| `davidson.csv` | Davidson et al. (2017), *Hate Speech and Offensive Language* | Kaggle "Hate Speech and Offensive Language Dataset" or https://github.com/t-davidson/hate-speech-and-offensive-language. **Rename** `Hate Speech and Offensive Language Dataset.csv` to `davidson.csv` | Cross-domain test (half) and short-text generator pool (half) |
| `toxic_comment_moderation.xlsx` | Toxic Comment Moderation Dataset (1,050 comments, 308 users) | Kaggle [TBD: exact dataset URL]. **Rename** `Toxic Comment Moderation Dataset.xlsx` to `toxic_comment_moderation.xlsx` | Real-data sanity check (E8) |
| `combined_hate_speech.csv` | Combined Hate Speech Dataset | Kaggle [TBD: exact dataset URL]. **Rename** `Combined Hate Speech Dataset.csv` to `combined_hate_speech.csv` | Only read to check overlap with Davidson; **excluded** from training because it contains the Davidson tweets (leakage), see DECISIONS.md D4 |

After placing the files, rebuild everything (DistilBERT on CPU, about 30–45 min):

```bash
python scripts/prepare_data.py
python scripts/generate_synthetic.py
python scripts/fit_risk_model.py
python scripts/evaluate.py
```

`data/processed/` (the generated parquet files and synthetic worlds) and the local databases (`data/*.db`, `data/safechat/*.db`, `data/safechat/bots.json`) are also excluded. They are recreated by the scripts and apps.

### Small data files that *are* committed
- `data/message_bank.json`: the SafeChat Lab message bank (short messages taken from the Jigsaw and Davidson datasets).
- `data/safechat/pilot_bots_export.zip`: the bots-only pilot export (study codes only, no display names or tokens), used by `scripts/evaluate_collected.py` and `reports/collected/`.

## DistilBERT model

| File | In repo? | Notes |
|---|---|---|
| `models/distilbert/model.quant.onnx` | **yes, via Git LFS** (about 64 MB) | int8-quantised ONNX model the engine uses on CPU (`distilbert_onnx`) |
| `models/distilbert/thresholds.json`, `config.json`, `tokenizer.json`, `tokenizer_config.json` | yes | small files needed to load the model |
| `models/distilbert/model.onnx` (about 256 MB), `model.safetensors` (about 255 MB), `training_args.bin` | **no** | full-precision checkpoint |
| `models/tfidf_lr.joblib` | yes, via Git LFS | TF-IDF + logistic regression baseline / fallback |
| `models/risk_model.json` | yes | fitted risk model and tier boundaries |
| `models/.secret_key` | **never** | HMAC pseudonymisation key; created automatically on first run |

After cloning, run `git lfs install` and `git lfs pull` to download the LFS files.

To re-create the full checkpoint (`model.onnx`, `model.safetensors`), open `notebooks/01_finetune_distilbert_colab.ipynb` in Google Colab with a T4 GPU and run every cell (about 1–1.5 h). Then unzip the resulting `distilbert_model.zip` into this folder so that `models/distilbert/thresholds.json` exists.
