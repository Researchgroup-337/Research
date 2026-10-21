PY ?= python

.PHONY: all data tfidf distilbert-smoke synthetic fit evaluate test api demo colab-zip chat bots

all: data tfidf synthetic fit evaluate test

data:            ## 1. clean + split datasets
	$(PY) scripts/prepare_data.py

tfidf:           ## 2a. classical baseline content model (~9 min CPU)
	$(PY) scripts/train_tfidf.py

distilbert-smoke: ## 2b. check the DistilBERT pipeline on CPU (2 min, tiny random model)
	$(PY) scripts/train_distilbert.py --smoke --out /tmp/distilbert_smoke

synthetic:       ## 3. synthetic multi-session worlds (uses DistilBERT if installed, else TF-IDF)
	$(PY) scripts/generate_synthetic.py

fit:             ## 4. fit alpha/beta/gamma, decay and tier thresholds
	$(PY) scripts/fit_risk_model.py

evaluate:        ## 5. full evaluation -> reports/EVALUATION_REPORT.md
	$(PY) scripts/evaluate.py

test:
	$(PY) -m pytest -q

api:             ## run the REST API + dashboard on http://127.0.0.1:8000/dashboard
	BAE_API_KEY=$${BAE_API_KEY:-dev-key-change-me} uvicorn bae.api:app --app-dir src --host 127.0.0.1 --port 8000

demo:
	$(PY) scripts/demo_stream.py --days 14

colab-zip:       ## package code + processed data for the Colab notebook
	cd .. && zip -q -r behavior_engine_colab.zip behavior_engine/src behavior_engine/scripts \
	  behavior_engine/data/processed/jigsaw_*.parquet behavior_engine/data/processed/davidson_eval.parquet \
	  behavior_engine/data/processed/tcm_eval.parquet behavior_engine/requirements.txt

chat:            ## SafeChat Lab test chat app: http://127.0.0.1:8100 (participants), /researcher (console)
	BAE_RESEARCHER_KEY=$${BAE_RESEARCHER_KEY:-lab-key-change-me} uvicorn bae.chatapp.server:app --app-dir src --host 0.0.0.0 --port 8100

bots:            ## run scripted bots in SafeChat Lab (6 sessions x 2 min)
	$(PY) scripts/run_chat_bots.py --sessions 6 --minutes 2
