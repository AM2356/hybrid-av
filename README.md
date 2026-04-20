# hybrid-av
# Hybrid-AV - Cost-Aware Hybrid Scoring Prototype (Static + Behaviour + Gated Hybrid)

This repository contains a prototype exploring **hybrid anti-malware scoring** as a *second stage* after signature scanning.

The work is organised into three main pipelines:

1) **Static (Drebin, Android)**: binary classification **Benign vs Suspicious/Malicious** using 215D feature vectors with RF and XGBoost baselines.  
2) **Behaviour (CAPE, Windows)**: **10-family classification** from sandbox reports by extracting API-like tokens and training a TF-IDF + RandomForest classifier.  
3) **Hybrid (CAPE dual-view)**: **cheap-first, expensive-when-needed** scoring using two TF-IDF views (cheap vs full) and fusion strategies (late, gated late, early).

A design-level integration demo is included using **ClamAV** as stage-1 (signature scan), then calling the hybrid scorer only when ClamAV does **not** return `FOUND`.

---

## Repository structure
HYBRID-AV/
├─ notebooks/
│  ├─ 01_drebin_static_baseline.ipynb
│  ├─ 02_cape_behavior_baseline.ipynb
│  └─ 03_hybrid_scoring.ipynb
├─ saved/models/
│  ├─ drebin_static_rf_best.pkl
│  ├─ drebin_static_xgb_best.pkl
│  ├─ cape_behavior_rf_best.pkl
│  ├─ cape_behavior_tfidf.pkl
│  ├─ cape_rf_cheap.pkl
│  ├─ cape_tfidf_cheap.pkl
│  ├─ cape_rf_full.pkl
│  ├─ cape_tfidf_full.pkl
│  └─ cape_early_fusion_rf.pkl
├─ src/
│  ├─ artifacts.py
│  ├─ cape_data.py
│  ├─ hybrid.py
│  └─ hybrid_scorer.py
├─ scripts/
│  └─ clamav_hybrid_demo.py
├─ test_samples/
│  ├─ EICAR_safe_malware.txt
│  ├─ sample1_emotet_like.txt
│  ├─ sample2_qakbot_like.txt
│  └─ sample3_clean_like.txt
├─ requirements.txt
└─ README.md

---

## Quick start

### 1) Environment setup

Create a virtual environment and install dependencies:


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

* The notebooks use: NumPy, pandas, scikit-learn, xgboost, matplotlib, scipy, joblib.
* The ClamAV demo requires clamscan installed (see below)

Data setup

Drebin (static)

Expected file:

* data/drebin/drebin215dataset5560malware9476benign.csv

This CSV contains 215 engineered static features plus a label column class:

* B = benign
* S = suspicious/malicious

CAPE (behaviour)

Expected files:

* data/cape/public_labels.csv
* data/cape/public_small_reports/<sha256>.json (CAPE report JSONs)

The CAPE loader extracts API-like tokens from multiple possible report layouts:

* behavior.processes[*].calls[*].api / call
* behavior.apistats (expanded by counts)
* behavior.summary (recursive token collection)
* fallback: static.pe_imports / static.imports

A minimum trace length filter is applied:

* min_api_len = 10

Running the notebooks (recommended)

Launch Jupyter and run notebooks in order:

1. Static baseline - notebooks/01_drebin_static_baseline.ipynb
    * Cleans the Drebin CSV
    * Stratified split 60/20/20 (seed=42)
    * Trains RF + XGB baselines
    * Robustness: top-k feature removal (obfuscation stress test)
    * Saves: drebin_static_rf_best.pkl, drebin_static_xgb_best.pkl
2. Behaviour baseline - notebooks/02_cape_behavior_baseline.ipynb
    * Builds cape_df from labels + JSON reports
    * Stratified split 60/20/20 (seed=42)
    * TF-IDF (1–2 grams, max_features=5000) + RF classifier
    * Short-run robustness: budgeted truncation (50/100/200/full)
    * Saves: cape_behavior_rf_best.pkl, cape_behavior_tfidf.pkl
3. Hybrid scoring - notebooks/03_hybrid_scoring.ipynb
    * CAPE dual-view setup:
        * cheap view: unigram TF-IDF, 1000 features
        * full view: 1–2gram TF-IDF, 5000 features
    * Trains rf_cheap and rf_full
    * Fusion strategies:
        * late fusion (probability blending)
        * gated late fusion (confidence gate)
        * early fusion (feature concat + RF)
    * Measures prediction-only and end-to-end timings
    * Implements “true gated end-to-end” (compute full TF-IDF only when needed)
    * Saves: cape_tfidf_cheap.pkl, cape_rf_cheap.pkl, cape_tfidf_full.pkl, cape_rf_full.pkl, cape_early_fusion_rf.pkl

Running the ClamAV and Hybrid demo

1) Install ClamAV (macOS)

Using Homebrew:
brew install clamav
freshclam
clamscan --version

2) Run the demo script

The demo performs:

* Stage 1: clamscan --no-summary <file>
* If verdict is not FOUND, run the hybrid scorer on extracted token text
* Print a structured JSON result per file

Run on these samples:
python3 scripts/clamav_hybrid_demo.py test_samples/

Example output fields:

* clamav_verdict: FOUND / CLEAN / UNKNOWN
* hybrid_used: whether hybrid was executed
* hybrid: predicted family + confidence + risk_label

Important: extract_api_text_from_file() is design-level and treats the input file as a space-separated token sequence. Real deployment would extract tokens from sandbox logs or a behavioural monitor.

Saved artefacts

Models and vectorisers are stored in saved/models/:

* Drebin:
    * drebin_static_rf_best.pkl
    * drebin_static_xgb_best.pkl
* CAPE behaviour:
    * cape_behavior_tfidf.pkl
    * cape_behavior_rf_best.pkl
* CAPE hybrid dual-view:
    * cape_tfidf_cheap.pkl, cape_rf_cheap.pkl
    * cape_tfidf_full.pkl, cape_rf_full.pkl
    * cape_early_fusion_rf.pkl

Loading helpers live in src/artifacts.py.

Design notes (what “hybrid” means here)

This project’s hybrid implementation is CAPE-only dual-view:

* Cheap “static-like” view: TF-IDF unigrams, 1000 features
* Full behavioural view: TF-IDF 1–2 grams, 5000 features
* Gated fusion: run cheap first; only run the full branch when cheap confidence is below a threshold (default 0.9).

It does not fuse Drebin with CAPE (different platforms/tasks).

Limitations (scope honesty)

* Drebin and CAPE represent different tasks (Android B/S vs Windows family classification).
* LOFO results reflect a closed-set classifier (no “unknown family” label).
* The ClamAV integration is design-level: it demonstrates control flow + JSON output, not full feature extraction from real binaries or direct sandbox ingestion.
