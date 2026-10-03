# Financial Document Intelligence Pipeline

**Author:** Aditya Raj | IIT Roorkee | [aditya_r@cy.iitr.ac.in](mailto:aditya_r@cy.iitr.ac.in)
**Context:** Independent extension of work done at Hudson Labs (Dec 2025 - Mar 2026)

## Results

Generated from `results/` by `python src/update_readme.py`. Rows marked "not run yet" fill in after the matching notebook runs.

<!-- RESULTS:START -->
| Module | Metric | Result | Baseline |
|---|---|---|---|
| LLM Extraction Accuracy | Exact match vs ground truth | 6.7% (offline regex baseline, not an LLM); gold figure in evidence 25.3% | GPT-4-Turbo about 19% (FinanceBench paper) |
| LLM Error Type 1 | Unit hallucination | 0.7% of errors | Not established |
| LLM Error Type 2 | Period mismatch | 0.0% of errors | Not established |
| LLM Error Type 3 | Table misread | 99.3% of errors | Not established |
| Fraud Detection AUC | RandomForest on JAR2020, test 2003-2008 | 0.644 | 0.50 random; 0.725 RUSBoost (Bao et al.) |
| Sentiment Signal | Pearson r, QA tone delta vs next-quarter EPS surprise | not available (n = 0 overlapping pairs) | 0 |
| Pipeline Coverage | Filings retrieved and text extracted | 100.0% (84/84) of FinanceBench pre-indexed filings, not live EDGAR | Manual: 0% |
| Ingestion Speed | Per filing automated | 0.009 ms (local JSON parse, no download) | 3+ hours manual (assumed) |
<!-- RESULTS:END -->

## What This Measures

This pipeline quantifies how LLMs fail on financial documents, engineers accounting fraud signals
from SEC enforcement data (Bao et al. 2020, JAR), and tests whether shifts in earnings call Q&A tone
lead EPS surprises.

## Tech Stack

Python 3.10+ | edgartools | VADER | scikit-learn | FinanceBench | Streamlit | yfinance

## Data Sources

- SEC EDGAR: `data.sec.gov` (free, public; requires an identity string, set `EDGAR_IDENTITY`)
- FinanceBench: `github.com/patronus-ai/financebench` (open-source 150 QA pairs)
- JarFraud: `github.com/JarFraud/FraudDetection` (Bao et al. 2020, JAR)
- EPS Data: Yahoo Finance via yfinance (reported EPS vs consensus estimate)
- Transcripts: JSON files in `data/raw/transcripts/` with keys `ticker, quarter, year, text`; synthetic placeholders otherwise

## Run It

```bash
pip install -r requirements.txt
python src/setup_data.py
export EDGAR_IDENTITY="Your Name your@email.com"
export MISTRAL_API_KEY="..."   # optional; without it notebook 02 runs in mock mode
jupyter notebook notebooks/01_ingestion.ipynb
jupyter notebook notebooks/04_fraud_signals.ipynb
jupyter notebook notebooks/02_llm_extraction_eval.ipynb
jupyter notebook notebooks/03_sentiment_analysis.ipynb
python src/update_readme.py
streamlit run dashboard/app.py
```

## Method Notes

- **Fraud model:** 14 Dechow et al. (2011) ratios as provided in the JAR2020 file. Train 1991 to 2001, test 2003 to 2008 with a one-year gap; serial frauds spanning the split are relabelled in training, following the paper.
- **LLM eval:** the gold evidence passage is supplied as context, so this isolates reading and extraction from retrieval. Exact match is strict; a 1% numeric match is reported alongside it. Mock mode exists only so the notebook runs offline and its numbers are not model results.
- **Sentiment:** VADER compound score per section; Q&A delta at quarter t is correlated with EPS surprise at quarter t+1.

## Key References

1. Bao, Ke, Li, Yu, Zhang (2020). Detecting Accounting Fraud in Publicly Traded U.S. Firms Using a Machine Learning Approach. Journal of Accounting Research 58(1). Erratum 2022.
2. Islam et al. (2023). FinanceBench. arXiv:2311.11944.
3. Fin-RATE (2026). LLM Evaluation on SEC Filings. Yale GGLab.

## Scope Note

No deep learning. No fine-tuning. No paid APIs required.
Research-grade pipeline, intermediate Python, fully reproducible.
