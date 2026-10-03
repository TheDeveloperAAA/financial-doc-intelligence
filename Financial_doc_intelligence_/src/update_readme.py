"""Rewrite the README results table from the CSVs in results/. Run after the notebooks."""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.paths import RESULTS, ROOT

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
NOT_RUN = "not run yet"


def row(name: str) -> dict:
    p = RESULTS / f"{name}.csv"
    return pd.read_csv(p).iloc[0].to_dict() if p.exists() else {}


def build_table() -> str:
    llm, fraud = row("llm_eval_summary"), row("fraud_model_summary")
    sent, ing = row("sentiment_correlation_summary"), row("ingestion_summary")

    if llm:
        mode = llm.get("mode")
        tag = {"mock": " (mock mode, not an LLM result)",
               "regex": " (offline regex baseline, not an LLM)"}.get(mode, f" ({mode})")
        if "answer_recoverability_pct" in llm:
            tag += f"; gold figure in evidence {llm['answer_recoverability_pct']}%"
        llm_acc = f"{llm['exact_match_pct']}%{tag}"
        t1 = f"{llm.get('err_Type1_unit_hallucination', 0.0)}% of errors"
        t2 = f"{llm.get('err_Type2_period_mismatch', 0.0)}% of errors"
        t3 = f"{llm.get('err_Type3_table_misread', 0.0)}% of errors"
    else:
        llm_acc = t1 = t2 = t3 = NOT_RUN

    fraud_auc = f"{fraud['auc']:.3f}" if fraud else NOT_RUN
    if sent and pd.notna(sent.get("pearson_r")):
        sent_r = f"{sent['pearson_r']:.2f} (p = {sent['p_value']:.2f}, n = {int(sent['n_samples'])})"
        if str(sent.get("synthetic_transcripts")) == "True":
            sent_r += ", synthetic transcripts"
    else:
        sent_r = f"not available (n = {int(sent['n_samples'])} overlapping pairs)" if sent else NOT_RUN
    corpus = ing.get("source") == "financebench"
    cov = (f"{ing['coverage_pct']}% ({int(ing['n_parsed'])}/{int(ing['n_total'])})"
           + (" of FinanceBench pre-indexed filings, not live EDGAR" if corpus else "")) if ing else NOT_RUN
    spd = (f"{ing['avg_parse_time_sec'] * 1000:.3f} ms (local JSON parse, no download)" if corpus
           else f"{ing['avg_parse_time_sec']} sec") if ing else NOT_RUN

    lines = [
        "| Module | Metric | Result | Baseline |",
        "|---|---|---|---|",
        f"| LLM Extraction Accuracy | Exact match vs ground truth | {llm_acc} | GPT-4-Turbo about 19% (FinanceBench paper) |",
        f"| LLM Error Type 1 | Unit hallucination | {t1} | Not established |",
        f"| LLM Error Type 2 | Period mismatch | {t2} | Not established |",
        f"| LLM Error Type 3 | Table misread | {t3} | Not established |",
        f"| Fraud Detection AUC | RandomForest on JAR2020, test 2003-2008 | {fraud_auc} | 0.50 random; 0.725 RUSBoost (Bao et al.) |",
        f"| Sentiment Signal | Pearson r, QA tone delta vs next-quarter EPS surprise | {sent_r} | 0 |",
        f"| Pipeline Coverage | Filings retrieved and text extracted | {cov} | Manual: 0% |",
        f"| Ingestion Speed | Per filing automated | {spd} | 3+ hours manual (assumed) |",
    ]
    return "\n".join(lines)


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END),
                 START + "\n" + build_table() + "\n" + END, text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    print(build_table())


if __name__ == "__main__":
    main()
