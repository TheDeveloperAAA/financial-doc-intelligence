import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT)
RESULTS = os.path.join(ROOT, "results")
PROCESSED = os.path.join(ROOT, "data", "processed")

st.set_page_config(page_title="Financial Intelligence Dashboard", page_icon="📊", layout="wide")
st.title("Financial Document Intelligence Pipeline")
st.caption("SEC Filing Analysis | LLM Extraction Evaluation | Fraud Risk Scoring | Earnings Sentiment")

TICKERS = ["AAPL", "MSFT", "JPM", "BAC", "GS", "AMZN", "GOOGL", "META",
           "NVDA", "TSLA", "JNJ", "PFE", "XOM", "WMT", "V"]
st.sidebar.header("Controls")
ticker = st.sidebar.selectbox("Select Company", TICKERS)


@st.cache_data
def load_results() -> dict:
    files = {
        "filings": os.path.join(PROCESSED, "filings_clean.csv"),
        "ingestion_summary": os.path.join(RESULTS, "ingestion_summary.csv"),
        "llm": os.path.join(RESULTS, "llm_accuracy_scorecard.csv"),
        "llm_summary": os.path.join(RESULTS, "llm_eval_summary.csv"),
        "fraud_scores": os.path.join(RESULTS, "fraud_risk_scores.csv"),
        "fraud_importance": os.path.join(RESULTS, "fraud_feature_importance.csv"),
        "fraud_summary": os.path.join(RESULTS, "fraud_model_summary.csv"),
        "sentiment": os.path.join(RESULTS, "sentiment_vs_eps_surprise.csv"),
        "sentiment_summary": os.path.join(RESULTS, "sentiment_correlation_summary.csv"),
    }
    return {k: pd.read_csv(p) if os.path.exists(p) else pd.DataFrame() for k, p in files.items()}


data = load_results()


def first_row(key: str) -> dict:
    df = data[key]
    return df.iloc[0].to_dict() if not df.empty else {}


# Every card reads from results/. Nothing is filled in by hand.
llm_s, fraud_s = first_row("llm_summary"), first_row("fraud_summary")
sent_s, ing_s = first_row("sentiment_summary"), first_row("ingestion_summary")

c1, c2, c3, c4 = st.columns(4)
with c1:
    if llm_s:
        mode = str(llm_s.get("mode"))
        st.metric("LLM Exact Match", f"{llm_s['exact_match_pct']:.1f}%",
                  delta="mock mode: pipeline test only" if mode == "mock" else f"source: {mode}",
                  delta_color="off")
    else:
        st.metric("LLM Exact Match", "not run")
with c2:
    if fraud_s:
        st.metric("Fraud Detection AUC", f"{fraud_s['auc']:.3f}", delta=f"{fraud_s['auc'] - 0.5:+.3f} vs random")
    else:
        st.metric("Fraud Detection AUC", "not run")
with c3:
    if sent_s and pd.notna(sent_s.get("pearson_r")):
        tag = "synthetic transcripts" if bool(sent_s.get("synthetic_transcripts")) else f"n = {int(sent_s['n_samples'])}"
        st.metric("Sentiment vs Next EPS Surprise", f"r = {sent_s['pearson_r']:.2f}", delta=tag, delta_color="off")
    else:
        st.metric("Sentiment vs Next EPS Surprise", "not available")
with c4:
    if ing_s:
        st.metric("Filing Coverage", f"{ing_s['coverage_pct']:.1f}%",
                  delta=f"{int(ing_s['n_parsed'])}/{int(ing_s['n_total'])} filings", delta_color="off")
    else:
        st.metric("Filing Coverage", "not run")

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["SEC Filing Intel", "LLM Accuracy", "Fraud Risk", "Earnings Sentiment"])

with tab1:
    st.subheader(f"Filing Summary: {ticker}")
    df_f = data["filings"]
    if not df_f.empty:
        st.dataframe(df_f[df_f["ticker"] == ticker], use_container_width=True)
    else:
        st.info("Run notebooks/01_ingestion.ipynb to populate filing data.")

with tab2:
    st.subheader("LLM Extraction Evaluation (FinanceBench)")
    df_llm = data["llm"]
    if not df_llm.empty:
        if (df_llm["source"] == "mock").all():
            st.warning("These rows come from the offline mock, not an LLM. Set MISTRAL_API_KEY and rerun notebook 02.")
        errs = df_llm[~df_llm["error_type"].isin(["correct", "api_error"])]["error_type"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        errs.plot(kind="bar", ax=ax, color=["#2196F3", "#FF9800", "#F44336"][: len(errs)])
        ax.set_title("LLM Error Type Distribution")
        ax.set_xlabel("Error Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        st.dataframe(df_llm[["question", "gold_answer", "predicted_answer", "exact_match", "error_type"]],
                     use_container_width=True)
    else:
        st.info("Run notebooks/02_llm_extraction_eval.ipynb to populate LLM eval data.")

with tab3:
    st.subheader("Fraud Risk Scoring (JAR2020, test years 2003-2008)")
    df_imp, df_sc = data["fraud_importance"], data["fraud_scores"]
    if not df_imp.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        d = df_imp.sort_values("importance").tail(10)
        ax2.barh(d["feature"], d["importance"], color="#2196F3")
        ax2.set_title("Top Fraud Signal Features (RandomForest)")
        st.pyplot(fig2)
    if not df_sc.empty:
        st.caption("Highest risk firm-years in the test period (misstate = 1 marks a known AAER case)")
        st.dataframe(df_sc.head(100), use_container_width=True)
    if df_imp.empty and df_sc.empty:
        st.info("Run notebooks/04_fraud_signals.ipynb to populate fraud scores.")

with tab4:
    st.subheader("Earnings Call Sentiment vs EPS Surprise")
    df_s = data["sentiment"]
    if not df_s.empty:
        if df_s.get("synthetic", pd.Series([False])).any():
            st.warning("Sentiment comes from synthetic placeholder transcripts. Add real transcripts to data/raw/transcripts/.")
        d = df_s[df_s["ticker"] == ticker]
        if not d.empty:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.plot(d["period"].astype(str), d["qa_sentiment"], marker="o", label="Q&A Sentiment", color="#2196F3")
            ax3.plot(d["period"].astype(str), d["ceo_sentiment"], marker="s", label="CEO Sentiment", color="#4CAF50")
            ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax3.set_title(f"{ticker}: Earnings Call Sentiment by Section")
            ax3.set_xlabel("Quarter")
            ax3.set_ylabel("VADER Compound Score")
            ax3.legend()
            plt.xticks(rotation=30)
            st.pyplot(fig3)
            st.dataframe(d, use_container_width=True)
        else:
            st.info(f"No sentiment data for {ticker}.")
    else:
        st.info("Run notebooks/03_sentiment_analysis.ipynb to populate sentiment data.")

st.divider()
st.caption("Data: SEC EDGAR | FinanceBench (Patronus AI, 2023) | JAR2020 (Bao et al.) | Yahoo Finance")
st.caption("Author: Aditya Raj | IIT Roorkee | Hudson Labs DS Intern Dec 2025 - Mar 2026")
