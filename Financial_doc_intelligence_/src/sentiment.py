"""Earnings call sentiment by section and its relation to next-quarter EPS surprise."""

import glob
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

TRANSCRIPT_DATASET_URL = (
    "https://huggingface.co/datasets/ashraq/financial-news/resolve/main/data/train-00000-of-00001.parquet"
)
REQUIRED_TRANSCRIPT_COLS = {"ticker", "quarter", "year", "text"}

SECTION_KEYWORDS = {
    "ceo_remarks": ["we delivered", "i am pleased", "our strategy", "looking ahead"],
    "cfo_remarks": ["revenue", "earnings per share", "operating margin", "guidance"],
    "qa_section": ["question", "analyst", "could you", "can you elaborate"],
}


# ---------- loading ----------

def load_local_transcripts(folder: str) -> list:
    """Read *.json files (each a dict or list of dicts with ticker, quarter, year, text)."""
    out = []
    for path in sorted(glob.glob(str(Path(folder) / "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out.extend(data if isinstance(data, list) else [data])
    return [t for t in out if REQUIRED_TRANSCRIPT_COLS <= set(t)]


def try_load_hf_dataset(url: str = TRANSCRIPT_DATASET_URL, timeout: int = 30) -> tuple:
    """
    Attempt the Hugging Face dataset. Returns (transcripts, reason).
    It is only usable if it carries per-call transcripts with ticker, quarter, year, text.
    """
    import io
    import requests

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content))
    except Exception as e:
        return [], f"download failed ({type(e).__name__})"

    missing = REQUIRED_TRANSCRIPT_COLS - set(df.columns)
    if missing:
        return [], f"not earnings call transcripts; columns are {list(df.columns)[:8]}"
    return df[list(REQUIRED_TRANSCRIPT_COLS)].to_dict("records"), "ok"


CEO_POS = ["We delivered record results this quarter.", "I am pleased with the momentum across every segment.",
           "Our strategy is working and customers are responding well.", "Looking ahead, demand remains strong."]
CEO_NEG = ["This was a challenging quarter for the business.", "We faced headwinds that hurt our results.",
           "Execution fell short of what we expected.", "Looking ahead, we remain cautious about demand."]
CFO_POS = ["Revenue grew strongly and operating margin expanded.", "Earnings per share improved year over year.",
           "We are raising our full-year guidance."]
CFO_NEG = ["Revenue declined and operating margin contracted.", "Earnings per share fell short of last year.",
           "We are lowering our full-year guidance."]
QA_POS = ["Analyst: Could you elaborate on the strong growth? CEO: Yes, we are very confident and excited.",
          "Analyst: Can you elaborate on margins? CFO: Margins improved and we see great opportunity."]
QA_NEG = ["Analyst: Could you explain the weak results? CEO: We are concerned and uncertain about the outlook.",
          "Analyst: Can you elaborate on the decline? CFO: Unfortunately pressure persists and risks are rising."]


def generate_synthetic_transcripts(tickers: list, year: int = 2025, n_quarters: int = 4, seed: int = 42) -> list:
    """
    Structured placeholder transcripts so the notebook runs end to end.
    Tone is random, so correlations computed on these are not evidence of anything.
    """
    rng = random.Random(seed)
    out = []
    for ticker in tickers:
        for q in range(1, n_quarters + 1):
            tone = rng.random()

            def pick(pos, neg, k):
                return " ".join(rng.choice(pos if rng.random() < tone else neg) for _ in range(k))

            text = "\n".join([
                "Operator: Good afternoon and welcome to the earnings call.",
                "Chief Executive Officer: " + pick(CEO_POS, CEO_NEG, 4),
                "Chief Financial Officer: Turning to the financial results. " + pick(CFO_POS, CFO_NEG, 3),
                "Operator: We will now open the floor for questions.",
                pick(QA_POS, QA_NEG, 3),
            ])
            out.append({"ticker": ticker, "quarter": q, "year": year, "text": text, "synthetic": True})
    return out


# ---------- scoring ----------

def split_transcript_sections(text: str) -> dict:
    """Split call text into CEO remarks, CFO remarks and Q&A using marker lines."""
    sections = {"ceo_remarks": "", "cfo_remarks": "", "qa_section": ""}
    current = "ceo_remarks"
    for line in text.split("\n"):
        low = line.lower()
        # Q&A is checked first: once the floor opens, later CFO answers belong to Q&A
        if any(k in low for k in ["q&a", "open the floor", "question-and-answer", "analyst:"]):
            current = "qa_section"
        elif current != "qa_section" and any(k in low for k in ["chief financial", "cfo:", "financial results"]):
            current = "cfo_remarks"
        sections[current] += " " + line
    return sections


def score_sentiment(text: str) -> float:
    """Mean VADER compound score over sentence-sized chunks. Range -1 to +1."""
    if not text or len(text.strip()) < 10:
        return 0.0
    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
    return round(float(np.mean([analyzer.polarity_scores(c)["compound"] for c in chunks])), 4)


def build_sentiment_df(transcripts: list) -> pd.DataFrame:
    """One row per ticker-quarter with section scores and the Q&A quarter-over-quarter delta."""
    rows = []
    for t in transcripts:
        s = split_transcript_sections(t["text"])
        rows.append({
            "ticker": t["ticker"],
            "quarter": int(t["quarter"]),
            "year": int(t["year"]),
            "period": f"{int(t['year'])}Q{int(t['quarter'])}",
            "synthetic": bool(t.get("synthetic", False)),
            "ceo_sentiment": score_sentiment(s["ceo_remarks"]),
            "cfo_sentiment": score_sentiment(s["cfo_remarks"]),
            "qa_sentiment": score_sentiment(s["qa_section"]),
        })
    df = pd.DataFrame(rows).sort_values(["ticker", "year", "quarter"]).reset_index(drop=True)
    df["qa_sentiment_delta"] = df.groupby("ticker")["qa_sentiment"].diff()
    return df


# ---------- EPS ----------

def compute_eps_surprise(tickers: list, limit: int = 12) -> pd.DataFrame:
    """
    Reported EPS vs consensus estimate from yfinance earnings dates.
    The fiscal period is approximated as the calendar quarter that ended before the report date.
    """
    import yfinance as yf

    frames = []
    for ticker in tickers:
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        except Exception:
            continue
        if ed is None or ed.empty or "Reported EPS" not in ed.columns:
            continue
        ed = ed.dropna(subset=["Reported EPS", "EPS Estimate"]).reset_index()
        date_col = ed.columns[0]
        report_date = pd.to_datetime(ed[date_col]).dt.tz_localize(None)
        period_end = (report_date - pd.offsets.QuarterEnd(1))
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "report_date": report_date.dt.date,
            "year": period_end.dt.year,
            "quarter": period_end.dt.quarter,
            "eps_estimate": ed["EPS Estimate"].values,
            "eps_actual": ed["Reported EPS"].values,
            "eps_surprise_pct": ed["Surprise(%)"].values if "Surprise(%)" in ed.columns
            else (100 * (ed["Reported EPS"] - ed["EPS Estimate"]) / ed["EPS Estimate"].abs()).values,
        }))
    if not frames:
        return pd.DataFrame(columns=["ticker", "report_date", "year", "quarter",
                                     "eps_estimate", "eps_actual", "eps_surprise_pct"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "year", "quarter"])


def fetch_eps_quarterly(ticker: str) -> pd.DataFrame:
    """
    Quarterly net income from the yfinance income statement. Its quarter-over-quarter
    % change is used as an earnings proxy: it is NOT a surprise versus consensus.
    """
    import yfinance as yf
    try:
        stmt = yf.Ticker(ticker).quarterly_income_stmt
        if stmt is None or stmt.empty or "Net Income" not in stmt.index:
            return pd.DataFrame()
        net_income = stmt.loc["Net Income"].sort_index()
        period = pd.to_datetime(net_income.index)
        df = pd.DataFrame({
            "ticker": ticker,
            "period": net_income.index.astype(str),
            "year": period.year,
            "quarter": period.quarter,
            "net_income": net_income.values.astype(float),
        })
        df["eps_surprise_pct"] = df["net_income"].pct_change() * 100
        return df.dropna(subset=["eps_surprise_pct"])
    except Exception:
        return pd.DataFrame()


def fetch_eps_for_tickers(tickers: list) -> pd.DataFrame:
    frames = [f for f in (fetch_eps_quarterly(t) for t in tickers) if not f.empty]
    cols = ["ticker", "period", "year", "quarter", "net_income", "eps_surprise_pct"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


# ---------- correlation ----------

def merge_sentiment_eps(sentiment_df: pd.DataFrame, eps_df: pd.DataFrame) -> pd.DataFrame:
    """Attach same-quarter and next-quarter EPS surprise to each sentiment row."""
    df = sentiment_df.copy()
    if eps_df.empty:
        df["eps_surprise_pct"] = np.nan
        df["next_eps_surprise_pct"] = np.nan
        return df
    eps = eps_df[["ticker", "year", "quarter", "eps_surprise_pct"]]
    df = df.merge(eps, on=["ticker", "year", "quarter"], how="left")

    nxt = eps.copy()
    # Shift EPS back one quarter so it lines up with the prior quarter's sentiment
    prev_q = nxt["quarter"] - 1
    nxt["year"] = np.where(prev_q == 0, nxt["year"] - 1, nxt["year"])
    nxt["quarter"] = np.where(prev_q == 0, 4, prev_q)
    nxt = nxt.rename(columns={"eps_surprise_pct": "next_eps_surprise_pct"})
    return df.merge(nxt, on=["ticker", "year", "quarter"], how="left")


def correlate_sentiment_to_eps(merged: pd.DataFrame, min_samples: int = 5) -> dict:
    """Pearson r between Q&A tone delta (quarter t) and EPS surprise (quarter t+1)."""
    pairs = merged.dropna(subset=["qa_sentiment_delta", "next_eps_surprise_pct"])
    n = len(pairs)
    synthetic = bool(merged.get("synthetic", pd.Series([False])).any())
    base = {"n_samples": n, "synthetic_transcripts": synthetic}

    if n < min_samples:
        return {**base, "pearson_r": np.nan, "p_value": np.nan,
                "neg_shift_then_miss": None, "note": f"insufficient overlapping data (n={n})"}

    r, p = stats.pearsonr(pairs["qa_sentiment_delta"], pairs["next_eps_surprise_pct"])
    neg = pairs[pairs["qa_sentiment_delta"] < 0]
    hits = int((neg["next_eps_surprise_pct"] < 0).sum())
    return {**base, "pearson_r": round(float(r), 4), "p_value": round(float(p), 4),
            "neg_shift_then_miss": f"{hits}/{len(neg)}",
            "note": "computed on synthetic transcripts" if synthetic else "computed"}
