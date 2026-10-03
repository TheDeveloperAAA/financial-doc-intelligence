"""
Filing ingestion from the FinanceBench open-source corpus.

sec.gov is not reachable from the build sandbox, so filings are taken from
FinanceBench, which ships passages from real SEC filings (10-K, 10-Q, 8-K) and
earnings releases. One row is produced per unique filing (doc_name).
"""

import json
import re
import time
from collections import Counter

import pandas as pd

SOURCE_NOTE = ("Source: FinanceBench pre-indexed SEC corpus (sec.gov blocked in sandbox); "
               "all 150 QA pairs span real filings")

DOC_NAME_RE = re.compile(r"^(?P<company>.+?)_(?P<period>\d{4}(?:Q\d)?)_(?P<type>10K|10Q|8K|EARNINGS)")
TYPE_MAP = {"10K": "10-K", "10Q": "10-Q", "8K": "8-K", "EARNINGS": "EARNINGS"}


def parse_doc_name(doc_name: str) -> dict:
    """'3M_2023Q2_10Q' -> company 3M, period 2023Q2, filing_type 10-Q."""
    m = DOC_NAME_RE.match(doc_name)
    if not m:
        return {"company": doc_name, "period_end": None, "filing_type": "UNKNOWN"}
    return {"company": m["company"], "period_end": m["period"], "filing_type": TYPE_MAP[m["type"]]}


def _evidence_texts(record: dict) -> list:
    ev = record.get("evidence", [])
    if isinstance(ev, list):
        return [e.get("evidence_text", "") for e in ev if isinstance(e, dict)]
    return [str(ev)]


def ingest_financebench(jsonl_path: str) -> pd.DataFrame:
    """Build one row per unique filing with keyword flags and measured parse time."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    by_doc: dict = {}
    for rec in records:
        by_doc.setdefault(rec["doc_name"], []).append(rec)

    rows = []
    for doc_name, recs in by_doc.items():
        start = time.perf_counter()
        meta = parse_doc_name(doc_name)
        # Unique evidence passages for this filing, joined
        passages = list(dict.fromkeys(t for r in recs for t in _evidence_texts(r) if t))
        text = "\n\n".join(passages)
        low = text.lower()
        rows.append({
            "ticker": recs[0].get("company") or meta["company"],
            "doc_name": doc_name,
            "filing_type": meta["filing_type"],
            "period_end": meta["period_end"],
            "revenue_mentioned": "revenue" in low,
            "eps_mentioned": "earnings per share" in low,
            "parse_success": True,
            "parse_time_sec": round(time.perf_counter() - start, 6),
            "evidence_length": len(text),
            "question_count": len(recs),
        })
    return pd.DataFrame(rows).sort_values(["ticker", "period_end"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    """Corpus statistics. Coverage is 100% by construction for a pre-indexed corpus."""
    n_total = len(df)
    n_parsed = int(df["parse_success"].sum())
    return {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "coverage_pct": round(100 * n_parsed / n_total, 1) if n_total else float("nan"),
        "avg_parse_time_sec": round(float(df["parse_time_sec"].mean()), 6),
        "avg_evidence_length": round(float(df["evidence_length"].mean()), 1),
        "unique_companies": int(df["ticker"].nunique()),
        "total_questions": int(df["question_count"].sum()),
        "filing_types": dict(Counter(df["filing_type"])),
        "source": "financebench",
        "note": SOURCE_NOTE,
    }
