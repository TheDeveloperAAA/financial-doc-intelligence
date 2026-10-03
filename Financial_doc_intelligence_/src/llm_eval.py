"""
Financial figure extraction evaluation on FinanceBench open-source QA pairs.

Default mode is an offline regex extractor: it reads the question and the gold
evidence passage (never the gold answer) and returns one figure. Optionally,
MISTRAL_API_KEY switches the extractor to Mistral.
"""

import json
import os
import re

import pandas as pd

MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-tiny")

DOLLAR_RE = r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?"
PERCENT_RE = r"[\d,]+(?:\.\d+)?%"
PLAIN_RE = r"\b[\d,]+(?:\.\d+)?\b"
FIGURE_RE = re.compile(f"{DOLLAR_RE}|{PERCENT_RE}|{PLAIN_RE}", re.IGNORECASE)
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}

STOPWORDS = set("""what which when where with that this from have does were their there about into than
the and for are was how much many give response question relying details shown statement
company fiscal year years amount usd millions billions answer using based provide""".split())
PERCENT_HINTS = ("percent", "%", "margin", "ratio", "rate", "growth", "change")


# ---------- loading ----------

def load_financebench(jsonl_path: str) -> list:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evidence_text(record: dict) -> str:
    ev = record.get("evidence", "")
    if isinstance(ev, list):
        return "\n\n".join(e.get("evidence_text", "") for e in ev if isinstance(e, dict))
    return str(ev)


# ---------- normalisation ----------

def normalize_answer(text) -> str:
    """Lowercase, strip $ and commas, collapse whitespace."""
    text = str(text).lower().strip()
    text = re.sub(r"[\$,]", "", text)
    return re.sub(r"\s+", " ", text).rstrip(".")


def normalize_figure(text) -> float | None:
    """'$1,577.00' -> 1577.0, '$2.1 billion' -> 2.1e9, '24.26%' -> 24.26. First figure only."""
    if text is None:
        return None
    m = FIGURE_RE.search(str(text))
    if not m:
        return None
    raw = m.group().lower()
    num = re.search(r"[\d,]*\.?\d+", raw.replace(",", ""))
    if not num:
        return None
    value = float(num.group())
    for word, mult in SCALE.items():
        if word in raw:
            value *= mult
    return value


def figures_equal(a, b) -> bool:
    va, vb = normalize_figure(a), normalize_figure(b)
    if va is None or vb is None:
        return False
    return round(va, 2) == round(vb, 2)


def significant_digits(text) -> str:
    """Digits only, leading and trailing zeros removed: '$1577.00' and '1,577' -> '1577'."""
    m = FIGURE_RE.search(str(text))
    if not m:
        return ""
    return re.sub(r"\D", "", m.group()).strip("0")


def numeric_match(gold, predicted) -> bool:
    """Scale-agnostic: same significant digit string."""
    g, p = significant_digits(gold), significant_digits(predicted)
    return bool(g) and g == p


# ---------- extraction ----------

def extract_candidates(evidence: str) -> list:
    """All dollar amounts, percentages and plain numbers, with the line they sit on."""
    out = []
    for line_no, line in enumerate(evidence.split("\n")):
        for m in FIGURE_RE.finditer(line):
            raw = m.group().strip()
            value = normalize_figure(raw)
            if value is None:
                continue
            out.append({"raw": raw, "value": value, "line": line, "line_no": line_no,
                        "is_percent": raw.endswith("%"), "is_dollar": raw.startswith("$")})
    return out


MONTH_DAY_RE = re.compile(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\b", re.I)


def _is_year_like(c: dict) -> bool:
    """Years and calendar days ('December 31') are labels, not reported figures."""
    if c["is_dollar"] or c["is_percent"] or not float(c["value"]).is_integer() or "," in c["raw"]:
        return False
    if 1990 <= c["value"] <= 2035:
        return True
    return any(m.group(1) == c["raw"] for m in MONTH_DAY_RE.finditer(c["line"]))


def _question_terms(question: str) -> set:
    words = re.findall(r"[a-z][a-z&\-]{3,}", question.lower())
    return {w for w in words if w not in STOPWORDS}


def extract_answer(question: str, evidence: str) -> str:
    """
    Gold-blind heuristic: score each candidate by overlap between the question's
    content words and the candidate's line; skip year-like numbers; prefer
    percentages when the question asks for a rate or ratio. Ties go to the earliest.
    """
    cands = extract_candidates(evidence)
    if not cands:
        return ""
    terms = _question_terms(question)
    wants_pct = any(h in question.lower() for h in PERCENT_HINTS)

    def score(c):
        overlap = len(terms & _question_terms(c["line"]))
        type_bonus = 0.5 if (wants_pct and c["is_percent"]) or (not wants_pct and not c["is_percent"]) else 0
        return (overlap + type_bonus, -c["line_no"])

    pool = [c for c in cands if not _is_year_like(c)] or cands
    return max(pool, key=score)["raw"]


def gold_in_evidence(gold: str, evidence: str) -> bool:
    """Oracle check: does the gold figure appear anywhere in the evidence? Uses the answer key."""
    g = normalize_figure(gold)
    if g is None:
        return False
    return any(round(c["value"], 2) == round(g, 2) for c in extract_candidates(evidence))


# ---------- scoring ----------

def classify_error(gold: str, predicted: str) -> str:
    """
    Type1 unit hallucination: same significant digits, different scale or unit
    Type2 period mismatch: gold cites a year and the prediction cites a different one
    Type3 table misread: everything else (wrong figure, wrong row, or nothing extracted)
    """
    if figures_equal(gold, predicted):
        return "correct"
    if numeric_match(gold, predicted):
        return "Type1_unit_hallucination"
    gold_years = set(re.findall(r"20\d{2}", str(gold)))
    pred_years = set(re.findall(r"20\d{2}", str(predicted)))
    if gold_years and pred_years and not (gold_years & pred_years):
        return "Type2_period_mismatch"
    return "Type3_table_misread"


def query_mistral(question: str, context: str, client) -> str:
    from mistralai.models.chat_completion import ChatMessage
    prompt = (
        "You are a financial analyst. Answer the question using only the provided context.\n\n"
        f"Context: {context[:6000]}\n\nQuestion: {question}\n\n"
        "Answer with the specific number or value only. Be precise."
    )
    try:
        resp = client.chat(model=MISTRAL_MODEL, messages=[ChatMessage(role="user", content=prompt)])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"API_ERROR: {type(e).__name__}"


def evaluate_extraction(records: list, api_key: str | None = None, progress: bool = True) -> pd.DataFrame:
    """One row per question. Regex extractor unless an API key is given."""
    client, source = None, "regex"
    if api_key:
        from mistralai.client import MistralClient
        client, source = MistralClient(api_key=api_key), "mistral"

    it = records
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(records, desc="Evaluating")

    rows = []
    for rec in it:
        question, gold = rec.get("question", ""), str(rec.get("answer", ""))
        ev = evidence_text(rec)
        predicted = query_mistral(question, ev, client) if client else extract_answer(question, ev)
        is_err = predicted.startswith("API_ERROR")
        rows.append({
            "financebench_id": rec.get("financebench_id"),
            "company": rec.get("company"),
            "doc_name": rec.get("doc_name"),
            "question_type": rec.get("question_type"),
            "question": question,
            "gold_answer": gold,
            "predicted_answer": predicted,
            "source": "api_error" if is_err else source,
            "model": MISTRAL_MODEL if source == "mistral" else "regex_baseline",
            "exact_match": figures_equal(gold, predicted),
            "numeric_match": numeric_match(gold, predicted),
            "gold_in_evidence": gold_in_evidence(gold, ev),
            "error_type": "api_error" if is_err else classify_error(gold, predicted),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """
    Recall    = correct / all questions
    Precision = correct / questions where the extractor returned something
    answer_recoverability_pct = gold figure present in the evidence (an upper bound for any extractor)
    """
    scored = df[df["source"] != "api_error"]
    n = len(scored)
    attempted = scored[scored["predicted_answer"].str.len() > 0]
    correct = int(scored["exact_match"].sum())
    errors = scored[scored["error_type"] != "correct"]["error_type"].value_counts(normalize=True) * 100
    return {
        "mode": ",".join(sorted(scored["source"].unique())) if n else "none",
        "n_questions": n,
        "n_api_errors": int((df["source"] == "api_error").sum()),
        "exact_match_pct": round(100 * correct / n, 1) if n else float("nan"),
        "numeric_match_pct": round(100 * scored["numeric_match"].mean(), 1) if n else float("nan"),
        "precision_pct": round(100 * correct / len(attempted), 1) if len(attempted) else float("nan"),
        "recall_pct": round(100 * correct / n, 1) if n else float("nan"),
        "answer_recoverability_pct": round(100 * scored["gold_in_evidence"].mean(), 1) if n else float("nan"),
        "error_breakdown_pct": {k: round(v, 1) for k, v in errors.items()},
    }
