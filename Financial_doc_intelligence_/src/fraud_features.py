"""Fraud signal model on the JAR2020 dataset (Bao, Ke, Li, Yu, Zhang 2020)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# The 14 financial ratios from Dechow et al. (2011) as included in data_FraudDetection_JAR2020.csv
FRAUD_FEATURES = [
    "dch_wc",       # change in working capital accruals
    "ch_rsst",      # RSST accruals
    "dch_rec",      # change in receivables
    "dch_inv",      # change in inventory
    "soft_assets",  # soft assets share of total assets
    "ch_cs",        # change in cash sales
    "ch_cm",        # change in cash margin
    "ch_roa",       # change in return on assets
    "issue",        # actual issuance of securities
    "bm",           # book-to-market
    "dpi",          # depreciation index
    "reoa",         # retained earnings over total assets
    "EBIT",         # EBIT over total assets
    "ch_fcf",       # change in free cash flow
]
LABEL_COL = "misstate"
ID_COLS = ["gvkey", "fyear", "p_aaer"]

# Mirrors the paper's out-of-sample design: train on earlier years, test on 2003-2008,
# leaving a one-year gap so fraud disclosed later cannot leak into training labels.
TRAIN_YEARS = (1991, 2001)
TEST_YEARS = (2003, 2008)


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    missing = [c for c in FRAUD_FEATURES + [LABEL_COL, "fyear"] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from dataset: {missing}")
    return df


def load_and_prepare(filepath: str) -> tuple:
    """Return X, y with rows that have all 14 features and the label."""
    df = load_dataset(filepath)[FRAUD_FEATURES + [LABEL_COL]].dropna()
    return df[FRAUD_FEATURES], df[LABEL_COL].astype(int)


def train_test_split_by_year(df_full: pd.DataFrame,
                             train_years: tuple = TRAIN_YEARS,
                             test_years: tuple = TEST_YEARS) -> tuple:
    """
    Temporal split. Serial frauds that span the split (same p_aaer case id in both
    periods) are relabelled 0 in training, as in Bao et al., to avoid label leakage.
    """
    df = df_full.copy()
    train = df[df["fyear"].between(*train_years)].copy()
    test = df[df["fyear"].between(*test_years)].copy()

    if "p_aaer" in df.columns:
        test_cases = set(test.loc[test[LABEL_COL] == 1, "p_aaer"].dropna())
        serial = train["p_aaer"].isin(test_cases) & (train[LABEL_COL] == 1)
        train.loc[serial, LABEL_COL] = 0

    keep = [c for c in ID_COLS if c in df.columns] + FRAUD_FEATURES + [LABEL_COL]
    train = train[keep].dropna(subset=FRAUD_FEATURES + [LABEL_COL])
    test = test[keep].dropna(subset=FRAUD_FEATURES + [LABEL_COL])
    return train, test


def run_fraud_model(filepath: str, random_state: int = 42) -> dict:
    """Train RandomForest on the temporal split and score the test years."""
    df = load_dataset(filepath)
    train_df, test_df = train_test_split_by_year(df)

    X_train, y_train = train_df[FRAUD_FEATURES], train_df[LABEL_COL].astype(int)
    X_test, y_test = test_df[FRAUD_FEATURES], test_df[LABEL_COL].astype(int)

    if y_test.nunique() < 2 or y_train.nunique() < 2:
        raise ValueError("Train or test split has a single class; AUC is undefined.")

    # Trees are scale invariant, so no StandardScaler
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=8,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    yearly_auc = {}
    for year, grp in test_df.assign(score=y_proba).groupby("fyear"):
        if grp[LABEL_COL].nunique() == 2:
            yearly_auc[int(year)] = round(roc_auc_score(grp[LABEL_COL], grp["score"]), 4)

    importance_df = (
        pd.DataFrame({"feature": FRAUD_FEATURES, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    scores_df = test_df[[c for c in ID_COLS if c in test_df.columns] + [LABEL_COL]].copy()
    scores_df["fraud_score"] = np.round(y_proba, 6)
    scores_df["risk_rank"] = scores_df["fraud_score"].rank(ascending=False, method="first").astype(int)
    scores_df = scores_df.sort_values("risk_rank").reset_index(drop=True)

    # Share of true frauds caught in the top 1% of scores, a practical screening metric
    top_n = max(1, int(len(scores_df) * 0.01))
    frauds_total = int(y_test.sum())
    frauds_top1 = int(scores_df.head(top_n)[LABEL_COL].sum())

    return {
        "auc": round(auc, 4),
        "yearly_auc": yearly_auc,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_years": TRAIN_YEARS,
        "test_years": TEST_YEARS,
        "fraud_rate_train": round(y_train.mean(), 4),
        "fraud_rate_test": round(y_test.mean(), 4),
        "frauds_in_test": frauds_total,
        "frauds_in_top1pct": frauds_top1,
        "top_features": importance_df.head(3)["feature"].tolist(),
        "feature_importance_df": importance_df,
        "scores_df": scores_df,
        "model": rf,
    }
