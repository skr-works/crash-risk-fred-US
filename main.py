#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crash Risk Monitor (FRED-only, free) — 2-month horizon (42 business days)
- Single-file main.py (no module split)
- Daily run writes latest.json (and optionally opens an Issue when regime == "危機")
- Two-stage decision:
  Stage-1: pre_alert threshold chosen on validation to achieve recall(dd_2m) >= 0.6
  Stage-2: crisis confirmation via gating; gate tuned on validation to achieve precision(dd_2m) >= 0.6

Notes:
- Score models: three separate logistic models
  p_dd_2m, p_crash_1d_fwd (<= -10%), p_crash_30d_fwd (<= -20% / 30BD within 42BD window)
  p_any = 1 - Π(1-p_i)
- Frequency policy A (for 2-month use): only daily/weekly in scoring; monthly is reference-only
- "D→B整列": all series are aligned to business-day index by last-observation-carried-forward (merge_asof),
  so weekly end-dates do not break execution.
"""

import os
import sys
import json
import math
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import requests
import numpy as np
import pandas as pd


# =========================
# Config (edit here)
# =========================

# --- Runtime / output ---
OUTPUT_LATEST_JSON = "latest.json"
DEBUG_IN_LATEST = True  # keep big debug block for now; can turn off after you are satisfied
PRINT_STDOUT_JSON = False  # if True, prints latest.json to stdout (Actions log)

# --- FRED ---
FRED_API_BASE = "https://api.stlouisfed.org/fred"
FRED_API_KEY_ENV = "FRED_API_KEY"
START_DATE = "2000-01-01"  # long history for labeling; split is adaptive if not enough data

# --- Horizon / events ---
HORIZON_BDAYS = 42
DD_LOOKBACK_BDAYS = 252
DD_THRESHOLD = -0.15          # dd_2m: hit -15% from 252d high within 42 BD
CRASH_1D_THRESHOLD = -0.10    # within 42 BD, any 1d return <= -10%
CRASH_30D_THRESHOLD = -0.20   # within 42 BD, any 30BD forward return <= -20%
CRASH_30D_WINDOW = 30         # "30営業日先リターン"

# --- Staleness policy A (fixed delay) ---
STALE_POLICY_A = {
    "daily": 7,
    "weekly": 21,
    "monthly": 120,
    "quarterly": 220,
}

# --- Coverage ---
MIN_COVERAGE_FOR_CONFIDENT_REGIME = 0.80

# --- Model training (simple logistic regression via minibatch GD) ---
TRAINING = {
    "epochs": 900,
    "lr": 0.05,
    "l2": 1.0,
    "batch": 4096,
    "seed": 7,
    "patience": 80,     # early stop on val loss
    "min_delta": 1e-4,
    "clip": 5.0,        # gradient clip
}

# --- Threshold targets (adopted as you requested) ---
TARGETS = {
    "stage1_recall_dd_2m_min": 0.60,     # pre_alert threshold chosen to reach recall >= 0.6 on validation
    "stage2_precision_dd_2m_min": 0.60,  # gate tuned to keep precision >= 0.6 on validation
}

# --- Splits (preferred fixed, but may adapt if insufficient rows) ---
PREFERRED_FIXED_SPLIT = {
    "train_end": "2013-12-31",
    "val_end": "2018-12-31",
    "test_start": "2019-01-01",
}
MIN_SPLIT_SIZES = {"train": 1200, "val": 400, "test": 400}

# --- Regime names ---
REGIMES = ["平常", "注意", "警戒", "早期警戒", "危機"]

# --- FRED series (scoring: daily+weekly only) ---
SERIES = {
    # Weekly (scoring)
    "WEI": {"fred_id": "WEI", "freq_group": "weekly"},
    "ICSA": {"fred_id": "ICSA", "freq_group": "weekly"},
    "CCSA": {"fred_id": "CCSA", "freq_group": "weekly"},
    "NFCI": {"fred_id": "NFCI", "freq_group": "weekly"},
    "STLFSI4": {"fred_id": "STLFSI4", "freq_group": "weekly"},
    # Daily (scoring)
    "T10Y3M": {"fred_id": "T10Y3M", "freq_group": "daily"},
    "HY_OAS": {"fred_id": "BAMLH0A0HYM2", "freq_group": "daily"},
    "IG_OAS": {"fred_id": "BAMLC0A0CM", "freq_group": "daily"},
    "VIX": {"fred_id": "VIXCLS", "freq_group": "daily"},
    "SP500": {"fred_id": "SP500", "freq_group": "daily"},
    # Monthly (reference-only)
    "FEDFUNDS_REF": {"fred_id": "FEDFUNDS", "freq_group": "monthly", "reference_only": True},
    "UNRATE_REF": {"fred_id": "UNRATE", "freq_group": "monthly", "reference_only": True},
}

# --- Gate tuning candidates ---
GATE_X_CANDIDATES = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96]

# --- Optional: GitHub Issues ---
# If regime == "危機", create an Issue (uses default GITHUB_TOKEN in Actions).
ENABLE_GITHUB_ISSUES = True
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_REPOSITORY_ENV = "GITHUB_REPOSITORY"  # "owner/repo"


# =========================
# Helpers
# =========================

def jst_today() -> dt.date:
    # Actions runners are UTC; this gives JST date reliably.
    now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    jst = now_utc.astimezone(dt.timezone(dt.timedelta(hours=9)))
    return jst.date()

def iso(d: dt.date) -> str:
    return d.isoformat()

def safe_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "." or x == "":
            return None
        return float(x)
    except Exception:
        return None

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

def bday_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="B")

def merge_asof_last(left_index: pd.DatetimeIndex, s: pd.Series) -> pd.Series:
    """
    Align series to left_index by last observation carried forward using merge_asof.
    """
    df_left = pd.DataFrame({"date": left_index})
    df_right = pd.DataFrame({"date": s.index, "value": s.values}).sort_values("date")
    out = pd.merge_asof(df_left, df_right, on="date", direction="backward")
    out = out.set_index("date")["value"]
    return out

def compute_stale(asof_date: dt.date, last_obs_date: dt.date, freq_group: str) -> Tuple[int, bool]:
    stale_days = (asof_date - last_obs_date).days
    lim = STALE_POLICY_A.get(freq_group, 999999)
    return stale_days, (stale_days > lim)

def json_dump(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# FRED client
# =========================

def fred_get_series_obs(api_key: str, series_id: str, observation_start: str) -> pd.Series:
    """
    Returns a pandas Series indexed by date with float values.
    """
    url = f"{FRED_API_BASE}/series/observations"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "observation_start": observation_start,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    dates = []
    vals = []
    for o in obs:
        d = o.get("date")
        v = safe_float(o.get("value"))
        if d is None or v is None:
            continue
        dates.append(pd.to_datetime(d))
        vals.append(v)
    if not dates:
        raise RuntimeError(f"Empty series: {series_id}")
    s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
    # Drop duplicate dates keeping last
    s = s[~s.index.duplicated(keep="last")]
    return s


# =========================
# Feature engineering
# =========================

def rolling_std(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).std()

def rolling_max_drawdown(price: pd.Series, window: int) -> pd.Series:
    """
    Rolling max drawdown over a window using prices.
    Output is positive number (e.g. 0.12 for -12%).
    """
    roll_max = price.rolling(window=window, min_periods=window).max()
    dd = (price / roll_max) - 1.0
    mdd = dd.rolling(window=window, min_periods=window).min().abs()
    return mdd

def log_return(price: pd.Series, n: int = 1) -> pd.Series:
    return np.log(price / price.shift(n))

def zscore_train(x: pd.Series, train_mask: pd.Series) -> Tuple[pd.Series, float, float]:
    """
    Compute z-score using mean/std from train_mask==True rows.
    Returns z-scored series and (mean, std).
    """
    train_vals = x[train_mask].dropna()
    if len(train_vals) < 50:
        # too little, still return but with safe std
        m = float(train_vals.mean()) if len(train_vals) else 0.0
        sd = float(train_vals.std(ddof=0)) if len(train_vals) else 1.0
    else:
        m = float(train_vals.mean())
        sd = float(train_vals.std(ddof=0))
    if sd <= 1e-12:
        sd = 1.0
    z = (x - m) / sd
    return z, m, sd

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Input df has aligned columns for:
      WEI, ICSA, CCSA, NFCI, STLFSI4, T10Y3M, HY_OAS, IG_OAS, VIX, SP500
    It may also include reference-only series FEDFUNDS_REF, UNRATE_REF.

    Returns:
      X (feature dataframe), meta dict for feature definitions.
    """
    feat_meta: Dict[str, Dict[str, Any]] = {}

    # --- Weekly: pulse / employment (use 4-week relative changes) ---
    # WEI: 4-week change (level_t - level_t-20BD approx 4 weeks)
    wei = df["WEI"]
    wei_chg_4w = wei - wei.shift(20)
    X_WEI = wei_chg_4w
    feat_meta["WEI"] = {"desc": "WEI 4週差分（20営業日差分で近似）", "source": "WEI"}

    # ICSA / CCSA: log(level / MA4W) and log(level / level_4w)
    icsa = df["ICSA"]
    ccsa = df["CCSA"]
    icsa_ma4 = icsa.rolling(window=20, min_periods=20).mean()
    ccsa_ma4 = ccsa.rolling(window=20, min_periods=20).mean()
    X_ICSA = np.log(icsa / icsa_ma4)
    X_CCSA = np.log(ccsa / ccsa_ma4)
    feat_meta["ICSA"] = {"desc": "ICSA log(現在/4週MA)", "source": "ICSA"}
    feat_meta["CCSA"] = {"desc": "CCSA log(現在/4週MA)", "source": "CCSA"}

    # Additional weekly changes for gate/features
    X_ICSA_CHG_4W = np.log(icsa / icsa.shift(20))
    X_CCSA_CHG_4W = np.log(ccsa / ccsa.shift(20))
    feat_meta["ICSA_CHG_4W"] = {"desc": "ICSA log(現在/20営業日前)", "source": "ICSA"}
    feat_meta["CCSA_CHG_4W"] = {"desc": "CCSA log(現在/20営業日前)", "source": "CCSA"}

    nfci = df["NFCI"]
    fsi = df["STLFSI4"]
    X_NFCI = nfci
    X_STLFSI4 = fsi
    X_NFCI_CHG_4W = nfci - nfci.shift(20)
    X_STLFSI4_CHG_4W = fsi - fsi.shift(20)
    feat_meta["NFCI"] = {"desc": "NFCI 水準", "source": "NFCI"}
    feat_meta["STLFSI4"] = {"desc": "STLFSI4 水準", "source": "STLFSI4"}
    feat_meta["NFCI_CHG_4W"] = {"desc": "NFCI 4週差分（20営業日差分）", "source": "NFCI"}
    feat_meta["STLFSI4_CHG_4W"] = {"desc": "STLFSI4 4週差分（20営業日差分）", "source": "STLFSI4"}

    # --- Daily: rates / credit / market ---
    # Spread: use -T10Y3M so "more inverted" increases risk
    spread = df["T10Y3M"]
    X_T10Y3M = -spread
    X_T10Y3M_CHG_20D = X_T10Y3M - X_T10Y3M.shift(20)
    feat_meta["T10Y3M"] = {"desc": "-(10Y-3M) 水準（逆イールドが上にくる向き）", "source": "T10Y3M"}
    feat_meta["T10Y3M_CHG_20D"] = {"desc": "-(10Y-3M) 20営業日差分", "source": "T10Y3M"}

    # Credit OAS: smooth with 20BD MA; also changes
    hy = df["HY_OAS"]
    ig = df["IG_OAS"]
    hy_ma20 = hy.rolling(window=20, min_periods=20).mean()
    ig_ma20 = ig.rolling(window=20, min_periods=20).mean()
    X_HY = hy_ma20
    X_IG = ig_ma20
    X_HY_CHG_5D = X_HY - X_HY.shift(5)
    X_HY_CHG_20D = X_HY - X_HY.shift(20)
    X_IG_CHG_5D = X_IG - X_IG.shift(5)
    X_IG_CHG_20D = X_IG - X_IG.shift(20)
    X_HY_IG_SPREAD = X_HY - X_IG
    X_HY_IG_CHG_20D = X_HY_IG_SPREAD - X_HY_IG_SPREAD.shift(20)
    feat_meta["HY_OAS"] = {"desc": "HY OAS 20営業日MA", "source": "BAMLH0A0HYM2"}
    feat_meta["IG_OAS"] = {"desc": "IG OAS 20営業日MA", "source": "BAMLC0A0CM"}
    feat_meta["HY_OAS_CHG_5D"] = {"desc": "HY OAS 5営業日差分（MA20ベース）", "source": "BAMLH0A0HYM2"}
    feat_meta["HY_OAS_CHG_20D"] = {"desc": "HY OAS 20営業日差分（MA20ベース）", "source": "BAMLH0A0HYM2"}
    feat_meta["IG_OAS_CHG_5D"] = {"desc": "IG OAS 5営業日差分（MA20ベース）", "source": "BAMLC0A0CM"}
    feat_meta["IG_OAS_CHG_20D"] = {"desc": "IG OAS 20営業日差分（MA20ベース）", "source": "BAMLC0A0CM"}
    feat_meta["HY_IG_SPREAD"] = {"desc": "HY-IG スプレッド（MA20）", "source": "derived"}
    feat_meta["HY_IG_CHG_20D"] = {"desc": "HY-IG スプレッド 20営業日差分", "source": "derived"}

    # VIX: level and 5d change (for gate)
    vix = df["VIX"]
    X_VIX = vix
    X_VIX_CHG_5D = vix - vix.shift(5)
    feat_meta["VIX"] = {"desc": "VIX 水準", "source": "VIXCLS"}
    feat_meta["VIX_CHG_5D"] = {"desc": "VIX 5営業日差分", "source": "VIXCLS"}

    # SP500: returns + short horizon metrics
    spx = df["SP500"]
    X_SP500 = log_return(spx, 1)              # 1d log return
    X_SPX_RET_5D = log_return(spx, 5)
    X_SPX_RET_20D = log_return(spx, 20)
    X_SPX_RVOL_1M = rolling_std(log_return(spx, 1), 21)  # ~1 month
    X_SPX_MDD_1M = rolling_max_drawdown(spx, 21)
    feat_meta["SP500"] = {"desc": "S&P500 1日logリターン", "source": "SP500"}
    feat_meta["SPX_RET_5D"] = {"desc": "S&P500 5営業日logリターン", "source": "derived"}
    feat_meta["SPX_RET_20D"] = {"desc": "S&P500 20営業日logリターン", "source": "derived"}
    feat_meta["SPX_RVOL_1M"] = {"desc": "S&P500 1か月実現ボラ（21日標準偏差）", "source": "derived"}
    feat_meta["SPX_MDD_1M"] = {"desc": "S&P500 1か月最大DD（21日ローリング）", "source": "derived"}

    X = pd.DataFrame(
        {
            # core (kept compatible names for your debug)
            "WEI": X_WEI,
            "ICSA": X_ICSA,
            "CCSA": X_CCSA,
            "T10Y3M": X_T10Y3M,
            "HY_OAS": X_HY,
            "IG_OAS": X_IG,
            "NFCI": X_NFCI,
            "STLFSI4": X_STLFSI4,
            "VIX": X_VIX,
            "SP500": X_SP500,
            "SPX_RVOL_1M": X_SPX_RVOL_1M,
            "SPX_MDD_1M": X_SPX_MDD_1M,
            # extras for recall/gate
            "SPX_RET_5D": X_SPX_RET_5D,
            "SPX_RET_20D": X_SPX_RET_20D,
            "HY_OAS_CHG_5D": X_HY_CHG_5D,
            "HY_OAS_CHG_20D": X_HY_CHG_20D,
            "IG_OAS_CHG_5D": X_IG_CHG_5D,
            "IG_OAS_CHG_20D": X_IG_CHG_20D,
            "HY_IG_SPREAD": X_HY_IG_SPREAD,
            "HY_IG_CHG_20D": X_HY_IG_CHG_20D,
            "NFCI_CHG_4W": X_NFCI_CHG_4W,
            "STLFSI4_CHG_4W": X_STLFSI4_CHG_4W,
            "ICSA_CHG_4W": X_ICSA_CHG_4W,
            "CCSA_CHG_4W": X_CCSA_CHG_4W,
            "VIX_CHG_5D": X_VIX_CHG_5D,
            "T10Y3M_CHG_20D": X_T10Y3M_CHG_20D,
        },
        index=df.index,
    )

    return X, feat_meta


# =========================
# Labeling (events)
# =========================

def label_dd_2m(price: pd.Series) -> pd.Series:
    """
    dd_2m label at t:
      within next HORIZON_BDAYS, reach drawdown <= DD_THRESHOLD from rolling 252d high at each future date.
    Operationally:
      For each future u, compute drawdown_u = price_u / max(price_{u-252..u}) - 1
      dd_2m(t)=1 if exists u in (t+1..t+H) where drawdown_u <= DD_THRESHOLD
    """
    idx = price.index
    roll_max_252 = price.rolling(window=DD_LOOKBACK_BDAYS, min_periods=DD_LOOKBACK_BDAYS).max()
    dd = price / roll_max_252 - 1.0  # negative numbers
    # forward window: check if any dd <= threshold in next H days
    hit = pd.Series(False, index=idx)
    dd_hit = (dd <= DD_THRESHOLD)
    # For each t, look ahead: any dd_hit in (t+1..t+H)
    arr = dd_hit.values.astype(np.int8)
    # Efficient rolling forward max using convolution-like loop
    # Create forward-looking max over horizon by shifting and OR
    out = np.zeros(len(arr), dtype=np.int8)
    for k in range(1, HORIZON_BDAYS + 1):
        shifted = np.zeros(len(arr), dtype=np.int8)
        shifted[:-k] = arr[k:]
        out = np.maximum(out, shifted)
    hit[:] = out.astype(bool)
    return hit.astype(int)

def label_crash_1d_fwd(price: pd.Series) -> pd.Series:
    """
    crash_1d_fwd(t)=1 if within next H, any 1d return <= -10%
    """
    r1 = price.pct_change(1)
    hit1 = (r1 <= CRASH_1D_THRESHOLD)
    arr = hit1.values.astype(np.int8)
    out = np.zeros(len(arr), dtype=np.int8)
    for k in range(1, HORIZON_BDAYS + 1):
        shifted = np.zeros(len(arr), dtype=np.int8)
        shifted[:-k] = arr[k:]
        out = np.maximum(out, shifted)
    return pd.Series(out.astype(bool), index=price.index).astype(int)

def label_crash_30d_fwd(price: pd.Series) -> pd.Series:
    """
    crash_30d_fwd(t)=1 if within next H, any 30BD forward return <= -20%
    i.e., exists u in (t+1..t+H) such that price_{u+30}/price_u -1 <= -20%
    """
    fwd30 = price.shift(-CRASH_30D_WINDOW) / price - 1.0
    hit30 = (fwd30 <= CRASH_30D_THRESHOLD)
    arr = hit30.values.astype(np.int8)
    out = np.zeros(len(arr), dtype=np.int8)
    for k in range(1, HORIZON_BDAYS + 1):
        shifted = np.zeros(len(arr), dtype=np.int8)
        shifted[:-k] = arr[k:]
        out = np.maximum(out, shifted)
    return pd.Series(out.astype(bool), index=price.index).astype(int)


# =========================
# Splitting
# =========================

@dataclass
class Split:
    mode: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

def make_split(index: pd.DatetimeIndex) -> Split:
    """
    Prefer fixed split; fallback to adaptive split if fixed leaves too little data.
    Adaptive split:
      last 400 -> test
      prior 400 -> val
      rest -> train (>=1200)
    """
    test_end = index[-1].date().isoformat()

    # Try fixed split
    train_end = pd.to_datetime(PREFERRED_FIXED_SPLIT["train_end"])
    val_end = pd.to_datetime(PREFERRED_FIXED_SPLIT["val_end"])
    test_start = pd.to_datetime(PREFERRED_FIXED_SPLIT["test_start"])
    if (train_end in index) and (val_end in index) and (test_start in index) and (index[-1] >= test_start):
        train_mask = (index <= train_end)
        val_mask = (index > train_end) & (index <= val_end)
        test_mask = (index >= test_start)
        if train_mask.sum() >= MIN_SPLIT_SIZES["train"] and val_mask.sum() >= MIN_SPLIT_SIZES["val"] and test_mask.sum() >= MIN_SPLIT_SIZES["test"]:
            return Split(
                mode="fixed",
                train_start=index[0].date().isoformat(),
                train_end=train_end.date().isoformat(),
                val_start=(train_end + pd.tseries.offsets.BDay(1)).date().isoformat(),
                val_end=val_end.date().isoformat(),
                test_start=test_start.date().isoformat(),
                test_end=test_end,
            )

    # Adaptive split
    n = len(index)
    test_n = MIN_SPLIT_SIZES["test"]
    val_n = MIN_SPLIT_SIZES["val"]
    train_min = MIN_SPLIT_SIZES["train"]
    if n < (train_min + val_n + test_n):
        raise RuntimeError("Split produced too little data; do not fallback (evaluation would be meaningless).")
    test_start_i = n - test_n
    val_start_i = test_start_i - val_n
    train_end_i = val_start_i - 1

    return Split(
        mode="adaptive",
        train_start=index[0].date().isoformat(),
        train_end=index[train_end_i].date().isoformat(),
        val_start=index[val_start_i].date().isoformat(),
        val_end=index[test_start_i - 1].date().isoformat(),
        test_start=index[test_start_i].date().isoformat(),
        test_end=test_end,
    )

def split_masks(index: pd.DatetimeIndex, sp: Split) -> Tuple[pd.Series, pd.Series, pd.Series]:
    train_end = pd.to_datetime(sp.train_end)
    val_start = pd.to_datetime(sp.val_start)
    val_end = pd.to_datetime(sp.val_end)
    test_start = pd.to_datetime(sp.test_start)
    train_mask = (index <= train_end)
    val_mask = (index >= val_start) & (index <= val_end)
    test_mask = (index >= test_start)
    return train_mask, val_mask, test_mask


# =========================
# Logistic model (simple)
# =========================

@dataclass
class LogitModel:
    features: List[str]
    mean_std: Dict[str, Tuple[float, float]]  # train mean/std per feature
    w: np.ndarray  # weights
    b: float

def prepare_xy(X: pd.DataFrame, y: pd.Series, features: List[str], train_mask: pd.Series) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]], pd.Series]:
    """
    Z-score each feature using train mean/std. Returns dense arrays with NaNs kept (for later filtering).
    """
    zs = {}
    ms = {}
    for f in features:
        z, m, sd = zscore_train(X[f], train_mask)
        zs[f] = z
        ms[f] = (m, sd)
    Z = pd.DataFrame(zs, index=X.index)
    return Z.values.astype(np.float64), y.values.astype(np.float64), ms, Z.isna().any(axis=1)

def filter_rows_for_training(Z: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Z2 = Z[mask]
    y2 = y[mask]
    return Z2, y2

def weighted_logloss(p: np.ndarray, y: np.ndarray, w_pos: float) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    w = np.ones_like(y)
    w[y == 1] = w_pos
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean())

def train_logit(Z: np.ndarray, y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray,
                cfg: Dict[str, Any]) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Train logistic regression with L2 and class weighting using minibatch GD.
    Z includes NaN-free rows already (caller should filter).
    """
    rng = np.random.default_rng(cfg["seed"])
    n, d = Z.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    # class weights
    pos_rate = float(y[train_mask].mean()) if train_mask.sum() else 0.0
    pos_weight = (1 - pos_rate) / max(pos_rate, 1e-9)
    pos_weight = float(np.clip(pos_weight, 1.0, 12.0))

    best = {"loss": float("inf"), "w": None, "b": None, "epoch": 0}
    patience = cfg["patience"]
    min_delta = cfg["min_delta"]
    wait = 0

    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]

    for epoch in range(1, cfg["epochs"] + 1):
        # minibatch shuffle
        rng.shuffle(idx_train)
        for start in range(0, len(idx_train), cfg["batch"]):
            batch_idx = idx_train[start:start + cfg["batch"]]
            Xb = Z[batch_idx]
            yb = y[batch_idx]
            # forward
            logits = Xb @ w + b
            p = sigmoid(logits)
            # weights
            sample_w = np.ones_like(yb)
            sample_w[yb == 1] = pos_weight
            # gradients
            diff = (p - yb) * sample_w
            grad_w = (Xb.T @ diff) / max(len(batch_idx), 1)
            grad_b = float(diff.mean())
            # L2
            grad_w += cfg["l2"] * w
            # clip
            clip = cfg["clip"]
            grad_w = np.clip(grad_w, -clip, clip)
            grad_b = float(np.clip(grad_b, -clip, clip))
            # update
            w -= cfg["lr"] * grad_w
            b -= cfg["lr"] * grad_b

        # val loss
        if len(idx_val) > 0:
            p_val = sigmoid(Z[idx_val] @ w + b)
            loss_val = weighted_logloss(p_val, y[idx_val], pos_weight)
        else:
            loss_val = float("inf")

        if loss_val + min_delta < best["loss"]:
            best = {"loss": float(loss_val), "w": w.copy(), "b": float(b), "epoch": epoch}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best["w"] is None:
        best["w"] = w.copy()
        best["b"] = float(b)
        best["epoch"] = cfg["epochs"]

    stats = {
        "train_pos_rate": pos_rate,
        "pos_weight": float(pos_weight),
        "train_loss_best": float(best["loss"]),
        "epochs": int(best["epoch"]),
        "l2": float(cfg["l2"]),
        "lr": float(cfg["lr"]),
        "batch": int(cfg["batch"]),
    }
    return best["w"], best["b"], stats

def predict_logit(Z: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(Z @ w + b)

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": float(prec), "recall": float(rec)}


# =========================
# Threshold / gate tuning
# =========================

def pick_notice_alert_thresholds(p_any: pd.Series, train_mask: pd.Series) -> Dict[str, float]:
    """
    Keep stable thresholds based on train distribution percentiles.
    """
    base = p_any[train_mask].dropna()
    if len(base) < 200:
        # fallback
        return {"notice_p": float(np.nanpercentile(p_any.dropna(), 60)),
                "alert_p": float(np.nanpercentile(p_any.dropna(), 80))}
    return {
        "notice_p": float(np.nanpercentile(base, 60)),
        "alert_p": float(np.nanpercentile(base, 80)),
    }

def pick_pre_alert_threshold(p_any: pd.Series, y_dd_2m: pd.Series, val_mask: pd.Series, recall_min: float) -> float:
    """
    Choose the smallest threshold that achieves recall>=recall_min on validation for dd_2m.
    """
    p = p_any[val_mask].dropna()
    y = y_dd_2m[val_mask].reindex(p.index)
    if len(p) < 100 or y.sum() == 0:
        # no signal, keep very high threshold
        return float(np.nanpercentile(p_any.dropna(), 95)) if len(p_any.dropna()) else 0.99

    # candidate thresholds: unique percentiles + a grid
    grid = sorted(set([float(np.nanpercentile(p, q)) for q in [50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96, 98]]))
    best = None
    for t in grid:
        pred = (p >= t).astype(int).values
        cm = confusion(y.values.astype(int), pred)
        if cm["recall"] >= recall_min:
            best = t
            break
    if best is None:
        # cannot hit recall_min; choose threshold that maximizes recall (lowest t)
        best = float(np.nanmin(grid)) if grid else float(np.nanpercentile(p, 50))
    return float(best)

def gate_signals(z: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Return the gate input signals as z-scored series (already aligned index).
    We'll gate on "bad tails":
      - credit: HY_OAS_CHG_20D, IG_OAS_CHG_20D, HY_IG_CHG_20D  (high is bad)
      - stress: VIX_CHG_5D, NFCI_CHG_4W, STLFSI4_CHG_4W        (high is bad)
      - market damage: SPX_MDD_1M, SPX_RVOL_1M (high is bad), SPX_RET_20D (low is bad)
    """
    out = {
        "CREDIT_HY_CHG20": z["HY_OAS_CHG_20D"],
        "CREDIT_IG_CHG20": z["IG_OAS_CHG_20D"],
        "CREDIT_HYIG_CHG20": z["HY_IG_CHG_20D"],
        "STRESS_VIX_CHG5": z["VIX_CHG_5D"],
        "STRESS_NFCI_CHG4W": z["NFCI_CHG_4W"],
        "STRESS_FSI_CHG4W": z["STLFSI4_CHG_4W"],
        "MKT_MDD_1M": z["SPX_MDD_1M"],
        "MKT_RVOL_1M": z["SPX_RVOL_1M"],
        "MKT_RET_20D": z["SPX_RET_20D"],
    }
    return out

def compute_tail_thresholds(series: pd.Series, val_mask: pd.Series, x: float, tail: str) -> float:
    """
    tail:
      - "high": threshold = percentile(x)  e.g. x=0.9 means top 10% is bad
      - "low":  threshold = percentile(1-x) (bottom 10% is bad)
    """
    v = series[val_mask].dropna()
    if len(v) < 200:
        v = series.dropna()
    if len(v) == 0:
        return 0.0
    if tail == "high":
        return float(np.nanpercentile(v, x * 100))
    else:
        return float(np.nanpercentile(v, (1 - x) * 100))

def gate_pass_for_day(g: Dict[str, float], row: Dict[str, float]) -> Tuple[bool, List[str], Dict[str, Dict[str, float]]]:
    """
    Evaluate gate for one day using precomputed thresholds.
    Returns passed, reasons, signals (value vs threshold).
    """
    # Helper checks
    def ok_high(key: str) -> bool:
        v = row.get(key, np.nan)
        t = g.get(key, np.nan)
        return (not np.isnan(v)) and (not np.isnan(t)) and (v >= t)

    def ok_low(key: str) -> bool:
        v = row.get(key, np.nan)
        t = g.get(key, np.nan)
        return (not np.isnan(v)) and (not np.isnan(t)) and (v <= t)

    credit_hits = []
    stress_hits = []
    market_hits = []

    # credit
    if ok_high("CREDIT_HY_CHG20"): credit_hits.append("HY_OAS_CHG_20D")
    if ok_high("CREDIT_IG_CHG20"): credit_hits.append("IG_OAS_CHG_20D")
    if ok_high("CREDIT_HYIG_CHG20"): credit_hits.append("HY_IG_CHG_20D")

    # stress
    if ok_high("STRESS_VIX_CHG5"): stress_hits.append("VIX_CHG_5D")
    if ok_high("STRESS_NFCI_CHG4W"): stress_hits.append("NFCI_CHG_4W")
    if ok_high("STRESS_FSI_CHG4W"): stress_hits.append("STLFSI4_CHG_4W")

    # market
    if ok_high("MKT_MDD_1M"): market_hits.append("SPX_MDD_1M")
    if ok_high("MKT_RVOL_1M"): market_hits.append("SPX_RVOL_1M")
    if ok_low("MKT_RET_20D"): market_hits.append("SPX_RET_20D(low)")

    # category satisfaction
    credit_ok = len(credit_hits) > 0
    stress_ok = len(stress_hits) > 0
    market_ok = len(market_hits) > 0

    # rule: (credit & stress) OR (credit & market) OR (two of three categories)
    passed = (credit_ok and stress_ok) or (credit_ok and market_ok) or ((credit_ok + stress_ok + market_ok) >= 2)

    reasons = []
    if credit_ok: reasons.append(f"信用:{','.join(credit_hits)}")
    if stress_ok: reasons.append(f"ストレス:{','.join(stress_hits)}")
    if market_ok: reasons.append(f"市場:{','.join(market_hits)}")

    # signals detail
    signals = {
        "credit": {
            "CREDIT_HY_CHG20": {"value": float(row.get("CREDIT_HY_CHG20", np.nan)), "threshold": float(g.get("CREDIT_HY_CHG20", np.nan))},
            "CREDIT_IG_CHG20": {"value": float(row.get("CREDIT_IG_CHG20", np.nan)), "threshold": float(g.get("CREDIT_IG_CHG20", np.nan))},
            "CREDIT_HYIG_CHG20": {"value": float(row.get("CREDIT_HYIG_CHG20", np.nan)), "threshold": float(g.get("CREDIT_HYIG_CHG20", np.nan))},
        },
        "stress": {
            "STRESS_VIX_CHG5": {"value": float(row.get("STRESS_VIX_CHG5", np.nan)), "threshold": float(g.get("STRESS_VIX_CHG5", np.nan))},
            "STRESS_NFCI_CHG4W": {"value": float(row.get("STRESS_NFCI_CHG4W", np.nan)), "threshold": float(g.get("STRESS_NFCI_CHG4W", np.nan))},
            "STRESS_FSI_CHG4W": {"value": float(row.get("STRESS_FSI_CHG4W", np.nan)), "threshold": float(g.get("STRESS_FSI_CHG4W", np.nan))},
        },
        "market": {
            "MKT_MDD_1M": {"value": float(row.get("MKT_MDD_1M", np.nan)), "threshold": float(g.get("MKT_MDD_1M", np.nan))},
            "MKT_RVOL_1M": {"value": float(row.get("MKT_RVOL_1M", np.nan)), "threshold": float(g.get("MKT_RVOL_1M", np.nan))},
            "MKT_RET_20D": {"value": float(row.get("MKT_RET_20D", np.nan)), "threshold": float(g.get("MKT_RET_20D", np.nan))},
        },
    }

    return passed, reasons, signals

def tune_gate_thresholds(z: pd.DataFrame, p_any: pd.Series, y_dd_2m: pd.Series,
                         val_mask: pd.Series, pre_alert_thr: float,
                         precision_min: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Choose X (tail quantile) from candidates; build gate thresholds on validation distribution.
    Constraint: precision of dd_2m among crisis_confirmed (pre_alert & gate_pass) must be >= precision_min.
    Objective: maximize recall(dd_2m) among crisis_confirmed on validation (while meeting precision constraint).
    """
    sig = gate_signals(z)
    best = None
    best_info = None

    pre_alert = (p_any >= pre_alert_thr) & val_mask

    # Build a compact table for day-wise evaluation
    sig_df = pd.DataFrame({k: v for k, v in sig.items()}, index=z.index)

    for x in GATE_X_CANDIDATES:
        # thresholds:
        # high-tail metrics
        g = {
            "CREDIT_HY_CHG20": compute_tail_thresholds(sig_df["CREDIT_HY_CHG20"], val_mask, x, "high"),
            "CREDIT_IG_CHG20": compute_tail_thresholds(sig_df["CREDIT_IG_CHG20"], val_mask, x, "high"),
            "CREDIT_HYIG_CHG20": compute_tail_thresholds(sig_df["CREDIT_HYIG_CHG20"], val_mask, x, "high"),
            "STRESS_VIX_CHG5": compute_tail_thresholds(sig_df["STRESS_VIX_CHG5"], val_mask, x, "high"),
            "STRESS_NFCI_CHG4W": compute_tail_thresholds(sig_df["STRESS_NFCI_CHG4W"], val_mask, x, "high"),
            "STRESS_FSI_CHG4W": compute_tail_thresholds(sig_df["STRESS_FSI_CHG4W"], val_mask, x, "high"),
            "MKT_MDD_1M": compute_tail_thresholds(sig_df["MKT_MDD_1M"], val_mask, x, "high"),
            "MKT_RVOL_1M": compute_tail_thresholds(sig_df["MKT_RVOL_1M"], val_mask, x, "high"),
            # low-tail metric (bad if very negative)
            "MKT_RET_20D": compute_tail_thresholds(sig_df["MKT_RET_20D"], val_mask, x, "low"),
        }

        # Evaluate gate across validation
        passed = []
        for t in sig_df.index:
            row = sig_df.loc[t].to_dict()
            ok, _, _ = gate_pass_for_day(g, row)
            passed.append(ok)
        gate_pass = pd.Series(passed, index=sig_df.index)

        crisis = pre_alert & gate_pass
        if crisis.sum() == 0:
            continue

        yv = y_dd_2m[crisis].astype(int).values
        pred = np.ones_like(yv)  # crisis implies predicted positive for dd_2m evaluation
        # precision is simply mean(y) on crisis days
        precision = float(yv.mean()) if len(yv) else 0.0
        recall = float((y_dd_2m[crisis].sum()) / max(y_dd_2m[val_mask].sum(), 1)) if y_dd_2m[val_mask].sum() > 0 else 0.0

        info = {
            "x": float(x),
            "crisis_days": int(crisis.sum()),
            "precision_dd_2m_on_crisis": float(precision),
            "recall_dd_2m_on_val": float(recall),
        }

        if precision >= precision_min:
            # objective: maximize recall, tie-breaker fewer crisis days (less noisy)
            if best is None or (recall > best_info["recall_dd_2m_on_val"]) or (math.isclose(recall, best_info["recall_dd_2m_on_val"]) and crisis.sum() < best_info["crisis_days"]):
                best = g
                best_info = info

    # If no candidate satisfies precision constraint, fall back to strictest (0.96)
    if best is None:
        x = 0.96
        sig_df = pd.DataFrame({k: v for k, v in sig.items()}, index=z.index)
        best = {
            "CREDIT_HY_CHG20": compute_tail_thresholds(sig_df["CREDIT_HY_CHG20"], val_mask, x, "high"),
            "CREDIT_IG_CHG20": compute_tail_thresholds(sig_df["CREDIT_IG_CHG20"], val_mask, x, "high"),
            "CREDIT_HYIG_CHG20": compute_tail_thresholds(sig_df["CREDIT_HYIG_CHG20"], val_mask, x, "high"),
            "STRESS_VIX_CHG5": compute_tail_thresholds(sig_df["STRESS_VIX_CHG5"], val_mask, x, "high"),
            "STRESS_NFCI_CHG4W": compute_tail_thresholds(sig_df["STRESS_NFCI_CHG4W"], val_mask, x, "high"),
            "STRESS_FSI_CHG4W": compute_tail_thresholds(sig_df["STRESS_FSI_CHG4W"], val_mask, x, "high"),
            "MKT_MDD_1M": compute_tail_thresholds(sig_df["MKT_MDD_1M"], val_mask, x, "high"),
            "MKT_RVOL_1M": compute_tail_thresholds(sig_df["MKT_RVOL_1M"], val_mask, x, "high"),
            "MKT_RET_20D": compute_tail_thresholds(sig_df["MKT_RET_20D"], val_mask, x, "low"),
        }
        best_info = {"x": float(x), "note": "precision constraint unmet on val; fallback to strictest gate"}

    return best, best_info


# =========================
# GitHub Issues (optional)
# =========================

def create_github_issue_if_needed(title: str, body: str) -> None:
    if not ENABLE_GITHUB_ISSUES:
        return
    token = os.environ.get(GITHUB_TOKEN_ENV, "").strip()
    repo = os.environ.get(GITHUB_REPOSITORY_ENV, "").strip()
    if not token or not repo:
        return
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    payload = {"title": title, "body": body}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code >= 300:
            print(f"[warn] issue create failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[warn] issue create exception: {e}")


# =========================
# Main pipeline
# =========================

def main() -> None:
    asof = jst_today()
    asof_str = iso(asof)
    print(f"[run] date={asof_str} JST")

    api_key = os.environ.get(FRED_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing FRED API key: set env {FRED_API_KEY_ENV}")

    # 1) Fetch series
    raw_series: Dict[str, pd.Series] = {}
    staleness: Dict[str, Dict[str, Any]] = {}

    for key, meta in SERIES.items():
        fred_id = meta["fred_id"]
        s = fred_get_series_obs(api_key, fred_id, START_DATE)

        # last observation date
        last_obs_dt = s.index.max().date()
        stale_days, stale_flag = compute_stale(asof, last_obs_dt, meta["freq_group"])

        staleness[key] = {
            "fred_id": fred_id,
            "freq_group": meta["freq_group"],
            "last_obs_date": last_obs_dt.isoformat(),
            "stale_days": int(stale_days),
            "stale": bool(stale_flag),
        }
        raw_series[key] = s

    # 2) Align to business-day calendar using merge_asof (D→B整列)
    idx = bday_index(START_DATE, asof_str)
    aligned = pd.DataFrame(index=idx)
    for key, meta in SERIES.items():
        s = raw_series[key]
        aligned[key] = merge_asof_last(idx, s)

    # 3) Apply staleness exclusion to scoring series (daily/weekly only; monthly is reference-only anyway)
    scoring_keys = [k for k, m in SERIES.items() if not m.get("reference_only", False)]
    for k in scoring_keys:
        if staleness[k]["stale"]:
            aligned[k] = np.nan

    # 4) Build features
    X_raw, feat_meta = build_features(aligned)

    # 5) Build labels using SP500 (price series)
    price = aligned["SP500"].copy()
    # Price can be NaN early if FRED starts later; ensure labels align
    y_dd_2m = label_dd_2m(price)
    y_c1 = label_crash_1d_fwd(price)
    y_c30 = label_crash_30d_fwd(price)

    # 6) Split
    sp = make_split(X_raw.index)
    train_mask, val_mask, test_mask = split_masks(X_raw.index, sp)

    # 7) Select feature set
    # Keep a compact "core + short-term + change" set:
    feature_keys = [
        "WEI", "ICSA", "CCSA",
        "T10Y3M",
        "HY_OAS", "IG_OAS",
        "NFCI", "STLFSI4",
        "VIX", "SP500",
        "SPX_RVOL_1M", "SPX_MDD_1M",
        "SPX_RET_5D", "SPX_RET_20D",
        "HY_OAS_CHG_5D", "HY_OAS_CHG_20D",
        "IG_OAS_CHG_5D", "IG_OAS_CHG_20D",
        "HY_IG_SPREAD", "HY_IG_CHG_20D",
        "NFCI_CHG_4W", "STLFSI4_CHG_4W",
        "ICSA_CHG_4W", "CCSA_CHG_4W",
        "VIX_CHG_5D",
        "T10Y3M_CHG_20D",
    ]

    # 8) Prepare Z, filter rows with any NaN among chosen features (for training only)
    Z_all, _, mean_std, any_nan = prepare_xy(X_raw, y_dd_2m, feature_keys, train_mask)

    # Require enough rows after NaN filtering
    ok_rows = ~any_nan.values
    if ok_rows.sum() < (MIN_SPLIT_SIZES["train"] + MIN_SPLIT_SIZES["val"] + MIN_SPLIT_SIZES["test"]):
        raise RuntimeError("Not enough data after NaN filtering; check staleness or feature windows.")

    # Masks on filtered matrix
    idx_ok = X_raw.index[ok_rows]
    # remap masks
    train_mask_ok = train_mask[ok_rows].values
    val_mask_ok = val_mask[ok_rows].values
    test_mask_ok = test_mask[ok_rows].values

    Z_ok = Z_all[ok_rows]
    y_dd_ok = y_dd_2m[ok_rows].values.astype(int)
    y_c1_ok = y_c1[ok_rows].values.astype(int)
    y_c30_ok = y_c30[ok_rows].values.astype(int)

    # 9) Train 3 models
    w_dd, b_dd, stats_dd = train_logit(Z_ok, y_dd_ok, train_mask_ok, val_mask_ok, TRAINING)
    w_c1, b_c1, stats_c1 = train_logit(Z_ok, y_c1_ok, train_mask_ok, val_mask_ok, TRAINING)
    w_c30, b_c30, stats_c30 = train_logit(Z_ok, y_c30_ok, train_mask_ok, val_mask_ok, TRAINING)

    # 10) Predict probabilities for ok rows
    p_dd = predict_logit(Z_ok, w_dd, b_dd)
    p_c1 = predict_logit(Z_ok, w_c1, b_c1)
    p_c30 = predict_logit(Z_ok, w_c30, b_c30)
    p_any = 1.0 - (1.0 - p_dd) * (1.0 - p_c1) * (1.0 - p_c30)

    p_dd_s = pd.Series(p_dd, index=idx_ok)
    p_c1_s = pd.Series(p_c1, index=idx_ok)
    p_c30_s = pd.Series(p_c30, index=idx_ok)
    p_any_s = pd.Series(p_any, index=idx_ok)

    # 11) Pick thresholds
    notice_alert = pick_notice_alert_thresholds(p_any_s, train_mask[ok_rows])
    pre_thr = pick_pre_alert_threshold(
        p_any_s, y_dd_2m[ok_rows], val_mask[ok_rows], TARGETS["stage1_recall_dd_2m_min"]
    )

    # 12) Build z-scored feature dataframe for gate (using train mean/std)
    Z_df = pd.DataFrame(Z_ok, index=idx_ok, columns=feature_keys)

    # 13) Tune gate thresholds on validation (precision constraint)
    gate_thr, gate_info = tune_gate_thresholds(
        Z_df, p_any_s, y_dd_2m[ok_rows], val_mask[ok_rows], pre_thr, TARGETS["stage2_precision_dd_2m_min"]
    )

    # 14) Evaluate stage-1 and stage-2 on val/test (for logging)
    def eval_stage_masks(mask_ok: np.ndarray) -> Dict[str, Any]:
        idx_part = idx_ok[mask_ok]
        p_part = p_any_s.loc[idx_part]
        y_part = y_dd_2m.loc[idx_part].astype(int)

        pre_alert = (p_part >= pre_thr)
        # gate evaluation
        sig_df = pd.DataFrame({k: v for k, v in gate_signals(Z_df).items()}, index=Z_df.index).loc[idx_part]
        gate_pass_list = []
        for t in idx_part:
            row = sig_df.loc[t].to_dict()
            ok, _, _ = gate_pass_for_day(gate_thr, row)
            gate_pass_list.append(ok)
        gate_pass = pd.Series(gate_pass_list, index=idx_part)
        crisis = pre_alert & gate_pass

        # Metrics:
        # Stage-1 recall/precision for dd_2m
        y1 = y_part.values
        pred1 = pre_alert.astype(int).values
        cm1 = confusion(y1, pred1)

        # Stage-2 precision/recall for dd_2m (treat crisis as positive prediction)
        if crisis.sum() > 0:
            prec2 = float(y_part[crisis].mean())
        else:
            prec2 = 0.0
        # recall2: how many dd_2m events captured by crisis days / total dd_2m in that split
        rec2 = float(y_part[crisis].sum() / max(y_part.sum(), 1)) if y_part.sum() > 0 else 0.0

        return {
            "days": int(len(idx_part)),
            "base_dd_2m": float(y_part.mean()) if len(y_part) else 0.0,
            "stage1_pre_alert_days": int(pre_alert.sum()),
            "stage1_cm_dd_2m": cm1,
            "stage2_crisis_days": int(crisis.sum()),
            "stage2_precision_dd_2m_on_crisis": float(prec2),
            "stage2_recall_dd_2m_on_split": float(rec2),
        }

    val_eval = eval_stage_masks(val_mask_ok)
    test_eval = eval_stage_masks(test_mask_ok)

    # 15) Build latest (as-of) calculation including coverage and gate status
    # Compute latest features with raw + z (for debug)
    # For as-of, we use aligned raw series at last business day <= asof
    last_bday = idx[-1]
    # Build latest raw features from X_raw (not z)
    latest_feat = X_raw.loc[last_bday, feature_keys]
    # z-score it with train mean/std
    latest_z = {}
    latest_raw = {}
    for f in feature_keys:
        m, sd = mean_std[f]
        val = latest_feat[f]
        latest_raw[f] = float(val) if pd.notna(val) else None
        latest_z[f] = float((val - m) / sd) if pd.notna(val) else None

    # coverage = non-null among feature_keys (as-of)
    non_null = sum(1 for f in feature_keys if latest_raw[f] is not None)
    coverage = non_null / len(feature_keys) if feature_keys else 0.0
    regime_confidence = "ok" if coverage >= MIN_COVERAGE_FOR_CONFIDENT_REGIME else "low"

    # compute as-of probabilities (need z-vector; if any missing, set to NaN)
    x_vec = np.array([latest_z[f] if latest_z[f] is not None else np.nan for f in feature_keys], dtype=np.float64)
    if np.isnan(x_vec).any():
        # fallback: use p_any_s last available (aligned ok rows)
        if last_bday in p_any_s.index:
            p_any_today = float(p_any_s.loc[last_bday])
            p_dd_today = float(p_dd_s.loc[last_bday])
            p_c1_today = float(p_c1_s.loc[last_bday])
            p_c30_today = float(p_c30_s.loc[last_bday])
        else:
            p_any_today = float("nan")
            p_dd_today = float("nan")
            p_c1_today = float("nan")
            p_c30_today = float("nan")
    else:
        p_dd_today = float(sigmoid(np.dot(x_vec, w_dd) + b_dd))
        p_c1_today = float(sigmoid(np.dot(x_vec, w_c1) + b_c1))
        p_c30_today = float(sigmoid(np.dot(x_vec, w_c30) + b_c30))
        p_any_today = float(1.0 - (1.0 - p_dd_today) * (1.0 - p_c1_today) * (1.0 - p_c30_today))

    pre_alert_today = bool((not math.isnan(p_any_today)) and (p_any_today >= pre_thr))

    # gate status today
    gate_ok_today = False
    gate_reasons = []
    gate_signals_detail = {}
    if pre_alert_today:
        # build gate row from z features
        row = {
            "CREDIT_HY_CHG20": latest_z.get("HY_OAS_CHG_20D", np.nan),
            "CREDIT_IG_CHG20": latest_z.get("IG_OAS_CHG_20D", np.nan),
            "CREDIT_HYIG_CHG20": latest_z.get("HY_IG_CHG_20D", np.nan),
            "STRESS_VIX_CHG5": latest_z.get("VIX_CHG_5D", np.nan),
            "STRESS_NFCI_CHG4W": latest_z.get("NFCI_CHG_4W", np.nan),
            "STRESS_FSI_CHG4W": latest_z.get("STLFSI4_CHG_4W", np.nan),
            "MKT_MDD_1M": latest_z.get("SPX_MDD_1M", np.nan),
            "MKT_RVOL_1M": latest_z.get("SPX_RVOL_1M", np.nan),
            "MKT_RET_20D": latest_z.get("SPX_RET_20D", np.nan),
        }
        gate_ok_today, gate_reasons, gate_signals_detail = gate_pass_for_day(gate_thr, row)

    crisis_confirmed_today = bool(pre_alert_today and gate_ok_today)

    # Regime mapping
    notice_p = notice_alert["notice_p"]
    alert_p = notice_alert["alert_p"]

    if math.isnan(p_any_today):
        regime = "平常"
    else:
        if p_any_today < notice_p:
            regime = "平常"
        elif p_any_today < alert_p:
            regime = "注意"
        elif p_any_today < pre_thr:
            regime = "警戒"
        else:
            regime = "早期警戒"
        if crisis_confirmed_today:
            regime = "危機"

    # 16) Build debug drivers (top features by |coef*z|)
    def top_drivers(w: np.ndarray, z_map: Dict[str, Optional[float]], k: int = 6) -> List[Dict[str, Any]]:
        rows = []
        for i, f in enumerate(feature_keys):
            z = z_map.get(f, None)
            if z is None:
                continue
            contrib = float(w[i] * z)
            rows.append((abs(contrib), f, float(z), float(w[i]), contrib))
        rows.sort(reverse=True, key=lambda x: x[0])
        out = []
        for _, f, z, coef, contrib in rows[:k]:
            out.append({"feature": f, "z": z, "coef": coef, "contrib": contrib})
        return out

    # 17) Compose latest.json object
    latest = {
        "asof": asof_str,
        "horizon_bdays": int(HORIZON_BDAYS),

        "p_any": float(p_any_today) if not math.isnan(p_any_today) else None,
        "p_components": {
            "p_dd_2m": float(p_dd_today) if not math.isnan(p_dd_today) else None,
            "p_crash_1d": float(p_c1_today) if not math.isnan(p_c1_today) else None,
            "p_crash_30d": float(p_c30_today) if not math.isnan(p_c30_today) else None,
        },
        "score": float(p_any_today * 100.0) if not math.isnan(p_any_today) else None,

        "regime": regime,
        "regime_confidence": regime_confidence,
        "coverage": float(coverage),

        "thresholds": {
            "notice_p": float(notice_p),
            "alert_p": float(alert_p),
            "pre_alert_p": float(pre_thr),
            "gate": gate_thr,
            "gate_info": gate_info,
        },

        "stage_flags": {
            "pre_alert": bool(pre_alert_today),
            "crisis_confirmed": bool(crisis_confirmed_today),
        },

        "gate_status": {
            "passed": bool(gate_ok_today) if pre_alert_today else False,
            "reasons": gate_reasons if pre_alert_today else [],
            "signals": gate_signals_detail if pre_alert_today else {},
        },

        "data_staleness": staleness,
        "reference_only": {
            "FEDFUNDS_REF": float(aligned["FEDFUNDS_REF"].loc[last_bday]) if pd.notna(aligned["FEDFUNDS_REF"].loc[last_bday]) else None,
            "UNRATE_REF": float(aligned["UNRATE_REF"].loc[last_bday]) if pd.notna(aligned["UNRATE_REF"].loc[last_bday]) else None,
        },

        "definitions": {
            "regimes": REGIMES,
            "events": {
                "dd_2m": f"{HORIZON_BDAYS}営業日以内に直近{DD_LOOKBACK_BDAYS}営業日高値から{int(DD_THRESHOLD*100)}%到達",
                "crash_1d_fwd": f"{HORIZON_BDAYS}営業日以内に(任意のuで)1日リターン<={int(CRASH_1D_THRESHOLD*100)}%",
                "crash_30d_fwd": f"{HORIZON_BDAYS}営業日以内に(任意のuで){CRASH_30D_WINDOW}営業日先リターン<={int(CRASH_30D_THRESHOLD*100)}%",
            },
            "stale_policy_A": STALE_POLICY_A,
            "split": {
                "mode": sp.mode,
                "train_end": sp.train_end,
                "val_end": sp.val_end,
                "test_start": sp.test_start,
                "preferred_fixed": PREFERRED_FIXED_SPLIT,
                "min_sizes": MIN_SPLIT_SIZES,
            },
            "targets": {
                "stage1_recall_dd_2m_min": TARGETS["stage1_recall_dd_2m_min"],
                "stage2_precision_dd_2m_min": TARGETS["stage2_precision_dd_2m_min"],
            },
            "note": "3モデル（dd / 1日-10 / 30日-20）を個別学習し、p_any=1-Π(1-p)で合成。2段階判定：pre_alertはrecall>=0.6、危機はゲートでprecision>=0.6を満たすようvalで調整。",
        },

        "backtest": {
            "windows": {
                "train_start": sp.train_start,
                "train_end": sp.train_end,
                "val_start": sp.val_start,
                "val_end": sp.val_end,
                "test_start": sp.test_start,
                "test_end": sp.test_end,
                "last_label_date": asof_str,
            },
            "model_train_stats": {
                "p_dd_2m": stats_dd,
                "p_crash_1d": stats_c1,
                "p_crash_30d": stats_c30,
            },
            "stage_eval": {
                "val": val_eval,
                "test": test_eval,
            },
        },
    }

    if DEBUG_IN_LATEST:
        latest["debug"] = {
            "latest_features": {
                f: {
                    "raw_level": latest_raw[f],
                    "z": latest_z[f],
                    "desc": feat_meta.get(f, {}).get("desc", ""),
                }
                for f in feature_keys
            },
            "top_drivers": {
                "p_dd_2m": top_drivers(w_dd, latest_z, 8),
                "p_crash_1d": top_drivers(w_c1, latest_z, 8),
                "p_crash_30d": top_drivers(w_c30, latest_z, 8),
            },
            "feature_keys": feature_keys,
        }

    # 18) Write latest.json
    json_dump(latest, OUTPUT_LATEST_JSON)

    if PRINT_STDOUT_JSON:
        print(json.dumps(latest, ensure_ascii=False, indent=2))

    # 19) Optional: Issue creation on "危機"
    if regime == "危機":
        title = f"[危機] Crash risk confirmed ({asof_str} JST)"
        body = (
            f"- asof: {asof_str} (JST)\n"
            f"- p_any: {latest.get('p_any')}\n"
            f"- p_dd_2m: {latest['p_components'].get('p_dd_2m')}\n"
            f"- p_crash_1d: {latest['p_components'].get('p_crash_1d')}\n"
            f"- p_crash_30d: {latest['p_components'].get('p_crash_30d')}\n"
            f"- pre_alert: {latest['stage_flags'].get('pre_alert')}\n"
            f"- gate_passed: {latest['gate_status'].get('passed')}\n"
            f"- gate_reasons: {', '.join(latest['gate_status'].get('reasons', []))}\n\n"
            f"latest.json を参照してください。"
        )
        create_github_issue_if_needed(title, body)

    print(f"[ok] wrote {OUTPUT_LATEST_JSON} regime={regime} coverage={coverage:.2f} confidence={regime_confidence}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Minimal hard-fail with context for Actions log
        print(f"[error] {type(e).__name__}: {e}")
        raise
