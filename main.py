import os
import json
import time
import gzip
import math
import datetime as dt
from typing import Dict, Tuple, Optional, List

import requests
import pandas as pd
import numpy as np


# =========================
# Spec (2 months horizon)
# =========================

START_DATE = "2000-01-01"
RUN_TZ_LABEL = "JST"

# keep history for monitoring convenience
HISTORY_YEARS_TO_KEEP = 20

# 2 months ≈ 42 business days
HORIZON_BDAYS = 42

# Events
CRASH_1D = -0.10          # 1-day forward return <= -10%
CRASH_30D = -0.30         # 30-business-day forward return <= -30%
CRASH_30D_WINDOW = 30     # business days
DD_2M = -0.15             # within next 42 bdays, drawdown from trailing 252b peak <= -15%

# Stale policy A (fixed)
STALE_DAYS = {
    "daily": 7,
    "weekly": 21,
    "monthly": 120,     # reference only
    "quarterly": 220,   # reference only
}

# FRED polite rate limit
FRED_SLEEP_SEC = 0.60

# Fixed split (preferred; used only if feasible with actual data coverage)
SPLIT_TRAIN_END = "2013-12-31"
SPLIT_VAL_END = "2018-12-31"
# Test is 2019-01-01 ~ last_label_date (if feasible)

# Logistic regression training (numpy only)
LR_L2 = 1.0
LR_LR = 0.05
LR_EPOCHS = 900
LR_BATCH = 4096

# Score scaling (probability -> 0..100)
SCORE_CENTER = 0.0
SCORE_SCALE = 100.0

# Issue automation
ISSUE_COOLDOWN_DAYS = 14

# Minimum rows after NaN filtering (tunable)
# NOTE: too strict values (e.g., 5000) will fail when a key series starts late (e.g., WEI).
MIN_ROWS = int(os.environ.get("MIN_ROWS", "1500"))


# =========================
# Series from FRED
# =========================
SERIES = {
    # Economy pulse (weekly)
    "WEI":     ("WEI", "weekly", "diff_4w_neg", True),

    # Labor (weekly) - YoY smoothing for short-horizon robustness
    "ICSA":    ("ICSA", "weekly", "log_yoy_52w_4wma", True),
    "CCSA":    ("CCSA", "weekly", "log_yoy_52w_4wma", True),

    # Rates / Curve (daily)
    "T10Y3M":  ("T10Y3M", "daily", "level_neg", True),

    # Credit (daily)
    "HY_OAS":  ("BAMLH0A0HYM2", "daily", "level_plus_diff_1m", True),
    "IG_OAS":  ("BAMLC0A0CM", "daily", "level_plus_diff_1m", True),

    # Financial conditions (weekly)
    "NFCI":    ("NFCI", "weekly", "level", True),
    "STLFSI4": ("STLFSI4", "weekly", "level", True),

    # Market stress (daily)
    "VIX":     ("VIXCLS", "daily", "level", True),
    "SP500":   ("SP500", "daily", "ret_1m2m_neg", True),

    # Reference only (monthly) - NOT used for score
    "FEDFUNDS_REF": ("FEDFUNDS", "monthly", "diff_1m", False),
    "UNRATE_REF":   ("UNRATE", "monthly", "diff_1m", False),
}

FEATURE_KEYS_BASE = [k for k, v in SERIES.items() if v[3] is True]  # base features from series
# Extra market features derived from SP500 (still free, higher short-horizon signal)
EXTRA_FEATURE_KEYS = ["SPX_RVOL_1M", "SPX_MDD_1M"]
FEATURE_KEYS = FEATURE_KEYS_BASE + EXTRA_FEATURE_KEYS


# =========================
# Helpers
# =========================

def jst_today_date() -> dt.date:
    now_utc = dt.datetime.utcnow()
    now_jst = now_utc + dt.timedelta(hours=9)
    return now_jst.date()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: dict) -> None:
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_secrets() -> dict:
    raw = os.environ.get("APP_SECRETS", "").strip()
    if not raw:
        raise RuntimeError("Missing APP_SECRETS env (set GitHub Secret APP_SECRETS as JSON).")
    data = json.loads(raw)
    if "FRED_API_KEY" not in data or not data["FRED_API_KEY"]:
        raise RuntimeError("APP_SECRETS JSON must contain non-empty key: FRED_API_KEY")
    return data

def fred_get_series_obs(api_key: str, series_id: str, start_date: str) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "observation_start": start_date,
    }
    for attempt in range(3):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5 * (2 ** attempt))
            continue
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        dates, vals = [], []
        for o in obs:
            d = o.get("date")
            v = o.get("value")
            if v in (".", None, ""):
                val = np.nan
            else:
                try:
                    val = float(v)
                except ValueError:
                    val = np.nan
            dates.append(pd.to_datetime(d))
            vals.append(val)
        return pd.Series(vals, index=pd.DatetimeIndex(dates, name="date"), name=series_id).sort_index()
    raise RuntimeError(f"FRED request failed after retries: {series_id}")

def native_stale(freq_group: str, last_obs_date: pd.Timestamp, run_date: dt.date) -> Tuple[int, bool]:
    if pd.isna(last_obs_date):
        return (10**9, True)
    delta_days = (pd.Timestamp(run_date) - last_obs_date.normalize()).days
    threshold = STALE_DAYS.get(freq_group, 21)
    return (delta_days, delta_days > threshold)

def calendar_index(start: str, end_date: dt.date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end_date), freq="D")

def business_index(start: str, end_date: dt.date) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end_date), freq="B")

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def history_load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(f, parse_dates=["date"])

def history_save(path: str, df: pd.DataFrame) -> None:
    ensure_dir(os.path.dirname(path))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)

def prune_history(df: pd.DataFrame, run_date: dt.date, years: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp(run_date) - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff].copy()


# =========================
# Feature engineering (risk-up => larger)
# =========================

def compute_feature(kind: str, s: pd.Series) -> pd.Series:
    b_4w = 20
    b_1m = 21
    b_2m = 42
    b_52w = 260  # ~52 weeks in business days (weekly data is ffilled to B)

    if kind == "level":
        return s
    if kind == "level_neg":
        return -s

    if kind == "diff_1m":
        return s - s.shift(b_1m)
    if kind == "diff_4w_neg":
        return -(s - s.shift(b_4w))

    if kind == "log_yoy_52w_4wma":
        x = s.rolling(b_4w, min_periods=max(5, b_4w // 4)).mean()
        x = x.replace(0, np.nan)
        lx = np.log(x)
        lx = lx.replace([np.inf, -np.inf], np.nan)
        return lx - lx.shift(b_52w)

    if kind == "level_plus_diff_1m":
        return s + (s - s.shift(b_1m))

    if kind == "ret_1m2m_neg":
        r1m = s.pct_change(b_1m)
        r2m = s.pct_change(b_2m)
        return -(0.6 * r1m + 0.4 * r2m)

    raise ValueError(f"Unknown feature kind: {kind}")

def zscore_past_only(x: pd.Series, win: int) -> pd.Series:
    mu = x.rolling(win, min_periods=max(120, win // 10)).mean()
    sd = x.rolling(win, min_periods=max(120, win // 10)).std()
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).clip(-6, 6)

def pick_rolling_window(freq_group: str) -> int:
    _ = freq_group
    return 1260  # ~5y on business days

def compute_market_derived_features(spx: pd.Series) -> pd.DataFrame:
    r = spx.pct_change()
    rvol_1m = r.rolling(21, min_periods=15).std()

    roll_max = spx.rolling(21, min_periods=15).max()
    dd = (spx / roll_max) - 1.0
    mdd_1m = (-dd.rolling(21, min_periods=15).min())

    return pd.DataFrame({
        "SPX_RVOL_1M": rvol_1m,
        "SPX_MDD_1M": mdd_1m,
    }, index=spx.index)


# =========================
# Events (forward)
# =========================

def compute_forward_events(sp500: pd.Series) -> pd.DataFrame:
    fwd1 = sp500.shift(-1) / sp500 - 1.0
    fwd30 = sp500.shift(-CRASH_30D_WINDOW) / sp500 - 1.0

    crash_1d_at_u = (fwd1 <= CRASH_1D).astype(float)
    crash_30d_at_u = (fwd30 <= CRASH_30D).astype(float)

    crash_1d_fwd = crash_1d_at_u[::-1].rolling(HORIZON_BDAYS, min_periods=1).max()[::-1].fillna(0).astype(int)
    crash_30d_fwd = crash_30d_at_u[::-1].rolling(HORIZON_BDAYS, min_periods=1).max()[::-1].fillna(0).astype(int)

    trailing_peak = sp500.rolling(252, min_periods=60).max()
    dd_now = (sp500 / trailing_peak) - 1.0
    dd_forward_min = dd_now[::-1].rolling(HORIZON_BDAYS, min_periods=10).min()[::-1]
    dd_2m = (dd_forward_min <= DD_2M).fillna(0).astype(int)

    return pd.DataFrame({
        "crash_1d_fwd": crash_1d_fwd,
        "crash_30d_fwd": crash_30d_fwd,
        "dd_2m": dd_2m,
    }, index=sp500.index)

def combine_prob_any(probs: Dict[str, pd.Series]) -> pd.Series:
    p = None
    for _, s in probs.items():
        if p is None:
            p = (1.0 - s)
        else:
            p = p * (1.0 - s)
    if p is None:
        return pd.Series(dtype=float)
    return 1.0 - p


# =========================
# Supervised model: ridge logistic regression (numpy only)
# =========================

def train_logreg_ridge(X: np.ndarray, y: np.ndarray,
                       l2: float = 1.0, lr: float = 0.05,
                       epochs: int = 900, batch: int = 4096) -> Tuple[np.ndarray, float, dict]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    pos = float(y.mean())
    if pos <= 0:
        pos_w = 1.0
    else:
        pos_w = min(12.0, (1.0 - pos) / max(pos, 1e-6))
    weights = np.where(y == 1, pos_w, 1.0).astype(float)

    rng = np.random.default_rng(42)
    idx = np.arange(n)

    def loss_and_grad(Xb, yb, wb):
        z = Xb @ w + b
        p = sigmoid(z)
        eps = 1e-9
        loss = -np.mean(wb * (yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps))) + 0.5 * l2 * np.sum(w * w)
        diff = (p - yb) * wb
        gw = (Xb.T @ diff) / len(yb) + l2 * w
        gb = float(np.mean(diff))
        return float(loss), gw, gb

    best = {"loss": float("inf"), "w": None, "b": None}

    for ep in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, batch):
            j = idx[s:s + batch]
            Xb = X[j]
            yb = y[j]
            wb = weights[j]
            loss, gw, gb = loss_and_grad(Xb, yb, wb)
            w -= lr * gw
            b -= lr * gb

        if (ep + 1) % 50 == 0 or ep == 0:
            z = X @ w + b
            p = sigmoid(z)
            eps = 1e-9
            tr_loss = -np.mean(weights * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) + 0.5 * l2 * np.sum(w * w)
            if tr_loss < best["loss"]:
                best["loss"] = float(tr_loss)
                best["w"] = w.copy()
                best["b"] = float(b)

    if best["w"] is None:
        best["w"] = w
        best["b"] = b

    stats = {
        "train_pos_rate": float(y.mean()),
        "pos_weight": float(pos_w),
        "train_loss_best": float(best["loss"]),
        "epochs": int(epochs),
        "l2": float(l2),
        "lr": float(lr),
        "batch": int(batch),
    }
    return best["w"], best["b"], stats

def predict_prob(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


# =========================
# Backtest metrics
# =========================

def confusion_counts(pred: pd.Series, y: pd.Series) -> dict:
    df = pd.DataFrame({"pred": pred, "y": y}).dropna()
    if df.empty:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": None, "recall": None}
    tp = int(((df["pred"] == 1) & (df["y"] == 1)).sum())
    fp = int(((df["pred"] == 1) & (df["y"] == 0)).sum())
    fn = int(((df["pred"] == 0) & (df["y"] == 1)).sum())
    tn = int(((df["pred"] == 0) & (df["y"] == 0)).sum())
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}

def compute_metrics_for_threshold(p_any: pd.Series, events: pd.DataFrame, thr: float) -> dict:
    df = pd.DataFrame({"p_any": p_any}).join(events, how="inner").dropna()
    if df.empty:
        return {"note": "no data"}

    crisis = (df["p_any"] >= thr).astype(int)

    out = {
        "threshold": float(thr),
        "days": int(len(df)),
        "crisis_days": int(crisis.sum()),
        "crisis_share": float(crisis.mean()),
    }

    for k in ["dd_2m", "crash_1d_fwd", "crash_30d_fwd"]:
        out[f"base_{k}"] = float(df[k].mean())

    if crisis.sum() == 0:
        for k in ["dd_2m", "crash_1d_fwd", "crash_30d_fwd"]:
            out[f"hit_{k}"] = None
            out[f"lift_{k}"] = None
            out[f"cm_{k}"] = confusion_counts(crisis, df[k].astype(int))
        return out

    for k in ["dd_2m", "crash_1d_fwd", "crash_30d_fwd"]:
        hit = float(df.loc[crisis == 1, k].mean())
        out[f"hit_{k}"] = hit
        base = out[f"base_{k}"]
        out[f"lift_{k}"] = (hit / base) if base > 0 else None
        out[f"cm_{k}"] = confusion_counts(crisis, df[k].astype(int))

    return out

def scan_thresholds(p_any: pd.Series, events: pd.DataFrame, percentiles: List[float]) -> List[dict]:
    df = pd.DataFrame({"p_any": p_any}).join(events, how="inner").dropna()
    if df.empty:
        return []
    ps = df["p_any"].values
    out = []
    for q in percentiles:
        thr = float(np.quantile(ps, q))
        m = compute_metrics_for_threshold(df["p_any"], df[events.columns], thr)
        m["percentile"] = float(q)
        out.append(m)
    return out

def pick_crisis_threshold(p_any_val: pd.Series, events_val: pd.DataFrame) -> Tuple[float, dict]:
    df = pd.DataFrame({"p_any": p_any_val}).join(events_val, how="inner").dropna()
    if df.empty:
        return 0.99, {"note": "no data"}

    ps = df["p_any"].values
    qs = np.linspace(0.80, 0.99, 20)
    best = {"obj": float("-inf"), "thr": None, "metrics": None}

    for q in qs:
        thr = float(np.quantile(ps, q))
        m = compute_metrics_for_threshold(df["p_any"], df[events_val.columns], thr)
        if m.get("crisis_days", 0) < 3:
            continue
        if m.get("hit_dd_2m") is None:
            continue

        hit_dd = m["hit_dd_2m"] if m["hit_dd_2m"] is not None else 0.0
        hit_c30 = m["hit_crash_30d_fwd"] if m["hit_crash_30d_fwd"] is not None else 0.0
        hit_c1 = m["hit_crash_1d_fwd"] if m["hit_crash_1d_fwd"] is not None else 0.0

        obj = (2.0 * hit_dd + 1.2 * hit_c30 + 0.6 * hit_c1 - 0.85 * m["crisis_share"])
        if obj > best["obj"]:
            best = {"obj": float(obj), "thr": thr, "metrics": m}

    if best["thr"] is None:
        thr = float(np.quantile(ps, 0.92))
        m = compute_metrics_for_threshold(df["p_any"], df[events_val.columns], thr)
        m["objective"] = None
        return thr, {"note": "fallback", "metrics": m}

    best["metrics"]["objective"] = best["obj"]
    return float(best["thr"]), best["metrics"]

def regimes_from_thresholds(p_any: pd.Series, thr_notice: float, thr_alert: float, thr_crisis: float) -> pd.Series:
    def r(v: float) -> str:
        if v >= thr_crisis:
            return "危機"
        if v >= thr_alert:
            return "警戒"
        if v >= thr_notice:
            return "注意"
        return "平常"
    return p_any.apply(lambda x: r(float(x)) if pd.notna(x) else None)

def summarize_by_regime(p_any: pd.Series, regime: pd.Series, events: pd.DataFrame) -> dict:
    df = pd.DataFrame({"p_any": p_any, "regime": regime}).join(events, how="inner").dropna()
    if df.empty:
        return {"note": "no data"}
    out = {}
    for rg in ["平常", "注意", "警戒", "危機"]:
        sub = df[df["regime"] == rg]
        if sub.empty:
            out[rg] = {"days": 0}
            continue
        out[rg] = {
            "days": int(len(sub)),
            "rate_dd_2m": float(sub["dd_2m"].mean()),
            "rate_crash_1d_fwd": float(sub["crash_1d_fwd"].mean()),
            "rate_crash_30d_fwd": float(sub["crash_30d_fwd"].mean()),
            "avg_p_any": float(sub["p_any"].mean()),
        }
    return out


# =========================
# Issues automation
# =========================

def github_api_headers() -> Dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing GITHUB_TOKEN env.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def repo_owner_name() -> Tuple[str, str]:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo or "/" not in repo:
        raise RuntimeError("GITHUB_REPOSITORY not set (expected owner/repo).")
    owner, name = repo.split("/", 1)
    return owner, name

def gh_create_issue(title: str, body: str) -> int:
    owner, name = repo_owner_name()
    url = f"https://api.github.com/repos/{owner}/{name}/issues"
    r = requests.post(url, headers=github_api_headers(), json={"title": title, "body": body}, timeout=30)
    r.raise_for_status()
    return int(r.json()["number"])

def gh_comment_issue(issue_number: int, body: str) -> None:
    owner, name = repo_owner_name()
    url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}/comments"
    r = requests.post(url, headers=github_api_headers(), json={"body": body}, timeout=30)
    r.raise_for_status()

def gh_close_issue(issue_number: int) -> None:
    owner, name = repo_owner_name()
    url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}"
    r = requests.patch(url, headers=github_api_headers(), json={"state": "closed"}, timeout=30)
    r.raise_for_status()


# =========================
# Main
# =========================

def main():
    secrets = load_secrets()
    api_key = secrets["FRED_API_KEY"]

    run_date = jst_today_date()
    print(f"[run] date={run_date.isoformat()} {RUN_TZ_LABEL}")

    # Adaptive split size controls
    MIN_TRAIN_SPLIT = int(os.environ.get("MIN_TRAIN_SPLIT", "1200"))
    MIN_VAL_SPLIT   = int(os.environ.get("MIN_VAL_SPLIT", "400"))
    MIN_TEST_SPLIT  = int(os.environ.get("MIN_TEST_SPLIT", "400"))

    print(f"[config] MIN_ROWS={MIN_ROWS}")
    print(f"[config] MIN_TRAIN_SPLIT={MIN_TRAIN_SPLIT} MIN_VAL_SPLIT={MIN_VAL_SPLIT} MIN_TEST_SPLIT={MIN_TEST_SPLIT}")

    idx_all = calendar_index(START_DATE, run_date)  # D
    idx_biz = business_index(START_DATE, run_date)  # B

    raw = {}
    meta = {}

    for key, (fred_id, freq_group, _, _) in SERIES.items():
        try:
            s = fred_get_series_obs(api_key, fred_id, START_DATE)
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            print(f"[fred] skip {key} ({fred_id}): HTTP {status}")
            meta[key] = {
                "fred_id": fred_id,
                "freq_group": freq_group,
                "last_obs_date": None,
                "stale_days": None,
                "stale": True,
                "error": f"HTTPError {status}",
            }
            raw[key] = pd.Series(np.nan, index=idx_all, name=key)
            time.sleep(FRED_SLEEP_SEC)
            continue
        except Exception as e:
            print(f"[fred] skip {key} ({fred_id}): {e}")
            meta[key] = {
                "fred_id": fred_id,
                "freq_group": freq_group,
                "last_obs_date": None,
                "stale_days": None,
                "stale": True,
                "error": str(e),
            }
            raw[key] = pd.Series(np.nan, index=idx_all, name=key)
            time.sleep(FRED_SLEEP_SEC)
            continue

        last_obs = s.dropna().index.max() if s.dropna().shape[0] else pd.NaT
        stale_days, stale_flag = native_stale(freq_group, last_obs, run_date)

        s_aligned = s.reindex(idx_all).ffill()
        raw[key] = s_aligned
        meta[key] = {
            "fred_id": fred_id,
            "freq_group": freq_group,
            "last_obs_date": None if pd.isna(last_obs) else str(last_obs.date()),
            "stale_days": int(stale_days) if stale_days < 10**8 else None,
            "stale": bool(stale_flag),
        }

        time.sleep(FRED_SLEEP_SEC)

    df_raw_all = pd.DataFrame(raw, index=idx_all)
    df_raw = df_raw_all.reindex(idx_biz).ffill()

    feat_z = {}
    feat_raw = {}

    for key, (_, freq_group, kind, use_for_score) in SERIES.items():
        if not use_for_score:
            continue
        x = compute_feature(kind, df_raw[key])

        if meta.get(key, {}).get("stale", True):
            x = x * np.nan

        win = pick_rolling_window(freq_group)
        z = zscore_past_only(x, win)

        feat_raw[key] = x
        feat_z[key] = z

    spx_extra = compute_market_derived_features(df_raw["SP500"])
    for k in EXTRA_FEATURE_KEYS:
        x = spx_extra[k]
        z = zscore_past_only(x, 1260)
        feat_raw[k] = x
        feat_z[k] = z

    df_feat = pd.DataFrame(feat_z, index=idx_biz)

    events = compute_forward_events(df_raw["SP500"])
    last_label_idx = events.dropna().index
    if last_label_idx.empty:
        raise RuntimeError("No events computed (SP500 too short).")
    last_label_date = last_label_idx.max()

    ds = df_feat.join(events, how="inner")

    # valid rows: all features present AND events present
    valid_mask = ds[FEATURE_KEYS].notna().all(axis=1)
    ds = ds.loc[valid_mask].copy()
    ds = ds.dropna(subset=["dd_2m", "crash_1d_fwd", "crash_30d_fwd"])

    # Diagnostics before failing
    if ds.empty or len(ds) < MIN_ROWS:
        diag = {
            "rows_after_filter": int(len(ds)),
            "required_min_rows": int(MIN_ROWS),
            "feature_keys": FEATURE_KEYS,
            "feature_non_nan_counts": {k: int(ds[k].notna().sum()) if k in ds.columns else 0 for k in FEATURE_KEYS},
            "ds_start": None if ds.empty else str(ds.index.min().date()),
            "ds_end": None if ds.empty else str(ds.index.max().date()),
        }
        print("[diag] not enough rows after NaN filtering")
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        raise RuntimeError("Not enough data after NaN filtering; check staleness, series coverage, or reduce MIN_ROWS.")

    # -----------------------------
    # Split (fixed if feasible; else adaptive)
    # -----------------------------
    n_all = len(ds)
    if n_all < (MIN_TRAIN_SPLIT + MIN_VAL_SPLIT + MIN_TEST_SPLIT):
        diag = {
            "n_all": int(n_all),
            "need_at_least": int(MIN_TRAIN_SPLIT + MIN_VAL_SPLIT + MIN_TEST_SPLIT),
            "min_train": MIN_TRAIN_SPLIT,
            "min_val": MIN_VAL_SPLIT,
            "min_test": MIN_TEST_SPLIT,
            "ds_start": str(ds.index.min().date()),
            "ds_end": str(ds.index.max().date()),
        }
        print("[diag] split impossible due to insufficient total rows")
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        raise RuntimeError("Not enough total rows to split; relax features or reduce min split sizes.")

    split_mode = "fixed"
    train_end = pd.Timestamp(SPLIT_TRAIN_END)
    val_end   = pd.Timestamp(SPLIT_VAL_END)
    test_start = pd.Timestamp("2019-01-01")

    train_fixed = ds[(ds.index <= train_end)]
    val_fixed   = ds[(ds.index > train_end) & (ds.index <= val_end)]
    test_fixed  = ds[(ds.index >= test_start)]

    fixed_ok = (len(train_fixed) >= MIN_TRAIN_SPLIT and
                len(val_fixed)   >= MIN_VAL_SPLIT and
                len(test_fixed)  >= MIN_TEST_SPLIT)

    if not fixed_ok:
        split_mode = "adaptive"

        # Adaptive split: keep chronological order and ensure minimum rows.
        test_len = max(MIN_TEST_SPLIT, int(0.20 * n_all))
        val_len  = max(MIN_VAL_SPLIT,  int(0.15 * n_all))

        if (test_len + val_len + MIN_TRAIN_SPLIT) > n_all:
            excess = (test_len + val_len + MIN_TRAIN_SPLIT) - n_all
            shrink_val = min(excess, max(0, val_len - MIN_VAL_SPLIT))
            val_len -= shrink_val
            excess -= shrink_val
            shrink_test = min(excess, max(0, test_len - MIN_TEST_SPLIT))
            test_len -= shrink_test
            excess -= shrink_test
            if excess > 0:
                diag = {
                    "n_all": int(n_all),
                    "min_train": MIN_TRAIN_SPLIT,
                    "min_val": MIN_VAL_SPLIT,
                    "min_test": MIN_TEST_SPLIT,
                    "proposed_val": int(val_len),
                    "proposed_test": int(test_len),
                    "ds_start": str(ds.index.min().date()),
                    "ds_end": str(ds.index.max().date()),
                }
                print("[diag] adaptive split still impossible")
                print(json.dumps(diag, ensure_ascii=False, indent=2))
                raise RuntimeError("Split impossible even after shrinking; reduce mins or widen data coverage.")

        test_start_idx = n_all - test_len
        val_start_idx  = test_start_idx - val_len

        train_end = ds.index[val_start_idx - 1]
        val_end   = ds.index[test_start_idx - 1]
        test_start = ds.index[test_start_idx]

        train = ds[(ds.index <= train_end)]
        val   = ds[(ds.index > train_end) & (ds.index <= val_end)]
        test  = ds[(ds.index >= test_start)]
    else:
        train, val, test = train_fixed, val_fixed, test_fixed

    if len(train) < MIN_TRAIN_SPLIT or len(val) < MIN_VAL_SPLIT or len(test) < MIN_TEST_SPLIT:
        diag = {
            "split_mode": split_mode,
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "min_train": MIN_TRAIN_SPLIT,
            "min_val": MIN_VAL_SPLIT,
            "min_test": MIN_TEST_SPLIT,
            "train_range": [str(train.index.min().date()), str(train.index.max().date())] if len(train) else None,
            "val_range": [str(val.index.min().date()), str(val.index.max().date())] if len(val) else None,
            "test_range": [str(test.index.min().date()), str(test.index.max().date())] if len(test) else None,
            "ds_start": str(ds.index.min().date()),
            "ds_end": str(ds.index.max().date()),
        }
        print("[diag] split produced too little data")
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        raise RuntimeError("Split produced too little data (after adaptive check).")

    print(f"[split] mode={split_mode} train={len(train)} val={len(val)} test={len(test)} "
          f"train_end={train_end.date()} val_end={val_end.date()} test_start={test_start.date()}")

    # Train/val/test arrays
    Xtr = train[FEATURE_KEYS].to_numpy(dtype=float)
    Xval = val[FEATURE_KEYS].to_numpy(dtype=float)
    Xte = test[FEATURE_KEYS].to_numpy(dtype=float)

    model_stats = {}
    coefs = {}
    probs_all = {}

    targets = {
        "p_dd_2m": ("dd_2m", 2.0),
        "p_crash_1d": ("crash_1d_fwd", 1.0),
        "p_crash_30d": ("crash_30d_fwd", 1.3),
    }

    for pname, (ycol, _) in targets.items():
        ytr = train[ycol].to_numpy(dtype=int)
        w, b, st = train_logreg_ridge(Xtr, ytr, l2=LR_L2, lr=LR_LR, epochs=LR_EPOCHS, batch=LR_BATCH)

        coef_list = [{"feature": FEATURE_KEYS[i], "coef": float(w[i])} for i in range(len(FEATURE_KEYS))]
        coef_list.sort(key=lambda x: abs(x["coef"]), reverse=True)
        coefs[pname] = coef_list
        model_stats[pname] = st

        Xall = ds[FEATURE_KEYS].to_numpy(dtype=float)
        p = predict_prob(Xall, w, b)
        probs_all[pname] = pd.Series(p, index=ds.index, name=pname)

    p_any = combine_prob_any({
        "dd": probs_all["p_dd_2m"],
        "c1": probs_all["p_crash_1d"],
        "c30": probs_all["p_crash_30d"],
    }).rename("p_any")

    p_any_val = p_any.loc[val.index]
    ev_val = events.loc[p_any_val.index][["dd_2m", "crash_1d_fwd", "crash_30d_fwd"]]

    thr_crisis, thr_pick_info = pick_crisis_threshold(p_any_val, ev_val)

    pv = p_any_val.dropna()
    thr_notice = float(np.quantile(pv.values, 0.70))
    thr_alert = float(np.quantile(pv.values, 0.85))
    thr_notice = min(thr_notice, thr_alert - 1e-6)
    thr_alert = min(thr_alert, thr_crisis - 1e-6)

    regime = regimes_from_thresholds(p_any, thr_notice, thr_alert, thr_crisis)

    scan_q = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
    scan_val = scan_thresholds(p_any_val, ev_val, scan_q)

    p_any_test = p_any.loc[test.index]
    ev_test = events.loc[p_any_test.index][["dd_2m", "crash_1d_fwd", "crash_30d_fwd"]]
    scan_test = scan_thresholds(p_any_test, ev_test, scan_q)

    metrics_val_at_pick = compute_metrics_for_threshold(p_any_val, ev_val, thr_crisis)
    metrics_test_at_pick = compute_metrics_for_threshold(p_any_test, ev_test, thr_crisis)

    by_regime_test = summarize_by_regime(p_any_test, regime.loc[p_any_test.index], ev_test)

    asof = str(idx_biz[-1].date())
    latest_p_any = float(p_any.iloc[-1]) if pd.notna(p_any.iloc[-1]) else None
    latest_score = None if latest_p_any is None else float(SCORE_CENTER + SCORE_SCALE * latest_p_any)
    latest_regime = str(regime.iloc[-1]) if pd.notna(regime.iloc[-1]) else None

    latest_p_components = {
        "p_dd_2m": float(probs_all["p_dd_2m"].iloc[-1]) if pd.notna(probs_all["p_dd_2m"].iloc[-1]) else None,
        "p_crash_1d": float(probs_all["p_crash_1d"].iloc[-1]) if pd.notna(probs_all["p_crash_1d"].iloc[-1]) else None,
        "p_crash_30d": float(probs_all["p_crash_30d"].iloc[-1]) if pd.notna(probs_all["p_crash_30d"].iloc[-1]) else None,
    }

    latest_detail = {}
    for k in FEATURE_KEYS:
        if k in SERIES:
            fred_id = meta.get(k, {}).get("fred_id")
            stale = bool(meta.get(k, {}).get("stale", True))
            last_obs_date = meta.get(k, {}).get("last_obs_date")
            stale_days = meta.get(k, {}).get("stale_days")
            raw_level = df_raw[k].iloc[-1] if k in df_raw.columns else np.nan
        else:
            fred_id = "DERIVED_FROM_SP500"
            stale = False
            last_obs_date = meta.get("SP500", {}).get("last_obs_date")
            stale_days = meta.get("SP500", {}).get("stale_days")
            raw_level = df_raw["SP500"].iloc[-1]

        fv = feat_raw[k].iloc[-1] if k in feat_raw else np.nan
        z = df_feat[k].iloc[-1] if k in df_feat.columns else np.nan

        latest_detail[k] = {
            "fred_id": fred_id,
            "raw_level": None if (isinstance(raw_level, float) and math.isnan(raw_level)) else float(raw_level),
            "feature_value": None if (isinstance(fv, float) and math.isnan(fv)) else float(fv),
            "z": None if (isinstance(z, float) and math.isnan(z)) else float(z),
            "stale": stale,
            "last_obs_date": last_obs_date,
            "stale_days": stale_days,
        }

    reference = {}
    for key, (_, _, _, use_for_score) in SERIES.items():
        if use_for_score:
            continue
        v = df_raw[key].iloc[-1]
        reference[key] = None if (isinstance(v, float) and math.isnan(v)) else float(v)

    latest_obj = {
        "asof": asof,
        "horizon_bdays": HORIZON_BDAYS,
        "p_any": latest_p_any,
        "p_components": latest_p_components,
        "score": latest_score,
        "regime": latest_regime,
        "thresholds": {
            "notice_p": thr_notice,
            "alert_p": thr_alert,
            "crisis_p": thr_crisis,
        },
        "data_staleness": meta,
        "reference_only": reference,
        "definitions": {
            "regimes": ["平常", "注意", "警戒", "危機"],
            "events": {
                "crash_1d_fwd": f"{HORIZON_BDAYS}営業日以内に(任意のuで)1日先リターン<={int(CRASH_1D*100)}%",
                "crash_30d_fwd": f"{HORIZON_BDAYS}営業日以内に(任意のuで){CRASH_30D_WINDOW}営業日先リターン<={int(CRASH_30D*100)}%",
                "dd_2m": f"{HORIZON_BDAYS}営業日以内に直近252営業日高値から{int(DD_2M*100)}%到達",
            },
            "stale_policy_A": STALE_DAYS,
            "split": {
                "mode": split_mode,
                "train_end": str(pd.Timestamp(train_end).date()),
                "val_end": str(pd.Timestamp(val_end).date()),
                "test_start": str(pd.Timestamp(test_start).date()),
                "preferred_fixed": {"train_end": SPLIT_TRAIN_END, "val_end": SPLIT_VAL_END, "test_start": "2019-01-01"},
                "min_sizes": {"train": MIN_TRAIN_SPLIT, "val": MIN_VAL_SPLIT, "test": MIN_TEST_SPLIT},
            },
            "note": "3モデル（dd / 1日-10 / 30日-30）を個別学習し、p_any=1-Π(1-p)で合成。月次は参考表示のみ。週次はD→B整列。",
        },
        "backtest": {
            "windows": {
                "train_start": str(train.index.min().date()),
                "train_end": str(train.index.max().date()),
                "val_start": str(val.index.min().date()),
                "val_end": str(val.index.max().date()),
                "test_start": str(test.index.min().date()),
                "test_end": str(test.index.max().date()),
                "last_label_date": str(pd.Timestamp(last_label_date).date()),
            },
            "model_train_stats": model_stats,
            "picked_threshold_info": thr_pick_info,
            "metrics_val_at_picked_crisis": metrics_val_at_pick,
            "metrics_test_at_picked_crisis": metrics_test_at_pick,
            "by_regime_test": by_regime_test,
            "threshold_scan_val": scan_val,
            "threshold_scan_test": scan_test,
        },
        "debug": {
            "latest_features": latest_detail,
            "coef_abs_sorted": {
                "p_dd_2m": coefs["p_dd_2m"][:20],
                "p_crash_1d": coefs["p_crash_1d"][:20],
                "p_crash_30d": coefs["p_crash_30d"][:20],
            },
            "feature_keys": FEATURE_KEYS,
        },
    }

    ensure_dir("data")
    write_json("data/latest.json", latest_obj)

    hist_path = "data/history.csv.gz"
    hist = history_load(hist_path)
    new_row = {
        "date": pd.Timestamp(run_date),
        "p_any": latest_p_any,
        "p_dd_2m": latest_p_components["p_dd_2m"],
        "p_crash_1d": latest_p_components["p_crash_1d"],
        "p_crash_30d": latest_p_components["p_crash_30d"],
        "score": latest_score,
        "regime": latest_regime,
        "notice_p": thr_notice,
        "alert_p": thr_alert,
        "crisis_p": thr_crisis,
        "split_mode": split_mode,
    }
    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    hist = prune_history(hist, run_date, HISTORY_YEARS_TO_KEEP)
    history_save(hist_path, hist)

    state_path = "data/state.json"
    state = read_json(state_path) or {
        "open_issue_number": None,
        "cooldown_until": None,
        "last_regime": None,
        "last_issue_date": None,
    }

    prev_regime = state.get("last_regime")
    open_issue = state.get("open_issue_number")
    cooldown_until = state.get("cooldown_until")

    def in_cooldown(today: dt.date) -> bool:
        if not cooldown_until:
            return False
        try:
            cd = dt.date.fromisoformat(cooldown_until)
        except Exception:
            return False
        return today < cd

    def issue_body(prefix: str) -> str:
        lines = []
        lines.append(prefix)
        lines.append("")
        lines.append(f"- date: {asof}")
        lines.append(f"- p_any: {latest_p_any}")
        lines.append(f"- score: {latest_score}")
        lines.append(f"- regime: {latest_regime}")
        lines.append(f"- split_mode: {split_mode}")
        lines.append(f"- thresholds: notice={thr_notice:.4f} alert={thr_alert:.4f} crisis={thr_crisis:.4f}")
        lines.append("")
        lines.append("## p components")
        for k, v in latest_p_components.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Backtest (picked crisis threshold, test)")
        mt = latest_obj["backtest"]["metrics_test_at_picked_crisis"]
        for k in sorted(mt.keys()):
            lines.append(f"- {k}: {mt[k]}")
        lines.append("")
        lines.append("## Top coefs (abs)")
        for name in ["p_dd_2m", "p_crash_1d", "p_crash_30d"]:
            lines.append(f"- {name}:")
            for c in latest_obj["debug"]["coef_abs_sorted"][name][:6]:
                lines.append(f"  - {c['feature']}: {c['coef']:.4f}")
        return "\n".join(lines)

    entered_crisis = (prev_regime != "危機") and (latest_regime == "危機")
    exited_crisis = (prev_regime == "危機") and (latest_regime != "危機")

    try:
        if entered_crisis:
            if in_cooldown(run_date):
                print("[issue] Entered crisis but in cooldown; skipping.")
            else:
                title = f"暴落リスク: 危機（2か月監視）（{asof}）"
                num = gh_create_issue(title, issue_body("危機に遷移しました（2か月先の暴落監視）。"))
                state["open_issue_number"] = num
                state["last_issue_date"] = asof
                print(f"[issue] Created issue #{num}")

        elif latest_regime == "危機" and open_issue:
            gh_comment_issue(int(open_issue), issue_body("危機継続の更新です。"))
            state["last_issue_date"] = asof
            print(f"[issue] Commented on issue #{open_issue}")

        elif exited_crisis and open_issue:
            gh_comment_issue(int(open_issue), issue_body("危機を解除しました（警戒以下に戻りました）。"))
            gh_close_issue(int(open_issue))
            print(f"[issue] Closed issue #{open_issue}")
            state["open_issue_number"] = None
            cd = run_date + dt.timedelta(days=ISSUE_COOLDOWN_DAYS)
            state["cooldown_until"] = cd.isoformat()

    except Exception as e:
        print(f"[issue] warning: issue automation failed: {e}")

    state["last_regime"] = latest_regime
    write_json(state_path, state)

    print("[done] updated data/latest.json (+verbose), data/history.csv.gz, data/state.json")
    print(f"[result] p_any={latest_p_any} score={latest_score} regime={latest_regime}")


if __name__ == "__main__":
    main()
