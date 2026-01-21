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
# 2M Spec Config
# =========================

START_DATE = "2000-01-01"
RUN_TZ_LABEL = "JST"
HISTORY_YEARS_TO_KEEP = 20

# 2 months ≈ 42 business days
HORIZON_BDAYS = 42

# Event definitions (2M monitoring)
CRASH_1D = -0.10          # forward 1-day <= -10%
CRASH_3D = -0.15          # forward 3-business-day <= -15%
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

# Model training / validation windows (rolling, based on last available label date)
TEST_YEARS = 5
VAL_YEARS = 5

# Logistic regression training
LR_L2 = 1.0        # L2 regularization strength
LR_LR = 0.05       # learning rate
LR_EPOCHS = 800
LR_BATCH = 4096

# Output score scaling (probability -> 0..100)
SCORE_CENTER = 0.0
SCORE_SCALE = 100.0

# Issue automation
ISSUE_COOLDOWN_DAYS = 14


# =========================
# Series (scoring uses only daily/weekly)
# =========================
# key: (fred_id, freq_group, feature_kind, use_for_score)
SERIES = {
    # Economy pulse (weekly)
    "WEI":     ("WEI", "weekly", "diff_4w_neg", True),

    # Labor (weekly)
    "ICSA":    ("ICSA", "weekly", "log_diff_4w", True),
    "CCSA":    ("CCSA", "weekly", "log_diff_4w", True),

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

FEATURE_KEYS = [k for k, v in SERIES.items() if v[3] is True]  # use_for_score


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
    ensure_dir(os.path.dirname(path))
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
# Feature engineering (2M)
# larger => riskier
# =========================

def compute_feature(kind: str, s: pd.Series) -> pd.Series:
    b_4w = 20
    b_1m = 21
    b_2m = 42

    if kind == "level":
        return s
    if kind == "level_neg":
        return -s

    if kind == "diff_1m":
        return s - s.shift(b_1m)
    if kind == "diff_4w_neg":
        return -(s - s.shift(b_4w))

    if kind == "log_diff_4w":
        ls = np.log(s.replace(0, np.nan))
        ls = ls.replace([np.inf, -np.inf], np.nan)
        return ls - ls.shift(b_4w)

    if kind == "level_plus_diff_1m":
        return s + (s - s.shift(b_1m))

    if kind == "ret_1m2m_neg":
        # Two horizons: 1M and 2M, both risk-up when negative
        r1m = s.pct_change(b_1m)
        r2m = s.pct_change(b_2m)
        return -(0.6 * r1m + 0.4 * r2m)

    raise ValueError(f"Unknown feature kind: {kind}")

def zscore_past_only(x: pd.Series, win: int) -> pd.Series:
    # past-only rolling normalization
    mu = x.rolling(win, min_periods=max(60, win // 10)).mean()
    sd = x.rolling(win, min_periods=max(60, win // 10)).std()
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).clip(-6, 6)

def pick_rolling_window(freq_group: str) -> int:
    # keep it stable for supervised model (avoid too short)
    return 1260  # ~5y business days


# =========================
# Events (FIX: forward returns)
# =========================

def compute_forward_events(sp500: pd.Series) -> pd.DataFrame:
    """
    For each business date t:
      crash_1d_fwd: within next HORIZON_BDAYS, exists a day u such that (P[u+1]/P[u]-1) <= -10%
      crash_3d_fwd: within next HORIZON_BDAYS, exists a day u such that (P[u+3]/P[u]-1) <= -15%
      dd_2m: within next HORIZON_BDAYS, drawdown at some future date (from trailing 252b peak at that future date) <= -15%
    """
    # forward 1-day and 3-day returns series at each u
    fwd1 = sp500.shift(-1) / sp500 - 1.0
    fwd3 = sp500.shift(-3) / sp500 - 1.0

    crash_1d_at_u = (fwd1 <= CRASH_1D).astype(float)
    crash_3d_at_u = (fwd3 <= CRASH_3D).astype(float)

    crash_1d_fwd = crash_1d_at_u[::-1].rolling(HORIZON_BDAYS, min_periods=1).max()[::-1].fillna(0).astype(int)
    crash_3d_fwd = crash_3d_at_u[::-1].rolling(HORIZON_BDAYS, min_periods=1).max()[::-1].fillna(0).astype(int)

    # drawdown definition evaluated on each future date (label is allowed to use future info)
    trailing_peak = sp500.rolling(252, min_periods=60).max()
    dd_now = (sp500 / trailing_peak) - 1.0
    dd_forward_min = dd_now[::-1].rolling(HORIZON_BDAYS, min_periods=10).min()[::-1]
    dd_2m = (dd_forward_min <= DD_2M).fillna(0).astype(int)

    return pd.DataFrame({
        "crash_1d_fwd": crash_1d_fwd,
        "crash_3d_fwd": crash_3d_fwd,
        "dd_2m": dd_2m,
    }, index=sp500.index)

def make_labels(events: pd.DataFrame) -> pd.DataFrame:
    y_any = (events[["crash_1d_fwd", "crash_3d_fwd", "dd_2m"]].max(axis=1)).astype(int)
    out = events.copy()
    out["event_any"] = y_any
    return out


# =========================
# Supervised model: ridge logistic regression (numpy only)
# =========================

def train_logreg_ridge(X: np.ndarray, y: np.ndarray,
                       l2: float = 1.0, lr: float = 0.05,
                       epochs: int = 800, batch: int = 4096) -> Tuple[np.ndarray, float, dict]:
    """
    Returns (w, b, train_stats).
    Minimizes: -loglik + 0.5*l2*||w||^2
    """
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    # class weight to reduce imbalance pain (positive heavier)
    pos = float(y.mean())
    if pos <= 0:
        pos_w = 1.0
    else:
        pos_w = min(10.0, (1.0 - pos) / max(pos, 1e-6))
    weights = np.where(y == 1, pos_w, 1.0).astype(float)

    rng = np.random.default_rng(42)
    idx = np.arange(n)

    def loss_and_grad(Xb, yb, wb):
        z = Xb @ w + b
        p = sigmoid(z)
        # weighted logloss
        eps = 1e-9
        loss = -np.mean(wb * (yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps))) + 0.5 * l2 * np.sum(w * w)
        # gradients
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

        # occasional snapshot
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
# Backtest metrics (include base rates & tradeoff)
# =========================

def compute_metrics_for_threshold(p: pd.Series, labels: pd.DataFrame, thr: float) -> dict:
    """
    Define "危機" as p >= thr, then compute:
      - crisis_days
      - hit rates for each event (conditional on crisis)
      - false alarm rates (1-hit)
      - base rates (unconditional)
      - lift
    """
    df = pd.DataFrame({"p": p}).join(labels, how="inner").dropna()
    if df.empty:
        return {"note": "no data"}

    crisis = df["p"] >= thr

    out = {}
    out["threshold"] = float(thr)
    out["days"] = int(len(df))
    out["crisis_days"] = int(crisis.sum())
    out["crisis_share"] = float(crisis.mean())

    # base rates
    for k in ["event_any", "dd_2m", "crash_1d_fwd", "crash_3d_fwd"]:
        base = float(df[k].mean())
        out[f"base_{k}"] = base

    # conditional hit rates
    if crisis.sum() == 0:
        for k in ["event_any", "dd_2m", "crash_1d_fwd", "crash_3d_fwd"]:
            out[f"hit_{k}"] = None
            out[f"lift_{k}"] = None
        return out

    for k in ["event_any", "dd_2m", "crash_1d_fwd", "crash_3d_fwd"]:
        hit = float(df.loc[crisis, k].mean())
        out[f"hit_{k}"] = hit
        base = out[f"base_{k}"]
        out[f"lift_{k}"] = (hit / base) if base > 0 else None

    return out

def scan_thresholds(p: pd.Series, labels: pd.DataFrame, percentiles: List[float]) -> List[dict]:
    df = pd.DataFrame({"p": p}).join(labels, how="inner").dropna()
    if df.empty:
        return []
    ps = df["p"].values
    out = []
    for q in percentiles:
        thr = float(np.quantile(ps, q))
        m = compute_metrics_for_threshold(df["p"], df[labels.columns], thr)
        m["percentile"] = float(q)
        out.append(m)
    return out

def pick_crisis_threshold(p: pd.Series, labels: pd.DataFrame) -> Tuple[float, dict]:
    """
    Pick a crisis threshold by optimizing an objective on validation set:
      objective = 2*hit_event_any + hit_dd_2m + 0.5*hit_crash_1d_fwd - 0.75*crisis_share
    This pushes recall up but penalizes declaring crisis too often.
    """
    df = pd.DataFrame({"p": p}).join(labels, how="inner").dropna()
    if df.empty:
        return 0.99, {"note": "no data"}

    ps = df["p"].values
    qs = np.linspace(0.85, 0.99, 15)  # scan top 15%..1%
    best = {"obj": float("-inf"), "thr": None, "metrics": None}

    for q in qs:
        thr = float(np.quantile(ps, q))
        m = compute_metrics_for_threshold(df["p"], df[labels.columns], thr)
        if m.get("hit_event_any") is None:
            continue
        obj = (
            2.0 * m["hit_event_any"]
            + 1.0 * m["hit_dd_2m"]
            + 0.5 * m["hit_crash_1d_fwd"]
            - 0.75 * m["crisis_share"]
        )
        if obj > best["obj"]:
            best = {"obj": float(obj), "thr": thr, "metrics": m}

    if best["thr"] is None:
        thr = float(np.quantile(ps, 0.92))
        return thr, {"note": "fallback", "metrics": compute_metrics_for_threshold(df["p"], df[labels.columns], thr)}
    best["metrics"]["objective"] = best["obj"]
    return float(best["thr"]), best["metrics"]

def regimes_from_thresholds(p: pd.Series, thr_notice: float, thr_alert: float, thr_crisis: float) -> pd.Series:
    def r(v: float) -> str:
        if v >= thr_crisis:
            return "危機"
        if v >= thr_alert:
            return "警戒"
        if v >= thr_notice:
            return "注意"
        return "平常"
    return p.apply(lambda x: r(float(x)) if pd.notna(x) else None)

def summarize_by_regime(p: pd.Series, regime: pd.Series, labels: pd.DataFrame) -> dict:
    df = pd.DataFrame({"p": p, "regime": regime}).join(labels, how="inner").dropna()
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
            "rate_event_any": float(sub["event_any"].mean()),
            "rate_dd_2m": float(sub["dd_2m"].mean()),
            "rate_crash_1d_fwd": float(sub["crash_1d_fwd"].mean()),
            "rate_crash_3d_fwd": float(sub["crash_3d_fwd"].mean()),
            "avg_p": float(sub["p"].mean()),
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

    # Align series on calendar days first (D) so weekly Saturday obs doesn't drop,
    # then convert to business days (B) for scoring/events.
    idx_all = calendar_index(START_DATE, run_date)  # D
    idx_biz = business_index(START_DATE, run_date)  # B

    raw = {}
    meta = {}

    # 1) Fetch all series
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

        s_aligned = s.reindex(idx_all).ffill()  # calendar align

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
    df_raw = df_raw_all.reindex(idx_biz).ffill()  # business-day view

    # 2) Build features (past-only normalized z for supervised model)
    feat = {}
    feat_raw = {}  # for debug
    for key, (_, freq_group, kind, use_for_score) in SERIES.items():
        if not use_for_score:
            continue
        x = compute_feature(kind, df_raw[key])
        # stale exclusion for scoring features
        if meta.get(key, {}).get("stale", True):
            x = x * np.nan
        win = pick_rolling_window(freq_group)
        z = zscore_past_only(x, win)
        feat[key] = z
        feat_raw[key] = x

    df_feat = pd.DataFrame(feat, index=idx_biz)

    # 3) Labels (forward events)
    events = compute_forward_events(df_raw["SP500"])
    labels = make_labels(events)

    # Labels are undefined near the end because forward shift needs future.
    # Use last_label_date to avoid training on rows without labels.
    last_label_idx = labels.dropna().index
    if last_label_idx.empty:
        raise RuntimeError("No labels computed (SP500 too short).")
    last_label_date = last_label_idx.max()

    # 4) Train/Val/Test split based on last_label_date
    end_ts = pd.Timestamp(last_label_date)
    test_start = end_ts - pd.DateOffset(years=TEST_YEARS)
    val_start = test_start - pd.DateOffset(years=VAL_YEARS)

    # dataset frame
    ds = df_feat.join(labels, how="inner")

    # valid rows: no NaN in features AND label columns
    valid_mask = ds[FEATURE_KEYS].notna().all(axis=1)
    ds = ds.loc[valid_mask].copy()
    ds = ds.dropna(subset=["event_any", "dd_2m", "crash_1d_fwd", "crash_3d_fwd"])

    if ds.empty or len(ds) < 2000:
        raise RuntimeError("Not enough training data after NaN filtering.")

    train = ds[ds.index < val_start]
    val = ds[(ds.index >= val_start) & (ds.index < test_start)]
    test = ds[(ds.index >= test_start) & (ds.index <= end_ts)]

    if len(train) < 1500 or len(val) < 300 or len(test) < 300:
        # fallback: simpler split
        cutoff = ds.index[int(len(ds) * 0.8)]
        train = ds[ds.index < cutoff]
        val = ds[(ds.index >= cutoff) & (ds.index <= end_ts)]
        test = val.copy()

    Xtr = train[FEATURE_KEYS].to_numpy(dtype=float)
    ytr = train["event_any"].to_numpy(dtype=int)

    # 5) Train supervised model to predict "event_any" within 2M
    w, b, tr_stats = train_logreg_ridge(
        Xtr, ytr, l2=LR_L2, lr=LR_LR, epochs=LR_EPOCHS, batch=LR_BATCH
    )

    # predict probabilities for whole ds (for backtest)
    Xall = ds[FEATURE_KEYS].to_numpy(dtype=float)
    pall = predict_prob(Xall, w, b)
    p_series = pd.Series(pall, index=ds.index, name="p_event_any")

    # 6) Pick thresholds using validation set (optimize but penalize too many crisis days)
    p_val = p_series.loc[val.index] if len(val) else p_series
    lbl_val = labels.loc[p_val.index]
    thr_crisis, thr_pick_info = pick_crisis_threshold(p_val, lbl_val)

    # Build notice/alert thresholds relative to crisis threshold using percentiles of p (validation)
    # (these are for UI, not "truth")
    pv = p_val.dropna()
    if pv.empty:
        thr_notice = 0.25
        thr_alert = 0.40
    else:
        # ensure ordering: notice < alert < crisis
        thr_notice = float(np.quantile(pv.values, 0.70))
        thr_alert = float(np.quantile(pv.values, 0.85))
        thr_notice = min(thr_notice, thr_alert - 1e-6)
        thr_alert = min(thr_alert, thr_crisis - 1e-6)

    regime = regimes_from_thresholds(p_series, thr_notice, thr_alert, thr_crisis)

    # 7) Backtest summaries (test period & full)
    labels_all = labels.loc[p_series.index]
    p_test = p_series.loc[test.index] if len(test) else p_series
    lbl_test = labels.loc[p_test.index]

    # threshold scans (this is the “latest.jsonだけで把握”用のログ)
    scan_q = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
    scan_val = scan_thresholds(p_val, labels.loc[p_val.index], scan_q)
    scan_test = scan_thresholds(p_test, labels.loc[p_test.index], scan_q)

    # Metrics at chosen crisis threshold
    metrics_val_at_pick = compute_metrics_for_threshold(p_val, labels.loc[p_val.index], thr_crisis)
    metrics_test_at_pick = compute_metrics_for_threshold(p_test, labels.loc[p_test.index], thr_crisis)

    by_regime_test = summarize_by_regime(p_test, regime.loc[p_test.index], labels.loc[p_test.index])

    # 8) Latest snapshot values (use idx_biz[-1] as asof)
    asof = str(idx_biz[-1].date())

    # latest probability score in 0..100
    latest_p = float(p_series.iloc[-1]) if pd.notna(p_series.iloc[-1]) else None
    latest_score = None if latest_p is None else float(SCORE_CENTER + SCORE_SCALE * latest_p)
    latest_regime = str(regime.iloc[-1]) if pd.notna(regime.iloc[-1]) else None

    # Debug: show latest raw & feature & z for each series
    latest_detail = {}
    for k in FEATURE_KEYS:
        latest_detail[k] = {
            "fred_id": meta.get(k, {}).get("fred_id"),
            "raw_level": None if pd.isna(df_raw[k].iloc[-1]) else float(df_raw[k].iloc[-1]),
            "feature_value": None if pd.isna(feat_raw[k].iloc[-1]) else float(feat_raw[k].iloc[-1]),
            "z": None if pd.isna(df_feat[k].iloc[-1]) else float(df_feat[k].iloc[-1]),
            "stale": bool(meta.get(k, {}).get("stale", True)),
            "last_obs_date": meta.get(k, {}).get("last_obs_date"),
            "stale_days": meta.get(k, {}).get("stale_days"),
        }

    # Coefficients (interpretation)
    coef = [{"feature": FEATURE_KEYS[i], "coef": float(w[i])} for i in range(len(FEATURE_KEYS))]
    coef.sort(key=lambda x: abs(x["coef"]), reverse=True)

    # Reference-only latest levels
    reference = {}
    for key, (_, _, _, use_for_score) in SERIES.items():
        if use_for_score:
            continue
        v = df_raw[key].iloc[-1]
        reference[key] = None if (isinstance(v, float) and math.isnan(v)) else float(v)

    # 9) Write latest.json with full debug (temporary)
    latest_obj = {
        "asof": asof,
        "horizon_bdays": HORIZON_BDAYS,

        # primary output
        "p_event_any": latest_p,
        "score": latest_score,
        "regime": latest_regime,

        # thresholds in probability space (not quantiles)
        "thresholds": {
            "notice_p": thr_notice,
            "alert_p": thr_alert,
            "crisis_p": thr_crisis,
        },

        # staleness
        "data_staleness": meta,

        # reference only
        "reference_only": reference,

        # definitions
        "definitions": {
            "regimes": ["平常", "注意", "警戒", "危機"],
            "events": {
                "crash_1d_fwd": f"{HORIZON_BDAYS}営業日以内に(任意のuで)1日先リターン<={int(CRASH_1D*100)}%",
                "crash_3d_fwd": f"{HORIZON_BDAYS}営業日以内に(任意のuで)3営業日先リターン<={int(CRASH_3D*100)}%",
                "dd_2m": f"{HORIZON_BDAYS}営業日以内に直近252営業日高値から{int(DD_2M*100)}%到達",
                "event_any": "上のいずれかが発生",
            },
            "stale_policy_A": STALE_DAYS,
            "note": "スコアは教師あり（event_anyの確率）。月次は参考表示のみ。週次が週末日付でも落ちないようにD→B整列。",
        },

        # backtest (ここを“latest.jsonだけで把握”できるように冗長にする)
        "backtest": {
            "windows": {
                "val_start": None if val.empty else str(val.index.min().date()),
                "test_start": None if test.empty else str(test.index.min().date()),
                "last_label_date": str(pd.Timestamp(last_label_date).date()),
            },
            "train_stats": tr_stats,
            "picked_threshold_info": thr_pick_info,
            "metrics_val_at_picked_crisis": metrics_val_at_pick,
            "metrics_test_at_picked_crisis": metrics_test_at_pick,
            "by_regime_test": by_regime_test,
            "threshold_scan_val": scan_val,
            "threshold_scan_test": scan_test,
        },

        # model debug
        "debug": {
            "latest_features": latest_detail,
            "coef_abs_sorted": coef[:20],
            "feature_keys": FEATURE_KEYS,
        },
    }

    ensure_dir("data")
    write_json("data/latest.json", latest_obj)

    # 10) history append (one row per run), keep 20 years
    hist_path = "data/history.csv.gz"
    hist = history_load(hist_path)
    new_row = {
        "date": pd.Timestamp(run_date),
        "p_event_any": latest_p,
        "score": latest_score,
        "regime": latest_regime,
        "notice_p": thr_notice,
        "alert_p": thr_alert,
        "crisis_p": thr_crisis,
    }
    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    hist = prune_history(hist, run_date, HISTORY_YEARS_TO_KEEP)
    history_save(hist_path, hist)

    # 11) Issues automation: open while "危機"
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
        lines.append(f"- p_event_any: {latest_p}")
        lines.append(f"- score: {latest_score}")
        lines.append(f"- regime: {latest_regime}")
        lines.append(f"- thresholds: notice={thr_notice:.4f} alert={thr_alert:.4f} crisis={thr_crisis:.4f}")
        lines.append("")
        lines.append("## Backtest (picked crisis threshold, test window)")
        mt = latest_obj["backtest"]["metrics_test_at_picked_crisis"]
        for k in sorted(mt.keys()):
            lines.append(f"- {k}: {mt[k]}")
        lines.append("")
        lines.append("## Latest features (top 8 by |coef|)")
        for c in coef[:8]:
            fk = c["feature"]
            ld = latest_detail.get(fk, {})
            lines.append(f"- {fk}: coef={c['coef']:.4f} z={ld.get('z')} feature={ld.get('feature_value')}")
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

    print("[done] updated data/latest.json (+debug), data/history.csv.gz, data/state.json")
    print(f"[result] p={latest_p} score={latest_score} regime={latest_regime}")


if __name__ == "__main__":
    main()
