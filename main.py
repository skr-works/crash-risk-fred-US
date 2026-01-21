import os
import json
import time
import gzip
import math
import datetime as dt
from typing import Dict, List, Tuple, Optional

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

# Crash definitions (2M monitoring)
CRASH_1D = -0.10          # 1-day <= -10%
CRASH_3D = -0.15          # 3-day cumulative <= -15%
DD_2M = -0.15             # within 42 bdays, drawdown from trailing 252b peak <= -15%

# Regime quantiles
Q_PLAIN = 0.60
Q_NOTICE = 0.80
Q_ALERT = 0.92
# "危機" >= P92

SCORE_CENTER = 50.0
SCORE_SCALE = 12.5
Z_CLIP = 4.0

# Use shorter rolling window than 10y because 2M horizon is short.
ROLL_WIN_DAILY = 1260     # ~5 years business days
ROLL_WIN_WEEKLY = 1260    # same index (we align to business days anyway)

# Method A: fixed staleness thresholds
STALE_DAYS = {
    "daily": 7,
    "weekly": 21,
    "monthly": 120,     # reference only
    "quarterly": 220,   # reference only
}

# FRED polite rate limit
FRED_SLEEP_SEC = 0.60

ISSUE_COOLDOWN_DAYS = 14


# =========================
# Series (scoring uses only daily/weekly)
# =========================

# key: (fred_id, freq_group, feature_kind, use_for_score)
SERIES = {
    # Economy pulse (weekly) - core replacement
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
    "SP500":   ("SP500", "daily", "ret_2m_neg", True),

    # Reference only (monthly/quarterly) - NOT used for score
    "FEDFUNDS_REF": ("FEDFUNDS", "monthly", "diff_1m", False),
    "UNRATE_REF":   ("UNRATE", "monthly", "diff_1m", False),
}

# Blocks (equal weight; only blocks with at least 1 valid series participate)
BLOCKS = {
    "景気パルス": ["WEI"],
    "雇用":       ["ICSA", "CCSA"],
    "金利":       ["T10Y3M"],
    "信用":       ["HY_OAS", "IG_OAS"],
    "金融ストレス": ["NFCI", "STLFSI4"],
    "市場ストレス": ["VIX", "SP500"],
}


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

def business_index(start: str, end_date: dt.date) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end_date), freq="B")

def rolling_z(x: pd.Series, win: int) -> pd.Series:
    mu = x.rolling(win, min_periods=max(30, win // 10)).mean()
    sd = x.rolling(win, min_periods=max(30, win // 10)).std()
    z = (x - mu) / sd
    return z.clip(lower=-Z_CLIP, upper=Z_CLIP)

def safe_median(df: pd.DataFrame) -> pd.Series:
    return df.median(axis=1, skipna=True)

def quantile_thresholds(score: pd.Series) -> Dict[str, float]:
    s = score.dropna()
    if len(s) < 300:
        return {"P60": 55.0, "P80": 65.0, "P92": 75.0}
    return {
        "P60": float(s.quantile(Q_PLAIN)),
        "P80": float(s.quantile(Q_NOTICE)),
        "P92": float(s.quantile(Q_ALERT)),
    }

def regime_from_score(score: float, thr: Dict[str, float]) -> str:
    if score < thr["P60"]:
        return "平常"
    if score < thr["P80"]:
        return "注意"
    if score < thr["P92"]:
        return "警戒"
    return "危機"

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
# All outputs: larger => riskier
# =========================

def compute_feature(kind: str, s: pd.Series) -> pd.Series:
    b_1w = 5
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

    if kind == "ret_2m_neg":
        return -(s.pct_change(b_2m))

    if kind == "level_plus_diff_1m":
        # level + speed of widening (both risk-up)
        return s + (s - s.shift(b_1m))

    raise ValueError(f"Unknown feature kind: {kind}")


# =========================
# Event definitions (2M horizon)
# =========================

def compute_forward_events(sp500: pd.Series) -> pd.DataFrame:
    """
    For each date t, label whether within next HORIZON_BDAYS we observe:
      - crash_1d_fwd: any 1-day <= -10%
      - crash_3d_fwd: any 3-day cumulative <= -15%
      - dd_2m: within next 42 bdays, drawdown from trailing 252b peak <= -15%
    """
    # forward window (use reverse rolling)
    r1 = sp500.pct_change(1)
    r3 = sp500.pct_change(3)

    crash_1d = (r1 <= CRASH_1D).astype(int)
    crash_3d = (r3 <= CRASH_3D).astype(int)

    crash_1d_fwd = crash_1d[::-1].rolling(HORIZON_BDAYS, min_periods=1).max()[::-1].astype(int)
    crash_3d_fwd = crash_3d[::-1].rolling(HORIZON_BDAYS, min_periods=1).max()[::-1].astype(int)

    trailing_peak = sp500.rolling(252, min_periods=30).max()
    dd_now = (sp500 / trailing_peak) - 1.0
    dd_forward_min = dd_now[::-1].rolling(HORIZON_BDAYS, min_periods=10).min()[::-1]
    dd_2m = (dd_forward_min <= DD_2M).astype(int)

    return pd.DataFrame({
        "crash_1d_fwd": crash_1d_fwd,
        "crash_3d_fwd": crash_3d_fwd,
        "dd_2m": dd_2m,
    }, index=sp500.index)

def backtest_summary(score: pd.Series, regime: pd.Series, events: pd.DataFrame) -> dict:
    df = pd.DataFrame({"score": score, "regime": regime}).join(events, how="inner").dropna()
    if df.empty:
        return {"note": "no backtest data yet"}

    crisis = df[df["regime"] == "危機"]
    if crisis.empty:
        return {
            "crisis_days": 0,
            "hit_rate_dd_2m": None,
            "hit_rate_crash_1d_fwd": None,
            "hit_rate_crash_3d_fwd": None,
        }

    return {
        "crisis_days": int(len(crisis)),
        "hit_rate_dd_2m": float(crisis["dd_2m"].mean()),
        "hit_rate_crash_1d_fwd": float(crisis["crash_1d_fwd"].mean()),
        "hit_rate_crash_3d_fwd": float(crisis["crash_3d_fwd"].mean()),
    }


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

    idx = business_index(START_DATE, run_date)

    raw = {}
    meta = {}

    # Fetch all series (including reference-only)
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
            raw[key] = pd.Series(np.nan, index=idx, name=key)
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
            raw[key] = pd.Series(np.nan, index=idx, name=key)
            time.sleep(FRED_SLEEP_SEC)
            continue

        last_obs = s.dropna().index.max() if s.dropna().shape[0] else pd.NaT
        stale_days, stale_flag = native_stale(freq_group, last_obs, run_date)

        s_aligned = s.reindex(idx).ffill()

        raw[key] = s_aligned
        meta[key] = {
            "fred_id": fred_id,
            "freq_group": freq_group,
            "last_obs_date": None if pd.isna(last_obs) else str(last_obs.date()),
            "stale_days": int(stale_days) if stale_days < 10**8 else None,
            "stale": bool(stale_flag),
        }

        time.sleep(FRED_SLEEP_SEC)

    df_raw = pd.DataFrame(raw, index=idx)

    # Compute z-scores for scoring series only (daily/weekly)
    z_map = {}
    x_map = {}

    for key, (_, freq_group, kind, use_for_score) in SERIES.items():
        s = df_raw[key]
        x = compute_feature(kind, s)
        x_map[key] = x

        if not use_for_score:
            # reference-only: do not produce z for score
            continue

        # exclude if stale (method A)
        if meta.get(key, {}).get("stale", True):
            x = x * np.nan

        win = ROLL_WIN_WEEKLY if freq_group == "weekly" else ROLL_WIN_DAILY
        z_map[key] = rolling_z(x, win)

    df_z = pd.DataFrame(z_map, index=idx)

    # Block scores (median of z within block)
    block_scores = {}
    block_rep = {}

    for bname, keys in BLOCKS.items():
        present = [k for k in keys if k in df_z.columns]
        if not present:
            block_scores[bname] = pd.Series(np.nan, index=idx)
            block_rep[bname] = (None, float("nan"))
            continue

        z_block = df_z[present].copy()
        bs = safe_median(z_block)
        block_scores[bname] = bs

        # representative driver at latest date: max z
        latest_vals = z_block.iloc[-1].to_dict()
        best_k, best_z = None, float("-inf")
        for k, v in latest_vals.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if float(v) > best_z:
                best_z = float(v)
                best_k = k
        block_rep[bname] = (best_k, best_z if best_k is not None else float("nan"))

    df_block = pd.DataFrame(block_scores, index=idx)

    # Combine blocks with equal weight, renormalize for missing blocks
    blocks_list = list(BLOCKS.keys())
    w = {b: 1.0 / len(blocks_list) for b in blocks_list}

    def combine_row(row: pd.Series) -> float:
        vals = {}
        for b in blocks_list:
            v = row.get(b)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            vals[b] = float(v)
        if not vals:
            return float("nan")
        tot = sum(w[b] for b in vals.keys())
        return sum(vals[b] * (w[b] / tot) for b in vals.keys())

    score_raw = df_block.apply(combine_row, axis=1)
    score = (SCORE_CENTER + SCORE_SCALE * score_raw).clip(lower=0, upper=100)

    thr = quantile_thresholds(score)
    regime = score.apply(lambda v: regime_from_score(float(v), thr) if pd.notna(v) else None)

    # Backtest: forward events within 2 months
    events = compute_forward_events(df_raw["SP500"])
    bt = backtest_summary(score, regime, events)

    latest_date = str(idx[-1].date())
    latest_score = float(score.iloc[-1]) if pd.notna(score.iloc[-1]) else None
    latest_regime = str(regime.iloc[-1]) if pd.notna(regime.iloc[-1]) else None

    # coverage (how many blocks are valid today)
    valid_blocks_today = int(pd.notna(df_block.iloc[-1]).sum())
    coverage = valid_blocks_today / len(blocks_list)

    # top drivers by block representative z
    drivers = []
    for b, (rep_key, rep_z) in block_rep.items():
        if rep_key is None or rep_z is None or (isinstance(rep_z, float) and math.isnan(rep_z)):
            continue
        drivers.append({"block": b, "series_key": rep_key, "z": float(rep_z)})
    drivers.sort(key=lambda x: x["z"], reverse=True)
    drivers = drivers[:3]

    # Reference snapshot (monthly/quarterly) - latest levels only
    reference = {}
    for key, (_, _, _, use_for_score) in SERIES.items():
        if use_for_score:
            continue
        v = df_raw[key].iloc[-1]
        reference[key] = None if (isinstance(v, float) and math.isnan(v)) else float(v)

    latest_obj = {
        "asof": latest_date,
        "horizon_bdays": HORIZON_BDAYS,
        "score": latest_score,
        "regime": latest_regime,
        "coverage": coverage,
        "thresholds": thr,
        "block_scores": {k: (None if pd.isna(df_block[k].iloc[-1]) else float(df_block[k].iloc[-1])) for k in df_block.columns},
        "top_drivers": drivers,
        "data_staleness": meta,
        "reference_only": reference,
        "backtest_summary": bt,
        "definitions": {
            "regimes": ["平常", "注意", "警戒", "危機"],
            "events": {
                "crash_1d_fwd": f"{HORIZON_BDAYS}営業日以内に1日リターン<={int(CRASH_1D*100)}%",
                "crash_3d_fwd": f"{HORIZON_BDAYS}営業日以内に3日累積リターン<={int(CRASH_3D*100)}%",
                "dd_2m": f"{HORIZON_BDAYS}営業日以内に直近252営業日高値から{int(DD_2M*100)}%到達",
            },
            "stale_policy_A": STALE_DAYS,
            "note": "スコア計算は日次・週次のみ。月次・四半期は参考表示のみ。",
        },
    }

    ensure_dir("data")
    write_json("data/latest.json", latest_obj)

    # history append (one row per run), keep 20 years
    hist_path = "data/history.csv.gz"
    hist = history_load(hist_path)
    new_row = {
        "date": pd.Timestamp(run_date),
        "score": latest_score,
        "regime": latest_regime,
        "coverage": coverage,
        "P60": thr["P60"],
        "P80": thr["P80"],
        "P92": thr["P92"],
        "bt_crisis_days": bt.get("crisis_days"),
        "bt_hit_dd_2m": bt.get("hit_rate_dd_2m"),
        "bt_hit_crash_1d_fwd": bt.get("hit_rate_crash_1d_fwd"),
        "bt_hit_crash_3d_fwd": bt.get("hit_rate_crash_3d_fwd"),
    }
    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    hist = prune_history(hist, run_date, HISTORY_YEARS_TO_KEEP)
    history_save(hist_path, hist)

    # Issues automation (same semantics: open while "危機")
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
        lines.append(f"- date: {latest_date}")
        lines.append(f"- score: {latest_score}")
        lines.append(f"- regime: {latest_regime}")
        lines.append(f"- coverage: {coverage:.2f}")
        lines.append(f"- thresholds: P60={thr['P60']:.2f}, P80={thr['P80']:.2f}, P92={thr['P92']:.2f}")
        lines.append("")
        lines.append("## 2か月イベント定義")
        lines.append(f"- crash_1d_fwd: {HORIZON_BDAYS}営業日以内に1日<= {int(CRASH_1D*100)}%")
        lines.append(f"- crash_3d_fwd: {HORIZON_BDAYS}営業日以内に3日累積<= {int(CRASH_3D*100)}%")
        lines.append(f"- dd_2m: {HORIZON_BDAYS}営業日以内にピーク比 {int(DD_2M*100)}%")
        lines.append("")
        lines.append("## ブロック別スコア（z中央値）")
        for b in blocks_list:
            v = latest_obj["block_scores"].get(b)
            lines.append(f"- {b}: {v}")
        lines.append("")
        lines.append("## 上位ドライバー（危険方向）")
        for d in latest_obj["top_drivers"]:
            k = d["series_key"]
            fid = meta.get(k, {}).get("fred_id") if k else None
            lines.append(f"- {d['block']}: {k} ({fid}) z={d['z']:.2f}")
        lines.append("")
        lines.append("## stale（除外対象）")
        stale_keys = [k for k, m in meta.items() if m.get("stale") and SERIES.get(k, (None,None,None,False))[3]]
        if stale_keys:
            for k in stale_keys:
                m = meta[k]
                lines.append(f"- {k}: last_obs={m.get('last_obs_date')} stale_days={m.get('stale_days')} err={m.get('error')}")
        else:
            lines.append("- stale なし")
        lines.append("")
        lines.append("## バックテスト（2000年以降、2か月先の発生率）")
        lines.append(f"- 危機日数: {bt.get('crisis_days')}")
        lines.append(f"- 命中率 dd_2m: {bt.get('hit_rate_dd_2m')}")
        lines.append(f"- 命中率 crash_1d_fwd: {bt.get('hit_rate_crash_1d_fwd')}")
        lines.append(f"- 命中率 crash_3d_fwd: {bt.get('hit_rate_crash_3d_fwd')}")
        return "\n".join(lines)

    entered_crisis = (prev_regime != "危機") and (latest_regime == "危機")
    exited_crisis = (prev_regime == "危機") and (latest_regime != "危機")

    try:
        if entered_crisis:
            if in_cooldown(run_date):
                print("[issue] Entered crisis but in cooldown; skipping.")
            else:
                title = f"暴落リスク: 危機（2か月監視）（{latest_date}）"
                num = gh_create_issue(title, issue_body("危機に遷移しました（2か月先の暴落監視）。"))
                state["open_issue_number"] = num
                state["last_issue_date"] = latest_date
                print(f"[issue] Created issue #{num}")

        elif latest_regime == "危機" and open_issue:
            gh_comment_issue(int(open_issue), issue_body("危機継続の更新です。"))
            state["last_issue_date"] = latest_date
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

    print("[done] updated data/latest.json, data/history.csv.gz, data/state.json")
    print(f"[result] score={latest_score} regime={latest_regime} coverage={coverage:.2f}")


if __name__ == "__main__":
    main()
