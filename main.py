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
# Config
# =========================

START_DATE = "2000-01-01"
HISTORY_YEARS_TO_KEEP = 20  # rolling retention
RUN_TZ_LABEL = "JST"  # label only (we run in UTC on Actions)
ISSUE_COOLDOWN_DAYS = 14

# Regime quantiles (computed from backtest score distribution)
Q_PLAIN = 0.60   # 平常 < P60
Q_NOTICE = 0.80  # 注意  P60..P80
Q_ALERT = 0.92   # 警戒  P80..P92
# 危機 >= P92

# Score scaling
SCORE_CENTER = 50.0
SCORE_SCALE = 12.5  # Score = 50 + 12.5*ScoreRaw

# z-score handling
Z_CLIP = 4.0
ROLL_WIN_DAILY = 2520   # ~10 years business days
ROLL_WIN_QUARTERLY = 5040  # ~20 years business days (for very sparse series)

# Staleness thresholds by "native" frequency category
STALE_DAYS = {
    "daily": 5,
    "weekly": 14,
    "monthly": 45,
    "quarterly": 120,
}

# FRED series and features
# feature_kind: larger => riskier after transform
SERIES = {
    # Core
    "USSLIND":   ("USSLIND", "monthly",  "diff_6m_neg"),
    "ICSA":      ("ICSA",    "weekly",   "log_diff_13w"),
    "T10Y3M":    ("T10Y3M",  "daily",    "level_neg"),
    "FEDFUNDS":  ("FEDFUNDS","monthly",  "diff_6m"),
    "HYOAS":     ("BAMLH0A0HYM2", "daily","level"),
    "SP500":     ("SP500",   "daily",    "ret_6m_neg"),
    "VIX":       ("VIXCLS",  "daily",    "level"),

    # Extended (default ON)
    "NFCI":      ("NFCI",    "weekly",   "level"),
    "ANFCI":     ("ANFCI",   "weekly",   "level"),
    "STLFSI4":   ("STLFSI4", "weekly",   "level"),
    "BAA10Y":    ("BAA10Y",  "daily",    "level"),

    # Replacement for removed ISM series:
    # NEWORDER: Nondefense Capital Goods Ex Aircraft
    # DGORDER: Durable Goods
    "NEWORDER":  ("NEWORDER","monthly",  "log_diff_6m_neg"),
    "DGORDER":   ("DGORDER", "monthly",  "log_diff_6m_neg"),

    "PERMIT":    ("PERMIT",  "monthly",  "log_diff_6m_neg"),
    "AMTMNO":    ("AMTMNO",  "monthly",  "log_diff_6m_neg"),
    "SLOOS_CI":  ("DRTSCILM","quarterly","level"),
}

# Block definitions (median of z within block)
BLOCKS = {
    "景気先行": ["USSLIND", "AMTMNO", "PERMIT", "NEWORDER", "DGORDER"],
    "雇用":     ["ICSA"],
    "金利政策": ["T10Y3M", "FEDFUNDS"],
    "信用":     ["HYOAS", "BAA10Y", "SLOOS_CI"],
    "ストレス": ["NFCI", "ANFCI", "STLFSI4", "VIX", "SP500"],
}
BLOCK_WEIGHT = 1.0 / len(BLOCKS)

# FRED polite rate limit (<= 2 req/sec)
FRED_SLEEP_SEC = 0.60


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
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"APP_SECRETS must be valid JSON. Error: {e}") from e
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
        # let caller handle 400/404 etc.
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
    threshold = STALE_DAYS.get(freq_group, 14)
    return (delta_days, delta_days > threshold)

def business_index(start: str, end_date: dt.date) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end_date), freq="B")

def compute_feature(kind: str, s: pd.Series) -> pd.Series:
    b_6m = 126   # ~6 months in business days
    b_13w = 65   # 13 weeks in business days

    if kind == "level":
        return s
    if kind == "level_neg":
        return -s
    if kind == "diff_6m":
        return s - s.shift(b_6m)
    if kind == "diff_6m_neg":
        return -(s - s.shift(b_6m))
    if kind == "log_diff_6m_neg":
        ls = np.log(s.replace(0, np.nan))
        ls = ls.replace([np.inf, -np.inf], np.nan)
        return -(ls - ls.shift(b_6m))
    if kind == "log_diff_13w":
        ls = np.log(s.replace(0, np.nan))
        ls = ls.replace([np.inf, -np.inf], np.nan)
        return ls - ls.shift(b_13w)
    if kind == "ret_6m_neg":
        return -(s.pct_change(b_6m))
    raise ValueError(f"Unknown feature kind: {kind}")

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

def compute_events(sp500: pd.Series) -> pd.DataFrame:
    r1 = sp500.pct_change(1)
    crash_1d = (r1 <= -0.10)

    r3 = sp500.pct_change(3)
    crash_3d = (r3 <= -0.20)

    trailing_peak = sp500.rolling(252, min_periods=30).max()
    dd_now = (sp500 / trailing_peak) - 1.0
    dd_forward_min = dd_now[::-1].rolling(126, min_periods=10).min()[::-1]
    dd_6m = (dd_forward_min <= -0.20)

    return pd.DataFrame({
        "crash_1d": crash_1d.astype(int),
        "crash_3d": crash_3d.astype(int),
        "dd_6m": dd_6m.astype(int),
    }, index=sp500.index)

def backtest_summary(score: pd.Series, regime: pd.Series, events: pd.DataFrame) -> dict:
    df = pd.DataFrame({"score": score, "regime": regime}).join(events, how="inner").dropna()
    if df.empty:
        return {"note": "no backtest data yet"}
    crisis = df[df["regime"] == "危機"]
    if crisis.empty:
        return {"crisis_days": 0, "hit_rate_dd_6m": None, "hit_rate_crash_1d": None, "hit_rate_crash_3d": None}
    return {
        "crisis_days": int(len(crisis)),
        "hit_rate_dd_6m": float(crisis["dd_6m"].mean()),
        "hit_rate_crash_1d": float(crisis["crash_1d"].mean()),
        "hit_rate_crash_3d": float(crisis["crash_3d"].mean()),
    }

def top_drivers(per_block_rep: Dict[str, Tuple[str, float]]) -> List[dict]:
    items = []
    for b, (rep_key, rep_z) in per_block_rep.items():
        if rep_key is None or rep_z is None or (isinstance(rep_z, float) and math.isnan(rep_z)):
            continue
        items.append({"block": b, "series_key": rep_key, "z": float(rep_z)})
    items.sort(key=lambda x: x["z"], reverse=True)
    return items[:3]


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

    for key, (fred_id, freq_group, _) in SERIES.items():
        try:
            s = fred_get_series_obs(api_key, fred_id, START_DATE)
        except requests.exceptions.HTTPError as e:
            # Do not kill the run; mark as unavailable and exclude from score
            status = getattr(e.response, "status_code", None)
            print(f"[fred] skip {key} ({fred_id}): HTTP {status}")
            s = pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"), name=fred_id)
            meta[key] = {
                "fred_id": fred_id,
                "freq_group": freq_group,
                "last_obs_date": None,
                "stale_days": None,
                "stale": True,
                "error": f"HTTPError {status}",
            }
            time.sleep(FRED_SLEEP_SEC)
            raw[key] = pd.Series(np.nan, index=idx, name=key)
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
            time.sleep(FRED_SLEEP_SEC)
            raw[key] = pd.Series(np.nan, index=idx, name=key)
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

    zs = {}
    for key, (_, freq_group, kind) in SERIES.items():
        s = df_raw[key]
        x = compute_feature(kind, s)

        if meta.get(key, {}).get("stale", True):
            x = x * np.nan

        win = ROLL_WIN_QUARTERLY if freq_group == "quarterly" else ROLL_WIN_DAILY
        zs[key] = rolling_z(x, win)

    df_z = pd.DataFrame(zs, index=idx)

    block_scores = {}
    block_rep = {}
    for bname, keys in BLOCKS.items():
        z_block = df_z[keys].copy()
        bs = safe_median(z_block)
        block_scores[bname] = bs

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

    w = {b: BLOCK_WEIGHT for b in BLOCKS.keys()}

    def combine_row(row: pd.Series) -> float:
        vals = {}
        for b in BLOCKS.keys():
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
    regime = score.apply(lambda v: regime_from_score(float(v), thr) if not (isinstance(v, float) and math.isnan(v)) else None)

    events = compute_events(df_raw["SP500"])
    bt = backtest_summary(score, regime, events)

    latest_date = str(idx[-1].date())
    latest_score = float(score.iloc[-1]) if pd.notna(score.iloc[-1]) else None
    latest_regime = str(regime.iloc[-1]) if pd.notna(regime.iloc[-1]) else None

    drivers = top_drivers(block_rep)

    latest_obj = {
        "asof": latest_date,
        "score": latest_score,
        "regime": latest_regime,
        "thresholds": thr,
        "block_scores": {k: (None if pd.isna(df_block[k].iloc[-1]) else float(df_block[k].iloc[-1])) for k in df_block.columns},
        "top_drivers": drivers,
        "data_staleness": meta,
        "backtest_summary": bt,
        "definitions": {
            "regimes": ["平常", "注意", "警戒", "危機"],
            "events": {
                "crash_1d": "1日リターン<=-10%",
                "crash_3d": "3営業日累積リターン<=-20%",
                "dd_6m": "126営業日以内に過去252営業日高値から-20%到達",
            }
        }
    }

    ensure_dir("data")
    write_json("data/latest.json", latest_obj)

    hist_path = "data/history.csv.gz"
    hist = history_load(hist_path)
    new_row = {
        "date": pd.Timestamp(run_date),
        "score": latest_score,
        "regime": latest_regime,
        "P60": thr["P60"],
        "P80": thr["P80"],
        "P92": thr["P92"],
        "bt_crisis_days": bt.get("crisis_days"),
        "bt_hit_dd_6m": bt.get("hit_rate_dd_6m"),
        "bt_hit_crash_1d": bt.get("hit_rate_crash_1d"),
        "bt_hit_crash_3d": bt.get("hit_rate_crash_3d"),
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
        lines.append(f"- date: {latest_date}")
        lines.append(f"- score: {latest_score}")
        lines.append(f"- regime: {latest_regime}")
        lines.append(f"- thresholds: P60={thr['P60']:.2f}, P80={thr['P80']:.2f}, P92={thr['P92']:.2f}")
        lines.append("")
        lines.append("## ブロック別スコア")
        for b in BLOCKS.keys():
            v = latest_obj["block_scores"].get(b)
            lines.append(f"- {b}: {v}")
        lines.append("")
        lines.append("## 上位ドライバー")
        for d in drivers:
            k = d["series_key"]
            fid = meta.get(k, {}).get("fred_id") if k else None
            lines.append(f"- {d['block']}: {k} ({fid}) z={d['z']:.2f}")
        lines.append("")
        lines.append("## stale（除外対象）")
        stale_keys = [k for k, m in meta.items() if m.get("stale")]
        if stale_keys:
            for k in stale_keys:
                m = meta[k]
                lines.append(f"- {k}: last_obs={m.get('last_obs_date')} stale_days={m.get('stale_days')} err={m.get('error')}")
        else:
            lines.append("- stale なし")
        return "\n".join(lines)

    entered_crisis = (prev_regime != "危機") and (latest_regime == "危機")
    exited_crisis = (prev_regime == "危機") and (latest_regime != "危機")

    try:
        if entered_crisis:
            if in_cooldown(run_date):
                print("[issue] Entered crisis but in cooldown; skipping.")
            else:
                title = f"暴落リスク: 危機（{latest_date}）"
                num = gh_create_issue(title, issue_body("危機に遷移しました。"))
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
    print(f"[result] score={latest_score} regime={latest_regime}")


if __name__ == "__main__":
    main()
