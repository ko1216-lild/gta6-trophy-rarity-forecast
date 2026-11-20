"""gta6-trophy-rarity-forecast-01 entry: damped logit-trend forecaster with an earners/players decomposition.

Model (per trophy, all on the logit scale of the earn rate in percent):

  * velocity: a weighted least-squares slope of logit(rate) against time over a recent window
    (48 h, 18 h half-life), blended 65/35 with a long-window slope, shrunk toward zero when the
    weighted time-spread of the reads is small, and capped at +/-2.5 logit/day;
  * decomposition: the same history gives earners = rate * players_tracked / 100 and the tracked
    count itself; since d/dt logit(rate) = (g_earners - g_players) / (1 - rate/100), their
    recent log-growth rates imply a second velocity, blended 55/45 with the direct one when the
    earner counts are large enough to trust. This separates real progress (earners growing)
    from dilution (the tracked pool growing), which a rate-only slope confuses;
  * drift damping: the drift is velocity * tau * (1 - exp(-H / tau)) with tau = 2 days, so a
    multi-day horizon cannot extrapolate a launch-week slope without bound; the total drift is
    capped to [-2.5, +3.5] logit and the median rate is softly capped at 92% and floored at
    0.02% (a trophy already past a cap keeps its last rate);
  * spread: sigma^2 = floor^2 + rw^2 * H_days + (1.5 * slope_se * H_eff)^2, where rw combines a
    base random-walk coefficient with the case's own pooled per-trophy step volatility, so
    quiet histories stay narrow and jumpy ones widen; sigma is capped at 1.75 logit.

Reads are taken as displayed (Exophase refreshes on its own clock; the answer is a displayed
read too). Null rates are skipped; a trophy missing from the last read falls back to its latest
non-null rate with a wider spread. Pure standard library, no randomness: two runs of the same
case write the same bytes.

I/O contract (same as the kit starter): read every <id>.in in $MAKEFUN_INPUT_DIR (default
/input), write <id>.out to $MAKEFUN_OUTPUT_DIR (default /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}, one row per trophy.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs with -P, which keeps this file's directory off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- velocity estimation ---
VEL_WINDOW_H = 48.0        # recent window for slopes
VEL_HALFLIFE_H = 18.0      # weight half-life inside the recent window
LONG_HALFLIFE_H = 72.0     # weight half-life for the long-window slope
W_RECENT = 0.65            # blend weight of the recent slope
SHRINK_SXX = 0.05          # weighted time-spread (day^2) at which a slope is half trusted
VEL_CAP = 2.5              # logit units per day

# --- drift damping and caps ---
TAU_DRIFT_D = 2.0          # damping time of the drift, days
DRIFT_UP = 3.5             # logit
DRIFT_DN = -2.5            # logit
RATE_CAP = 92.0            # percent; no week-one median above this unless already there
RATE_FLOOR = 0.02          # percent

# --- spread ---
SIGMA_FLOOR = 0.03
SIGMA_RW = 0.34            # base random-walk coefficient, logit per sqrt-day
VOL_CAP = 0.9              # cap on the pooled step-volatility estimate
RW_CAP = 0.95              # cap on the combined random-walk coefficient
SE_SPREAD_FACTOR = 1.5     # slope standard error -> spread multiplier
SIGMA_MAX = 1.75
STALE_SIGMA = 0.15         # extra spread when the last read lacks the trophy


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def logistic_pct(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (like JavaScript's Math.round); whole numbers stay ints."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> float:
    """ISO 8601 timestamp -> epoch seconds. Falls back to 0.0 on anything unexpected."""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def wls_slope(ts: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    """Weighted least-squares slope of ys on ts (days), shrunk toward 0, plus its standard error.

    The shrinkage keeps a slope fitted on a few reads minutes apart from being trusted as a
    per-day velocity: the slope is scaled by sxx / (sxx + SHRINK_SXX).
    """
    sw = sum(ws)
    if sw <= 0 or len(ts) < 2:
        return 0.0, 0.0
    mt = sum(w * t for t, w in zip(ts, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (t - mt) ** 2 for t, w in zip(ts, ws))
    if sxx <= 1e-12:
        return 0.0, 0.0
    sxy = sum(w * (t - mt) * (y - my) for t, y, w in zip(ts, ys, ws))
    slope = sxy / sxx
    resid = sum(w * (y - my - slope * (t - mt)) ** 2 for t, y, w in zip(ts, ys, ws)) / sw
    se = math.sqrt(resid / sxx) if resid > 0 else 0.0
    shrink = sxx / (sxx + SHRINK_SXX)
    return slope * shrink, se


def window_weights(ts_h: list[float], t_ref: float, halflife_h: float) -> list[float]:
    return [0.5 ** ((t_ref - t) / halflife_h) for t in ts_h]


def damped(days_horizon: float, tau_days: float) -> float:
    """Effective days of drift for a horizon under exponential damping."""
    return tau_days * (1.0 - math.exp(-days_horizon / tau_days))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_d = case["horizon_minutes"] / 1440.0
    h_eff = damped(horizon_d, TAU_DRIFT_D)

    # Read times in hours from the first read; unparseable times fall back to slot order.
    raw_times = [parse_time(read.get("at") or read.get("slot") or "") for read in history]
    if all(t == 0.0 for t in raw_times):
        raw_times = [i * 1800.0 for i in range(len(history))]
    t0 = raw_times[0]
    times_h = [(t - t0) / 3600.0 for t in raw_times]
    t_last = times_h[-1]

    # Players-tracked series (global to the case) and its recent log-growth per day.
    n_pts = [(times_h[j], history[j]["players_tracked"]) for j in range(len(history))
             if history[j].get("players_tracked")]
    g_players = 0.0
    n_last = None
    if n_pts:
        n_last = n_pts[-1][1]
    if len(n_pts) >= 3 and n_last:
        win = [(t, v) for t, v in n_pts if t >= t_last - VEL_WINDOW_H and v > 0]
        if len(win) >= 3 and win[-1][0] > win[0][0]:
            ts_d = [t / 24.0 for t, _ in win]
            g_players, _ = wls_slope(ts_d, [math.log(v) for _, v in win],
                                     window_weights([t for t, _ in win], win[-1][0], VEL_HALFLIFE_H))
            g_players = min(1.5, max(-1.5, g_players))

    # Pooled step volatility of logit rates across all trophies (robust, median based):
    # how much a displayed rate typically moves per square-root day in this case's history.
    steps = []
    for i in range(n):
        prev = None
        for j in range(len(history)):
            rate = history[j]["rates"][i] if i < len(history[j]["rates"]) else None
            if rate is None:
                continue
            if prev is not None:
                dt_d = (times_h[j] - prev[0]) / 24.0
                if dt_d > 0:
                    steps.append(abs(logit(rate) - prev[1]) / math.sqrt(dt_d))
            prev = (times_h[j], logit(rate))
    vol = 0.0
    if steps:
        steps.sort()
        vol = min(VOL_CAP, steps[len(steps) // 2] / 0.6745)
    rw = min(RW_CAP, math.sqrt(SIGMA_RW ** 2 + vol ** 2))

    rows = []
    for i, trophy in enumerate(trophies):
        pts = []  # (time_h, rate, players_tracked at that read)
        for j in range(len(history)):
            rates = history[j]["rates"]
            rate = rates[i] if i < len(rates) else None
            if rate is not None:
                pts.append((times_h[j], rate, history[j].get("players_tracked")))
        stale = not pts or pts[-1][0] < t_last
        if not pts:
            # Never seen: a token rate with a wide spread.
            lg, sigma = logit(0.05), 0.6
            rows.append({"id": trophy["id"],
                         "q": [round4(logistic_pct(lg + z * sigma)) for z in Z]})
            continue
        t_i, r_last, _ = pts[-1]
        l_last = logit(r_last)

        # Direct velocity on the logit scale: recent window blended with the long window.
        win = [(t, r) for t, r, _ in pts if t >= t_i - VEL_WINDOW_H]
        if len(win) >= 2:
            v_recent, se_recent = wls_slope(
                [t / 24.0 for t, _ in win], [logit(r) for _, r in win],
                window_weights([t for t, _ in win], t_i, VEL_HALFLIFE_H))
        else:
            v_recent, se_recent = 0.0, 0.0
        if len(pts) >= 2:
            v_long, _ = wls_slope(
                [t / 24.0 for t, _, _ in pts], [logit(r) for _, r, _ in pts],
                window_weights([t for t, _, _ in pts], t_i, LONG_HALFLIFE_H))
        else:
            v_long = 0.0
        v = W_RECENT * v_recent + (1.0 - W_RECENT) * v_long

        # Decomposition velocity: earners vs tracked-pool growth, on the same recent window.
        # d/dt logit(rate) = (g_earners - g_players) / (1 - rate/100); blending it with the
        # direct slope separates progress from dilution. Skipped when earner counts are tiny.
        e_last = r_last / 100.0 * pts[-1][2] if pts[-1][2] else None
        if e_last is not None and e_last >= 3.0:
            e_win = [(t, r / 100.0 * p) for t, r, p in pts
                     if p and t >= t_i - VEL_WINDOW_H]
            if len(e_win) >= 3 and e_win[-1][0] > e_win[0][0]:
                g_earn, _ = wls_slope(
                    [t / 24.0 for t, _ in e_win],
                    [math.log(max(e, 0.5)) for _, e in e_win],
                    window_weights([t for t, _ in e_win], t_i, VEL_HALFLIFE_H))
                g_earn = min(2.0, max(-1.0, g_earn))
                v_alt = (g_earn - g_players) / max(0.05, 1.0 - r_last / 100.0)
                v = 0.55 * v + 0.45 * v_alt
        v = min(VEL_CAP, max(-VEL_CAP, v))
        l_med = l_last + v * h_eff

        # Caps: bounded total drift, soft ceiling/floor on the median rate.
        drift = min(DRIFT_UP, max(DRIFT_DN, l_med - l_last))
        l_med = l_last + drift
        l_med = min(l_med, max(l_last, logit(RATE_CAP)))
        l_med = max(l_med, min(l_last, logit(RATE_FLOOR)))

        # Spread: floor + random walk over the horizon + velocity uncertainty over the
        # damped horizon; wider when the last read did not list this trophy.
        sig_vel = min(1.0, SE_SPREAD_FACTOR * se_recent) * h_eff
        sigma = math.sqrt(SIGMA_FLOOR ** 2 + rw ** 2 * horizon_d + sig_vel ** 2)
        if stale:
            sigma = math.sqrt(sigma ** 2 + STALE_SIGMA ** 2)
        sigma = min(SIGMA_MAX, max(SIGMA_FLOOR, sigma))

        rows.append({"id": trophy["id"],
                     "q": [round4(logistic_pct(l_med + z * sigma)) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
