# gta6-trophy-rarity-forecast-01 — model notes

## Model (main.py, Python 3.13, standard library only)

Per trophy, on the logit scale of the earn rate (percent):

- **Velocity**: weighted least-squares slope of logit(rate) vs time over a recent 48 h window
  (18 h half-life), blended 65/35 with a long-window slope (72 h half-life). Slopes are shrunk
  toward zero by sxx/(sxx+0.05) so a few reads minutes apart are not trusted as a per-day
  velocity, and capped at ±2.5 logit/day.
- **Decomposition velocity**: with earners = rate × players_tracked / 100, the identity
  d/dt logit(rate) = (g_earners − g_players) / (1 − rate/100) gives a second velocity from the
  recent log-growth of earners and of the tracked pool; blended 55/45 with the direct velocity
  when the last earner count is ≥ 3. This separates player progress from pool dilution.
- **Drift**: velocity × tau × (1 − exp(−H/tau)), tau = 2 days, so multi-day horizons saturate
  instead of extrapolating a launch-week slope linearly; total drift capped to [−2.5, +3.5]
  logit; median rate softly capped at 92% and floored at 0.02% (a trophy already past a cap
  keeps its last rate).
- **Spread**: sigma² = 0.03² + rw²·H_days + (1.5·slope_SE·H_eff)², where rw² = 0.34² + vol²
  and vol is the case's own pooled per-trophy step volatility (median |Δlogit|/√Δt / 0.6745,
  capped 0.9; rw capped 0.95). Quiet histories stay narrow, jumpy ones widen; +0.15 in
  quadrature when the last read lacks the trophy; sigma capped at 1.75.
- Null rates are skipped per trophy; a never-seen trophy gets a token 0.05% median with
  sigma 0.6. No randomness — two runs of a case are byte-identical.

## Backtest (inside the practice cases)

Method: for each practice case, cut the history at every read k ≥ 2nd, pretend the case ends
there, forecast (a) the next read and (b) the case's final read, and score mean pinball loss
on the logit scale (the contest's scoring) against the actual read. The practice histories
are flat (Exophase had not refreshed yet: 638 players, identical reads), so this validates
the spread calibration and the no-trend behaviour; trend behaviour was checked separately on
a synthetic 200-read / 2.5-day history with a 4.52-day horizon (fast riser 18% → median
47%, dilution faller 9% → 4.7%, flat 0.6% → 0.60%; runtime 0.19 s).

| case | persistence (starter) | this model |
|------|----------------------|------------|
| p00  | 0.019200 | **0.014454** |
| p01  | 0.020124 | **0.015051** |
| overall | 0.019704 | **0.014779** (−25.0%) |

This model also wins on both sub-splits (next-read 0.011431 vs 0.014350; final-read
0.017477/0.018670 vs 0.024049/0.025899). Spread constants (SIGMA_RW 0.34, SIGMA_FLOOR 0.03)
were picked from a small grid on this backtest, kept mid-range rather than at the corner the
flat data rewards, because the pooled-volatility term — zero on the flat practice histories —
is what widens the spread on the final's moving history.

## Practice run check

`MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py`:
both cases valid — exactly one `trophies` key, 55 rows, ids matching the case, five finite
non-decreasing quantiles in [0, 100], 2,673 bytes per file (cap 65,536), ~0.7 s for both
cases, and two runs byte-identical.
