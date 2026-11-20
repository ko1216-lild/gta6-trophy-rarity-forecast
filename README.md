# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ko1216-lild/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ko1216-lild/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01 — entry

A forecasting program for make.fun's bounty
*gta6-trophy-rarity-forecast-01*. `main.py` (Python 3.13, standard
library only, no network) reads one case file from the input
directory — every Exophase read of Grand Theft Auto VI's PlayStation 5
trophy earn rates recorded before the case's checkpoint — and writes,
for each of the game's 55 trophies, five quantiles (5%, 25%, 50%,
75%, 95%) of that trophy's earn rate at the case's target. The final
case is cut at the deadline, two days after launch; its target is one
week after launch, 2026-11-26 00:00 UTC. The score is the mean
pinball loss on the logit scale.

## Method

Per trophy, on the logit scale: a velocity from a recent-window
weighted least-squares fit (48 h window, 18 h half-life) blended with
a long-window slope, plus a second velocity from the
earners/players decomposition, which separates real progress from
dilution as the tracked player pool grows. The drift is damped
exponentially over the horizon (tau = 2 days) and capped, so a
multi-day horizon cannot run away; the spread combines a random-walk
term scaled by the case's own step volatility with the slope's
standard error. On backtests inside the kit's two practice cases the
model's mean pinball loss is about 25% below the persistence
baseline's. See NOTES.md for the full derivation and the backtest
protocol.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
