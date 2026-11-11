# Fatigue scoring

**Purpose:** turn a temperature trace into one number, "how much wear", so that two ways of running the same work can be compared.

**In one line:** count every heat-up/cool-down cycle (rainflow counting, a stack algorithm), score each by `swing ^ q`, and sum. Report it as a ratio to running uncontrolled.

## Why this exists

Chips wear out partly from **thermal cycling**. The die, solder bumps, paste and copper expand by different amounts when heated, so each cycle flexes the joints between them a little. "Keep the temperature steady" needs a number to optimise, and the obvious ones mislead:
* **Standard deviation** misses that one big swing hurts far more than many small ones.
* **Peak temperature** misses repetition.

## How it works

**Step 1: find the cycles (rainflow counting).** A temperature trace is a series of peaks and valleys. Small wiggles ride on top of big swings, and both should count. Rainflow does this with a stack:

```python
for each turning point (peak or valley):
    push it
    while the stack has 3+ points and the newest swing >= the previous swing:
        the previous swing is one full cycle: record it, and remove its two points
# whatever's left on the stack counts as half cycles
```

It's essentially matching nested parentheses, so a 1 °C wiggle inside a 20 °C swing is counted as its own small cycle, and the 20 °C swing is still counted whole.

**Step 2: score each cycle.** The Coffin-Manson rule: damage per cycle ∝ swing ^ q.

| q | Typical for | What it means |
|---|---|---|
| 2 | solder (published values ~2–2.5) | 2× the swing = 4× the damage |
| 4 | harder metals | 2× the swing = 16× the damage |
| 6 | brittle materials | 2× the swing = 64× the damage |

Nobody knows q for this chip, so results are shown for 2, 4 and 6. A higher q cares more about the biggest swings.

**Step 3: compare.** Report `fatigue with control ÷ fatigue without`, on the same work. ×0.5 means half the wear.

**One variant:** "q = 2, ignoring swings under 2 °C". Tiny strains may cause no fatigue at all, and many tiny cycles otherwise dominate a q = 2 score.

## Limits

* **Ratios, not lifetimes.** There's no claim like "adds 2 years".
* **The junction sensor is one point,** not necessarily where a joint fails.
* **The benefit of running cooler on average** (a separate wear mechanism) isn't counted.

## Code

`gpu/thermal_metrics.py`: `rainflow`, `fatigue(temperature, exponent, smallest_swing)`, `time_std`.
