# Governor core

**Purpose:** decide, for each GPU call and each idle millisecond, whether to hold work back or add work, so the junction temperature stays inside a band.

**In one line:** a scheduler with a thermal model in it. It turns "stay under the top of the band X seconds from now" into an allowed average power, then sizes a gap before each call so the GPU's duty cycle hits that average.

**It's plain C with no GPU code,** so it can be tested anywhere. The caller provides the clock, the telemetry, and a description of each call.

## The band

The governor keeps the junction within `target ± band` (default ±1 °C). The target follows the junction, but with speed limits:
* it may rise at most `rise` °C per second (default 1);
* it may fall at most `fall` °C per second (default 0.05);
* optionally, it never goes above `ceiling`.

**Why a moving target:** "keep the temperature steady" has no single right temperature. Over a whole job the temperature *should* go up when work starts. What wears the chip is the *fast* swings. The speed limits decide which changes are allowed through (slow ones) and which are removed (fast ones). A job start becomes a ramp; a 50 ms burst gets flattened.

## Decision 1: a gap before each call (too hot)

```
allowed  = the average power that keeps the junction under the band's top by the horizon
           (the forecast, solved for power; the top may rise by rise x horizon meanwhile)
duty     = (allowed - idle power) / (this call's power - idle power)
gap      = this call's run time x (1 - duty) / duty,   at most max_gap (5 ms)
```

For example, a 1 ms call drawing 560 W, when only 400 W average is allowed (idle 90 W): duty = 310 / 470 = 0.66, so gap = 1 ms × 0.34 / 0.66 = 0.52 ms. The call then runs 66% of the time and averages 400 W.

Before computing it, the core advances the model through the calls already queued on the GPU. With asynchronous launches, the new call won't start until those finish.

**Why at most 5 ms:** the [control study](../control/sleeps-and-caps.md) found that pauses longer than ~20 ms let the die cool and reheat, which makes extra thermal cycles.

## Decision 2: filler while idle (too cold)

The mirror image: the *least* average power that keeps the junction above the band's bottom by the horizon. If that's more than idle power and the GPU has nothing to do, start a 1 ms piece of filler work, spaced at the duty that delivers that average.

## Decision 3: learning what each call costs

Each kind of call has its own learned run time and power. A kind is the function name, plus its sizes rounded to powers of two, plus its modes. So `dgemm` 4096³ and `dgemm` 1024³ are different kinds.
* **A new kind** is assumed to run at peak power. Its run time comes from FLOPs ÷ 20 TFLOP/s for GEMMs, or 1 ms otherwise.
* **When a call finishes,** its measured time and energy update that kind's learned values, as an EMA with weight 0.2.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `horizon` | 5 s | How far ahead to look |
| `band` | 1 °C | Allowed distance from the target |
| `rise`, `fall` | 1, 0.05 °C/s | How fast the target may move. **`rise` is the main cost knob** |
| `setpoint` | follow | A fixed target instead |
| `ceiling` | none | A hard cap on the target |
| `max_gap` | 5 ms | Longest single gap |
| `fill` | 1 ms | Filler piece length; 0 turns filler off |

## How it got this shape

Three versions failed in the simulated tests before this one. Each failure explains part of the current design:

1. **"Gap until the forecast fits" (search for a gap):** when even the maximum gap couldn't bring a 5 s forecast under the band, every call got 5 ms. On rocHPL that cost +20% runtime for no benefit. **Fixed** by sizing gaps from an allowed average power, which degrades smoothly.
2. **"Fill if idle would cool too far":** a 5 s horizon of idle always predicts cooling, so it filled almost constantly. On a 50 ms on/off load that raised the peak by 8 °C and made heavy wear (q = 6) 5× worse. **Fixed** by mirroring the gap rule: fill at the duty that just holds the bottom of the band.
3. **"Compare the forecast with today's band":** the forecast is X seconds ahead, but the target can move in that time. Comparing against today's band capped the rise rate at about band ÷ horizon, whatever `rise` said, so `rise` did nothing. **Fixed** by letting the band move by `rise × horizon` over the horizon.

I also tried a target that's a slow average of the junction. It flattened very well, but cost +50% to +220% runtime even with a 0.25 s window, because of the same band ÷ horizon effect. The speed-limited target was kept because its one knob means what it says.

## API

```c
Tg tg;
tg_init(&tg, &fit, &TG_DEFAULTS);
tg_observe(&tg, now, power_W, junction_C);                     // every ~1 ms
long id = tg_before_op(&tg, now, kind_key, flops, &gap_s);     // before each call
tg_op_finished(&tg, id, start, end, energy_J);                 // when the GPU finishes it
long fid = tg_fill(&tg, now, filler_key, &seconds);            // while idle: -1, or start filler
double worst = tg_worst_case(&tg);                             // the forecast, for reporting
```

## Code

`gpu/runtime/thermal_core.h`, `thermal_core.c` (about 300 lines). Tested by `test_core.c` (see [simulated tests](simulated-tests.md)).
