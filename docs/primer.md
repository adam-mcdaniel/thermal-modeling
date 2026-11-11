# Primer: the physics, in CS terms

**What this page is for:** the vocabulary and the handful of ideas the other pages assume, explained through things a computer scientist already knows.

**What the project is modelling, in one sentence:** given the watts a GPU is drawing over time (which you can log), predict the temperature of the hottest point on its chip (which wears the chip out when it swings), so a runtime can pace the work to keep that temperature steady.

## Units, concretely

| Unit | What it means | Example |
|---|---|---|
| W (watts) | Power: energy per second going into the chip. The input | An MI250X package draws ~90 W idle and ~560 W running DGEMM |
| J (joules) | Energy. What the energy counter counts | 560 W for 1 s = 560 J |
| °C | Temperature. The output | Junction ~40 °C idle, ~65 °C under load |
| °C/W (= K/W) | Thermal resistance: degrees of rise per watt, once settled | 0.055 °C/W means 500 W settles 27.5 °C above the water |
| J/°C (= J/K) | Heat capacity: joules needed to warm it 1 °C | 1,000 J/°C at 500 W warms at most 0.5 °C per second |
| s (seconds) | Time constants: how long until the temperature mostly catches up after power changes | Die ~0.03 s, package ~1.5 s, cold plate ~16 s |
| W/cm² | Heat flux: power per area of chip | 560 W over 14.5 cm² of die = 39 W/cm² |

K (kelvin) and °C differences are the same size, so °C/W and K/W are interchangeable.

Everything in this project rests on about five ideas, and each has a direct counterpart in computing.

## 1. A layer of the chip is an exponential moving average

Heat behaves like a queue with a leak:

* **Power** (watts) is the arrival rate: joules per second going in.
* **Temperature** is the backlog: how much has built up.
* **Heat capacity, C** (joules per °C), is the buffer size: how much heat it takes to raise the temperature 1 °C. A big C fills slowly.
* **Thermal resistance, R** (°C per watt), is how badly heat drains out to the next layer. Heat leaves at `temperature difference / R`, so a high R means a high backlog before it drains as fast as it fills.

Put together, one layer updates exactly like an EMA:

```c
// every dt seconds, with power P going in and the layer sitting `lift` degrees above what it drains into
lift += dt / (R * C) * (P * R - lift);    // the same as: ema += alpha * (sample - ema), with alpha = dt / tau
```

So a layer's lift is a **moving average of `P × R`, with window `tau = R × C`**. This has three consequences:
* Hold the power steady, and the lift settles at `P × R`, the steady state.
* Change the power, and the lift gets about 63% of the way to the new steady state in one `tau`, and is essentially there after 5 `tau`.
* Swings much faster than `tau` are averaged out. Swings much slower pass straight through.

## 2. The GPU is three moving averages in a chain

Heat goes from the die (where it's made), through the package, into the copper cold plate, and then into the water:

```
power --> [ die ] --R1--> [ package ] --R2--> [ cold plate ] --R3--> water
            C1               C2                  C3
```

On an MI250X the fitted windows are roughly:

| Layer | Window (tau) | Lift at 500 W | Like |
|---|---|---|---|
| Die | ~30 ms | ~8 °C | L1: reacts instantly, forgets fast |
| Package | ~1.5 s | ~12 °C | L2 |
| Cold plate | ~16 s | ~8 °C | memory: slow, remembers the last minute |

**junction = water temperature + the three lifts.** The sensor reads the junction. Calibration fits R and C for each layer (six numbers) plus the water temperature.

## 3. It's linear, so the future is a dot product

Each update is linear in the state and the power. So the junction X seconds from now is:

```
junction_later = water + w_die * die_lift + w_pkg * package_lift + w_plate * plate_lift + w_power * future_power
```

The weights depend only on X and the fitted numbers, so they're computed once. A forecast is then four multiplications. More power can only mean more heat, so the **worst case** is simply `future_power = peak power`.

## 4. The water is a shared, slow resource

Many GPUs share the same water loop. When a big job starts everywhere at once, the water warms by a few degrees over about a minute, until the cooling plant (the CDU) catches up and brings it back. It's a shared resource with contention:
* **A single GPU can't see its neighbours' load** from its own power.
* **When the whole group runs one job,** each GPU's own power is a good proxy for everyone's.

## 5. Wear comes from cycles, and big ones count far more

Chips fail partly from **thermal cycling**. Materials expand when hot, by different amounts, so every heat-up and cool-down flexes the joints between layers, like bending a paperclip. The standard model (Coffin-Manson) says each cycle's damage grows as `swing ^ q`, with q around 2 for solder and 4–6 for more brittle materials. So one 20 °C swing does as much harm as 16–64 swings of 10 °C, depending on q.

To find the cycles in a temperature trace, use **rainflow counting**. It's a stack algorithm: push each turning point, and whenever the newest swing is at least as big as the previous one, pop a matched pair as one cycle. It's essentially matching nested parentheses, so small wiggles inside a big swing are counted on their own and the big swing is still counted whole. See `gpu/thermal_metrics.py`.

**Results are reported as ratios** (×0.5 = half the fatigue of running uncontrolled), because q isn't known for this chip. They're not lifetimes in years.

## 6. The measurements

| Signal | Source | Rate | Catch |
|---|---|---|---|
| Power | rocm_smi energy counter: power = Δenergy / Δtime | ~1–3 ms | Single samples are noisy (spikes to 3,700 W are timing artifacts). Fine on average |
| Junction | rocm_smi temperature, sensor 1 | ~6.5 ms | Whole °C only, and flickers ±1 °C |
| Coolant | Not measured per GPU | | Estimated as junction − die lift |

On an MI250X, the two GCDs in a package share one energy counter, which only the even-numbered device reports.
