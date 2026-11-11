/* Streaming version of predict_cooling from frontier_workload_stress.ipynb.
 *
 * Call predict_cooling() each time you have a new power sample. It remembers everything between calls.
 * Use from your own code:  cc yourcode.c predict_cooling.c -lm
 * Build the program:       cc -DPREDICT_COOLING_MAIN predict_cooling.c -lm -o predict_cooling
 *                          ./predict_cooling samples.txt > cooling.csv    (see main() at the bottom for the format)
 *
 *   double predict_cooling(double current_power, double current_timestamp, double facility_water_C);
 *
 *     current_power      AC power of ONE cabinet [kW], assumed the same on every cabinet in the cooling group
 *     current_timestamp  time of this sample [s]; any clock works, and samples don't need to be evenly spaced
 *     facility_water_C   temperature of the building water reaching the CDU [°C]; about 21 in the fitted run
 *     returns            temperature of the water going to the cabinets [°C], at this sample's time
 *
 * The first call starts the model settled at that power. After that, every sample advances the model by exactly the time since
 * the previous sample, with the previous sample's power holding over that gap. Nothing is averaged or dropped, so 1 ms data
 * gives 1 ms output. Gaps longer than 1 s are run in steps of up to 1 s, with the same power.
 * current_power must be CABINET power, not GPU power; frontier_rochpl_cooling.ipynb shows the conversion for rocHPL data.
 * Other outputs: predict_cooling_outputs(). To start over: predict_cooling_reset().
 * It keeps one shared state, so it models one cooling group and isn't thread-safe.
 *
 * Fitted to one 47-minute run of one Frontier cooling group (Feb 2026), about 90-350 kW per cabinet. Expect about +/- 0.5 °C
 * in that range. The valve limits are just the highest and lowest flow seen in that run. That data was 1 Hz, so the model runs
 * at any sample rate, but its behaviour faster than about 1 s has never been checked against measurements.
 *
 * FOR ANOTHER MACHINE: don't edit the numbers by hand. Run fit_cooling_model.py on that machine's telemetry. It writes a
 * constants file, which you pass in when compiling: cc -DCOOLING_CONSTANTS='"cooling_lux.h"' predict_cooling.c ...
 *
 * HOW THE NUMBERS ARE OBTAINED. Log these at about 1 Hz for one CDU group, for at least 30 minutes, across a big job start and stop:
 *     P[i]  electrical power of each cabinet     In[i], Out[i]  water temperature entering and leaving each cabinet
 *     Fin, Fout  building-water temperatures     F  building-water flow
 *
 * Two kinds of number appear below, and each is labelled:
 *   [MEASURE]  computed directly from the log, with simple statistics.
 *   [SEARCH]   can't be read off the log. The method: guess the number, run the model over the logged inputs, score it by RMS error
 *              against the logged output, and let an optimizer (scipy least_squares) adjust the guess to lower the score.
 *              Run in 3 stages, so an error in one part can't hide in another:
 *                Stage 1: input P[i], In[i]              -> match Out[i]    gives Ch, UA, mcp, inlet_delay, offset (per cabinet)
 *                Stage 2: input stage 1's return, Fin, F -> match In, Fout  gives UA_hx, hx_delay, tau_s, tau_p, kappa, b_p
 *                Stage 3: input In                       -> match F         gives T_set, Kp, Ki, sensor_delay
 *              Only the first part of the log is used to search. The rest is held out, to test on data it never saw.
 * "Usable range" means: fix the number at that value and re-search the others, and the Frontier fit stays about as good.
 * A wide range means this one run doesn't pin the number down, so don't read much into its exact value.
 * Reading some of these straight off a plot does NOT work here. For example, timing 63 % of the outlet's rise after a job start gives
 * about 135 s, not the true 66-78 s, because the inlet temperature is moving at the same time.
 *
 * HOW THE HEAT MOVES (one loop, repeated at every sample)
 *
 *   electricity --> cabinet hardware --> cabinet water ("secondary loop") --> CDU heat exchanger --> building water
 *                     (stores heat)              ^                                   |               ("primary loop")
 *                                                |          cooled water             |
 *                                                +-----------------------------------+
 *
 *   1. A cabinet's electrical power becomes heat in its hardware. The hardware stores some of it and warms up.
 *   2. Water passing through picks up heat from the hardware and leaves warmer (the "outlet" temperature).
 *   3. All the outlets mix and flow to the CDU's heat exchanger, which hands the heat to the building water.
 *   4. A valve sets how much building water flows. It watches the water going back to the cabinets (the "supply") and tries
 *      to hold it at a target temperature. More building water removes more heat.
 *
 * WHY THE NUMBERS COME IN THREES: on Frontier one CDU serves 3 cabinets, and each cabinet was fitted separately.
 */
#include <math.h>
#include <string.h>

#define SETTLE_S 6000       /* constant inputs for this long leave the model settled */
#define SLOTS_PER_S 1000    /* resolution of the delay history: 1 ms */

/* ---------------------------------------- The fitted numbers ----------------------------------------
 * These were chosen so the model matches measured telemetry. "W/K" means watts per degree of temperature difference.
 * Every comment's example uses the Frontier values below. */
#ifdef COOLING_CONSTANTS
#include COOLING_CONSTANTS          /* another machine's numbers, written by fit_cooling_model.py */
#else

#define N_CAB 3                     /* cabinets served by one CDU */
#define HISTORY 69                  /* longest delay (the valve's 68 s sensor delay) + 1 */

/* HEAT CAPACITY of each cabinet's hardware [J/K]: the energy needed to warm the hardware by 1 °C. It's why a cabinet warms
 * gradually instead of instantly. Example: 1 080 000 J/K heated by 240 kW with no cooling warms about 0.22 °C per second.
 * Treat these as "effective" values, not measured masses. The differences between cabinets are probably not real hardware
 * differences: fixing all 3 at about 1.1 MJ/K and refitting the rest fits nearly as well (at most 0.03 K worse RMS on the fitted
 * part of the run). What the data does show is that cabinet 3's outlet reacted faster than cabinet 1's in this run.
 * [SEARCH, stage 1] Usable range: cabinet 1 about 1.3-2.0, cabinet 2 about 0.8-1.3, cabinet 3 about 0.5-1.1 MJ/K.
 * Needs a big power step; the speed of the outlet's response is what sets it. */
static const double Ch[N_CAB] = {1.61807e+06, 1.08261e+06, 763183};

/* WATER-SIDE CONDUCTANCE [W/K]: how easily heat moves from the hardware into the water; think of the size and quality of the
 * contact between metal and water. It is the watts that would flow for each 1 °C the hardware is hotter than the water.
 * Larger means the water gets closer to the hardware's temperature. Fitted together with Ch, so it shares the same caveat.
 * [SEARCH, stage 1] It moves together with Ch: a larger Ch needs a larger UA to keep the same response speed. */
static const double UA[N_CAB] = {35222.6, 20265.8, 15145};

/* WATER STREAM CAPACITY [W/K]: flow rate x specific heat, or "how much power the stream absorbs for each 1 °C it warms up".
 * This is the most directly meaningful of the three. It sets how much the water warms in steady running:
 *     temperature rise = power / stream capacity,   e.g. 320 000 W / 28 873 W/K = about 11 °C.
 * 28 873 W/K is about 110 gpm of water (1 gpm of water carries 263 W/K).
 * [MEASURE] A straight-line fit: find the stretches where power and temperatures are steady, take y = Out - In and x = P, and fit
 * y = x / mcp + offset. So mcp = 1 / slope. It needs at least two different power levels, such as idle and full load.
 * Frontier check: this gives 29.7, 28.5, 26.5 kW/K, against 29.8, 28.9, 27.0 from the search below. If a flow meter exists on
 * this loop, use it instead: mcp = gpm x 263 W/K. */
static const double mcp[N_CAB] = {29758, 28872.9, 26950.5};

/* How long a change in supply temperature takes to reach each cabinet [s]. [SEARCH, stage 1] Tiny here, so it hardly matters. */
static const int inlet_delay[N_CAB] = {2, 0, 0};

/* Added to each outlet temperature to match its sensor [°C]. Sensors disagree slightly, and this absorbs that.
 * [MEASURE] The intercept of the same straight-line fit as mcp. Frontier check: it gives +0.32, +0.53, +0.54. */
static const double offset[N_CAB] = {0.269659, 0.624875, 0.621333};

/* THE CDU HEAT EXCHANGER
 * All [SEARCH, stage 2]. It takes the mixed cabinet return water, Fin and F as given, and is scored on matching In and Fout.
 * UA_hx    Conductance between the two water loops [W/K]: the same idea as UA, across the exchanger's plates.
 *          Larger means the two streams get closer in temperature. Usable range: about 200 000-600 000.
 * tau_s    Smoothing on the supply temperature [s]. The fit pushed it to its 0.5 s lower limit, so effectively there's none.
 *          Usable range: up to about 10 s.
 * tau_p    Smoothing on the building-water return temperature [s]. It trails the heat by about 8 s. Usable range: 1-15 s.
 * kappa    Correction on the building water's temperature rise. In the data, flow x rise doesn't quite match the heat, as if the
 *          flow meter or temperature sensors were off. This is a calibration factor, not physics. Usable range: about 0.8-0.95.
 *          It trades off against b_p: a straight-line fit on steady stretches gave 0.71 and +3.8 °C instead, so don't trust either alone.
 * b_p      Offset added to the building-water return temperature to match its sensor [°C].
 * hx_delay How long the mixed cabinet return water takes to affect the exchanger's output [s]. Usable range: 0-20 s. */
static const double UA_hx = 335996, tau_s = 0.5, tau_p = 7.859, kappa = 0.884233, b_p = 0.940919;
static const int hx_delay = 13;

/* THE VALVE CONTROLLER: a thermostat for the supply water, the standard "PI" kind.
 * T_set, Kp, Ki and sensor_delay are [SEARCH, stage 3]: run this controller on the logged In, and match the logged F.
 * T_set        Target supply temperature [°C]. Sanity check: it should sit near the median of In on steady stretches. It isn't a
 *              published setpoint; the real one would be better if the operators can supply it.
 * Kp           "Proportional" gain [gpm per °C]: when the error grows by 1 °C, immediately add this much flow.
 *              Loosely pinned: anything from 0 to about 5.5 fits fairly similarly.
 * Ki           "Integral" gain [gpm per °C per second]: keep adding this much flow each second while any error remains.
 *              This is what eventually removes the error completely. Tightly pinned: 0.06 or 0.25 fit far worse.
 * sensor_delay The valve acts on the supply temperature from this long ago [s]. Usable range: about 50-80 s. It lumps together any
 *              sensing, transport and decision delays, which this data can't tell apart.
 * flow_min/max Limits on the building-water flow [gpm]. [MEASURE] min(F) and max(F) in the log, so they're only what this run
 *              reached. The real valve limits from the CDU's specification would be better.
 * A direct regression of flow changes on temperature errors fails on this data (it gives a negative Kp), because the flow signal is
 * too coarse and noisy from second to second. Simulating the whole history is what makes it work. */
static const int sensor_delay = 68;
static const double T_set = 28.6372, Kp = 5.52879, Ki = 0.12266, flow_min = 95.8, flow_max = 208.6;

/* Fraction of the electrical power that ends up in the water. 1 for Frontier's fanless cabinets; lower if fans carry heat away.
 * [MEASURE] An estimate: average(F x 263 x (Fout - Fin)) / average(total P), over a long run. Frontier gives 0.96. */
static const double heat_to_water = 1;

/* W/K carried by 1 gpm of water: converts building-water flow into a stream capacity. Physics, not fitted:
 * 6.309e-5 m^3/s per gpm x 998 kg/m^3 x 4180 J/(kg K). Change it if the loop isn't plain water. */
static const double GPM = 263.2;
#endif

typedef struct {
    double supply_C;        /* water going to the cabinets */
    double outlet_C[N_CAB]; /* water leaving each cabinet */
    double facility_gpm;    /* building-water flow chosen by the CDU's valve */
    double facility_out_C;  /* building water leaving the CDU */
} PredictCoolingOutputs;

/* ---------------------------------------- What the model remembers ---------------------------------------- */
#define RING ((HISTORY + 2) * SLOTS_PER_S)   /* enough history for the longest delay, at 1 ms per slot */

static struct {
    int started;
    double hw[N_CAB];             /* temperature of each cabinet's hardware [°C]: the heat it has stored so far */
    double supply[RING];          /* recent supply temperatures, for the delays [°C] (a ring buffer, one slot per ms) */
    double ret[RING];             /* recent mixed return temperatures from the cabinets [°C] (a ring buffer, one slot per ms) */
    long long slot;               /* the newest slot; slot % RING is its position in the ring */
    double elapsed;               /* model time since the start [s] */
    double flow;                  /* building-water flow right now [gpm] */
    double fac_out;               /* building-water return temperature, before its sensor offset [°C] */
    double s_now;                 /* supply temperature leaving the exchanger [°C] */
    double e_prev;                /* the controller's previous temperature error [°C] */
    double power_W, water_C;      /* the latest sample's inputs, which hold until the next sample */
    double t;                     /* time of the latest sample */
    PredictCoolingOutputs out;
} S;

/* A ring buffer's value at `slot`, or from `age` seconds before it */
static double back(const double *buf, long long slot, int age) { return buf[(slot - (long long)age * SLOTS_PER_S) % RING]; }

/* Store a new value at new_slot. Any slots skipped since from_slot keep the old value: a temperature holds until it next changes. */
static void push(double *buf, long long from_slot, long long new_slot, double value)
{
    for (long long k = from_slot + 1; k < new_slot; k++) buf[k % RING] = buf[from_slot % RING];
    buf[new_slot % RING] = value;
}

/* CHANNEL EFFECTIVENESS, from 0 to 1: how far the water warms toward the hardware's temperature on its way through.
 * 0 means it picks up nothing; 1 means it leaves at the hardware's temperature. It combines the two numbers above: better contact
 * (UA) raises it, and faster flow (mcp) lowers it because the water has less time to warm. Frontier: 0.69, 0.50, 0.43. */
static double epsilon(int i) { return 1 - exp(-UA[i] / mcp[i]); }

/* Copy the current state into the output struct */
static void record(void)
{
    S.out.supply_C = S.supply[S.slot % RING];
    for (int i = 0; i < N_CAB; i++) {
        double t_in = back(S.supply, S.slot, inlet_delay[i]);
        S.out.outlet_C[i] = t_in + epsilon(i) * (S.hw[i] - t_in) + offset[i];
    }
    S.out.facility_gpm = S.flow;
    S.out.facility_out_C = S.fac_out + b_p;
}

static double total_capacity(void)
{
    double c = 0;
    for (int i = 0; i < N_CAB; i++) c += mcp[i];
    return c;
}

/* Advance the model by dt seconds, with this power per cabinet [W] and building-water temperature holding the whole time */
static void step(double power_W, double water_C, double dt)
{
    double Cs = total_capacity(), ret_new = 0;              /* Cs: stream capacity of all the cabinets' water together */
    S.elapsed += dt;
    long long new_slot = RING + llround(S.elapsed * SLOTS_PER_S);

    /* --- Each cabinet: heat goes into the hardware, and the water carries some of it away --- */
    for (int i = 0; i < N_CAB; i++) {
        double eps = epsilon(i), t_in = back(S.supply, S.slot, inlet_delay[i]);
        /* The water warms part of the way from its inlet temperature to the hardware's temperature */
        double t_out = t_in + eps * (S.hw[i] - t_in);
        /* The hardware temperature where heat out would equal heat in. It closes the gap to this exponentially, at a speed set
         * by the time constant Ch / (eps * mcp), about 66-78 s on Frontier. This update is exact for any dt. */
        double target = t_in + power_W / (eps * mcp[i]);
        S.hw[i] = target + (S.hw[i] - target) * exp(-dt * eps * mcp[i] / Ch[i]);
        ret_new += mcp[i] * t_out;
    }
    push(S.ret, S.slot, new_slot, ret_new / Cs);            /* mix the outlets, weighted by how much each stream carries */
    double tr = back(S.ret, new_slot, hx_delay);
    /* --- The heat exchanger: how much heat is crossing to the building water right now ---
     * Cp is the building water's stream capacity. The stream with the smaller capacity (Cmin) limits how much heat can move.
     * This is the textbook counterflow formula (the "effectiveness-NTU" method): (1 - x) / (1 - ratio * x) is the exchanger's
     * effectiveness, the fraction of the largest possible heat transfer it achieves. Q is the heat moved [W]. */
    double Cp = S.flow * GPM, Cmin = fmin(Cs, Cp), Cmax = fmax(Cs, Cp);
    double ntu = UA_hx / Cmin, x = exp(-ntu * (1 - Cmin / Cmax));
    double Q = (1 - x) / (1 - Cmin / Cmax * x) * Cmin * (tr - water_C);
    /* Removing Q cools the cabinet water by Q / Cs, and adding Q warms the building water by Q / Cp */
    double s_target = tr - Q / Cs, p_target = water_C + kappa * Q / Cp;
    S.s_now = s_target + (S.s_now - s_target) * exp(-dt / tau_s);
    S.fac_out = p_target + (S.fac_out - p_target) * exp(-dt / tau_p);
    push(S.supply, S.slot, new_slot, S.s_now);
    S.slot = new_slot;
    /* --- The valve: too warm means open up, too cool means close down, using a reading from sensor_delay seconds ago --- */
    double e = back(S.supply, S.slot, sensor_delay) - T_set;
    S.flow = fmin(fmax(S.flow + Kp * (e - S.e_prev) + Ki * e * dt, flow_min), flow_max);
    S.e_prev = e;
}

/* First sample: make a rough starting guess, then run at constant inputs until settled, so the guess no longer matters */
static void start(double power_W, double water_C, double t)
{
    double Cs = total_capacity(), total = N_CAB * power_W;
    memset(&S, 0, sizeof S);
    S.slot = RING;
    for (int i = 0; i < N_CAB; i++) S.hw[i] = T_set + power_W / (epsilon(i) * mcp[i]);
    for (int i = 0; i < RING; i++) { S.supply[i] = T_set; S.ret[i] = T_set + total / Cs; }
    S.flow = fmin(fmax(total / (GPM * fmax(T_set - water_C, 1.0)), flow_min), flow_max);
    S.fac_out = water_C + kappa * total / (GPM * 150);
    S.s_now = T_set;
    for (int k = 0; k < SETTLE_S; k++) step(power_W, water_C, 1.0);
    S.power_W = power_W; S.water_C = water_C; S.t = t; S.started = 1;
}

double predict_cooling(double current_power, double current_timestamp, double facility_water_C)
{
    double power_W = current_power * 1e3 * heat_to_water;
    if (!S.started) {
        if (isfinite(current_power) && isfinite(current_timestamp) && isfinite(facility_water_C)) {
            start(power_W, facility_water_C, current_timestamp);
            record();
        }
        return S.started ? S.out.supply_C : NAN;
    }
    if (isfinite(current_timestamp) && current_timestamp > S.t) {
        /* The previous sample's power held from then until now. After a very long gap the model is settled anyway, so don't run
         * more than that takes. A gap over 1 s runs in steps of up to 1 s, which keeps the valve controller well behaved. */
        double gap = fmin(current_timestamp - S.t, SETTLE_S);
        while (gap > 1.0) { step(S.power_W, S.water_C, 1.0); gap -= 1.0; }
        step(S.power_W, S.water_C, gap);
        S.t = current_timestamp;
        record();
    }
    /* A repeated or out-of-order timestamp doesn't move time, but its inputs still apply from here on. Bad inputs are skipped. */
    if (isfinite(current_power)) S.power_W = power_W;
    if (isfinite(facility_water_C)) S.water_C = facility_water_C;
    return S.out.supply_C;
}

#ifndef COOLING_CONSTANTS
/* Convert GPU power to the cabinet power predict_cooling() needs. These are Frontier's fits; another machine needs its own.
 * gpu_sum_W: total power of one node's 4 GPU packages [W]. Returns one cabinet's AC power [kW], assuming all 128 nodes match this
 * node. Above about 2190 W per node (281 kW of node power per cabinet) this is outside the measured range and uses a straight line.
 *   Step 1, GPUs to node: node W = 1.14882 x GPUs + 146.239, a straight-line fit of the node's power sensor against the GPU sum.
 *   Step 2, nodes to AC: a quadratic fit of cabinet AC against the Redfish node-power sum, on steady stretches of the run. */
double cabinet_power_from_gpus(double gpu_sum_W)
{
    static const double lo = 59249.2, hi = 280716;
    double node_sum = 128 * (1.14882 * gpu_sum_W + 146.239);
    double x = fmin(fmax(node_sum, lo), hi);
    return (-2.59758e-06 * x * x + 1.98739 * x - 21125.4 + (node_sum - x) / (0.98 * 0.96)) / 1e3;
}
#endif

/* The full set of outputs from the latest call */
const PredictCoolingOutputs *predict_cooling_outputs(void) { return &S.out; }

/* Forget everything; the next call starts settled again */
void predict_cooling_reset(void) { S.started = 0; }

#ifdef PREDICT_COOLING_MAIN
#include <stdio.h>

/* Command-line program. Reads one sample per line from a file or stdin:
 *     power_kW  timestamp_s  [facility_water_C]
 * separated by spaces or commas. Other lines (headers, comments) are skipped. Prints one CSV row per sample. */
int main(int argc, char **argv)
{
    FILE *in = stdin;
    if (argc > 2 || (argc == 2 && !(in = fopen(argv[1], "r")))) {
        fprintf(stderr, "usage: %s [samples.txt]\n  each line: power_kW timestamp_s [facility_water_C]\n", argv[0]);
        return 1;
    }
    char line[512];
    double water = 21.2;
    printf("timestamp_s,power_kW,supply_C");
    for (int i = 0; i < N_CAB; i++) printf(",outlet_%d_C", i + 1);
    printf(",facility_gpm,facility_out_C\n");
    while (fgets(line, sizeof line, in)) {
        double power, t, w;
        for (char *c = line; *c; c++) if (*c == ',') *c = ' ';
        int n = sscanf(line, "%lf %lf %lf", &power, &t, &w);
        if (n < 2) continue;
        if (n == 3) water = w;
        double supply = predict_cooling(power, t, water);
        const PredictCoolingOutputs *o = predict_cooling_outputs();
        printf("%.6f,%.4f,%.6f", t, power, supply);
        for (int i = 0; i < N_CAB; i++) printf(",%.6f", o->outlet_C[i]);
        printf(",%.5f,%.6f\n", o->facility_gpm, o->facility_out_C);
    }
    if (in != stdin) fclose(in);
    return 0;
}
#endif
