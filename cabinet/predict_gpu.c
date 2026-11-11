/* Estimate the coolant temperature at a GPU from the two things you can measure: its power and its junction temperature.
 *
 *   double predict_gpu_coolant(int gpu, double power_W, double junction_C, double timestamp_s);
 *
 *     gpu          which GPU package, 0 to N_GPU - 1. Each has its own fitted numbers and its own state.
 *     power_W      this package's power [W], for example the change in the rocm_smi energy counter / the change in time
 *     junction_C   this package's measured junction temperature [°C]; pass NAN when there's no new reading
 *     timestamp_s  time of this sample [s]; any clock works, and samples don't need to be evenly spaced
 *     returns      estimated coolant temperature at this GPU [°C] (NAN until the first junction reading)
 *
 * Use from your own code:  cc yourcode.c predict_gpu.c -lm
 * Build the program:       cc -DPREDICT_GPU_MAIN predict_gpu.c -lm -o predict_gpu
 *                          ./predict_gpu 0 samples.txt > gpu0.csv         (see main() at the bottom for the format)
 *
 * HOW IT WORKS. Heat from the chip passes through three layers before it reaches the coolant:
 *
 *     power --> [ die ] --R1--> [ package ] --R2--> [ cold plate ] --R3--> coolant
 *                 C1               C2                  C3
 *
 * Each layer warms by (heat in - heat out) / its heat capacity, and heat flows between two layers at (temperature difference) / R.
 * Running that with the measured power tells us how far the GPU's own power has lifted its junction above the coolant. So:
 *
 *     coolant = measured junction - lift from own power
 *
 * Every power sample advances the layers by exactly the time since the previous sample. Power is never averaged.
 *
 * WHAT IS SMOOTHED, AND WHAT THAT COSTS. The junction sensor reads in 1 °C steps, so that difference jumps around from reading to
 * reading. Only the difference is smoothed, over COOLANT_SMOOTHING_S. The estimate therefore trails real coolant changes, by about
 * 45 s in the test below, and can't see anything faster.
 *
 * HOW GOOD IT IS. Simulated job using measured water temperatures, with a sensor in 1 °C steps: typically within 0.5 °C, up to 2.7 °C
 * off during a job's fast start. A power reading 10 % too high makes that 1.7 °C typical. It has not been tested against real coolant
 * measurements taken at the same time as a GPU trace, because no such data has been available yet.
 *
 * WHAT IT CAN'T TELL YOU. Only the coolant at this GPU. Cabinet outlets, valve flow and building water aren't visible from one GPU.
 * The level includes heat from any GPUs upstream in the same coolant stream, so comparing GPUs needs care.
 * The layer names are an interpretation: the fit only sees 3 time scales.
 *
 * FOR ANOTHER GPU OR MACHINE: run a power trace with junction temperature through "fit_cooling_model gpu", from a run where the
 * coolant stayed steady (the fit assumes it did). It writes a constants file: cc -DGPU_CONSTANTS='"gpu_lux.h"' predict_gpu.c ...
 */
#include <math.h>
#include <string.h>

/* ---------------------------------------- The fitted numbers ----------------------------------------
 * These live in gpu_frontier.h, written by ./reproduce_constants.sh, which regenerates them from your rocHPL trace.
 *
 * R, thermal resistance [K/W]. "How much hotter does this layer sit than the next, per watt flowing through it?"
 *   Add the 3 for a GPU, and you get its steady lift: junction - coolant = power x (R1 + R2 + R3).
 *   Example: 0.055 K/W at 560 W lifts the junction about 31 °C above the coolant.
 *   Check it by hand: (junction under load - junction at idle) / (power under load - power at idle). The script prints this.
 *
 * C, heat capacity [J/K]. "How much energy does this layer soak up to warm by 1 °C?" Bigger means slower to change temperature.
 *   R x C is how fast that layer reacts: about 30 ms for the die, 1-2 s for the package, 15 s for the plate. That's why a millisecond
 *   power spike hardly moves the junction, while a minute-long change moves it fully.
 *
 * How the numbers are found: start from a guess, run the layers over the logged power, compare with the logged junction sensor, adjust,
 * and repeat until the error stops falling. They aren't measurements of the hardware, and the split between layers is less certain
 * than the total. The run needs big power changes, both fast and slow: a job starting, plus stalls during it. */
#ifndef GPU_CONSTANTS
#define GPU_CONSTANTS "gpu_frontier.h"
#endif
#include GPU_CONSTANTS

#ifndef WATER_CAPACITY
/* Coolant stream past this GPU [W/K]: how much power it absorbs per 1 °C it warms. Used only for water_out_C. Not fitted: it is an
 * estimate for Frontier, one node's share of the cabinet's stream (28 500 W/K / 128). Change it for other hardware. */
#define WATER_CAPACITY 223.0
#endif
#ifndef DIE_AREA_CM2
/* Die area used for heat flux [cm^2]. MI250X: 2 dies of 724 mm^2 each. */
#define DIE_AREA_CM2 14.48
#endif
#ifndef COOLANT_SMOOTHING_S
/* How long the difference is smoothed over [s]. Longer is steadier but later. 30 s is a choice, not a fit. */
#define COOLANT_SMOOTHING_S 30.0
#endif

typedef struct {
    double coolant_C;       /* estimated coolant reaching this GPU's plate */
    double junction_model_C;/* estimated coolant + modelled lift; compare with the sensor to check the model */
    double water_out_C;     /* estimated coolant leaving this GPU's plate: coolant + heat into water / WATER_CAPACITY */
    double flux_generated;  /* heat made by the chip, power / die area [W/cm^2] */
    double flux_die_exit;   /* heat leaving the die into the package [W/cm^2] */
    double flux_to_water;   /* heat entering the coolant [W/cm^2 of die area] */
} PredictGpuOutputs;

/* ---------------------------------------- What the model remembers ---------------------------------------- */
static struct {
    int started, have_coolant;
    double t;              /* time of the latest sample */
    double power;          /* the latest power, which holds until the next sample */
    double z[3];           /* each layer's lift above the coolant, in the form where each evolves on its own (see below) */
    double lam[3];         /* how fast each of those decays [1/s] */
    double Q[3][3];        /* converts between that form and layer lifts */
    double coolant, coolant_t;  /* the smoothed estimate, and when it was last updated */
    PredictGpuOutputs out;
} G[N_GPU];

/* The three layers affect each other, which makes an exact update awkward. Mathematically they can be rewritten as three independent
 * exponential decays (the matrix's eigenvectors). Each decay then updates exactly for any gap, with one exp(). This finds that form,
 * once per GPU, using Jacobi rotations on a symmetric 3 x 3 matrix. */
static void find_modes(int g)
{
    const double *r = R[g], *c = C[g];
    double a[3][3] = {{-1 / (r[0] * c[0]), 1 / (r[0] * sqrt(c[0] * c[1])), 0},
                      {1 / (r[0] * sqrt(c[0] * c[1])), -1 / (r[0] * c[1]) - 1 / (r[1] * c[1]), 1 / (r[1] * sqrt(c[1] * c[2]))},
                      {0, 1 / (r[1] * sqrt(c[1] * c[2])), -1 / (r[1] * c[2]) - 1 / (r[2] * c[2])}};
    double q[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int sweep = 0; sweep < 60; sweep++)
        for (int p = 0; p < 2; p++)
            for (int k = p + 1; k < 3; k++) {
                if (fabs(a[p][k]) < 1e-14) continue;
                double th = 0.5 * atan2(2 * a[p][k], a[k][k] - a[p][p]), s = sin(th), co = cos(th);
                for (int i = 0; i < 3; i++) { double x = a[i][p], y = a[i][k]; a[i][p] = co * x - s * y; a[i][k] = s * x + co * y; }
                for (int i = 0; i < 3; i++) { double x = a[p][i], y = a[k][i]; a[p][i] = co * x - s * y; a[k][i] = s * x + co * y; }
                for (int i = 0; i < 3; i++) { double x = q[i][p], y = q[i][k]; q[i][p] = co * x - s * y; q[i][k] = s * x + co * y; }
            }
    for (int i = 0; i < 3; i++) { G[g].lam[i] = a[i][i]; for (int j = 0; j < 3; j++) G[g].Q[i][j] = q[i][j]; }
}

/* Each layer's current lift above the coolant [°C]: lift[0] is the die's */
static void layer_lifts(int g, double lift[3])
{
    for (int i = 0; i < 3; i++) {
        lift[i] = 0;
        for (int m = 0; m < 3; m++) lift[i] += G[g].Q[i][m] * G[g].z[m];
        lift[i] /= sqrt(C[g][i]);
    }
}

double predict_gpu_coolant(int gpu, double power_W, double junction_C, double timestamp_s)
{
    if (gpu < 0 || gpu >= N_GPU) return NAN;
    if (!G[gpu].started) {
        if (!isfinite(power_W) || !isfinite(timestamp_s)) return NAN;
        find_modes(gpu);
        /* Start settled at this power: the same heat flows through every layer, so each sits above the next by power x R */
        const double *r = R[gpu];
        double lift[3] = {power_W * (r[0] + r[1] + r[2]), power_W * (r[1] + r[2]), power_W * r[2]};
        for (int m = 0; m < 3; m++) {
            G[gpu].z[m] = 0;
            for (int i = 0; i < 3; i++) G[gpu].z[m] += G[gpu].Q[i][m] * lift[i] * sqrt(C[gpu][i]);
        }
        G[gpu].power = power_W; G[gpu].t = timestamp_s; G[gpu].started = 1;
    } else if (isfinite(timestamp_s) && timestamp_s > G[gpu].t) {
        /* The previous power held over this gap. It enters at the die; each independent decay relaxes exactly toward where that power
         * would settle it. */
        double dt = timestamp_s - G[gpu].t, b0 = G[gpu].power / sqrt(C[gpu][0]);
        for (int m = 0; m < 3; m++) {
            double target = -G[gpu].Q[0][m] * b0 / G[gpu].lam[m];
            G[gpu].z[m] = target + (G[gpu].z[m] - target) * exp(G[gpu].lam[m] * dt);
        }
        G[gpu].t = timestamp_s;
    }
    /* A repeated or out-of-order timestamp doesn't move time, but its power still applies from here on. Bad inputs are skipped. */
    if (isfinite(power_W)) G[gpu].power = power_W;

    double lift[3]; layer_lifts(gpu, lift);
    if (isfinite(junction_C)) {
        double raw = junction_C - lift[0];                  /* coolant = measured junction - lift from own power */
        if (!G[gpu].have_coolant) { G[gpu].coolant = raw; G[gpu].coolant_t = G[gpu].t; G[gpu].have_coolant = 1; }
        else if (G[gpu].t > G[gpu].coolant_t) {
            G[gpu].coolant = raw + (G[gpu].coolant - raw) * exp(-(G[gpu].t - G[gpu].coolant_t) / COOLANT_SMOOTHING_S);
            G[gpu].coolant_t = G[gpu].t;
        }
    }
    double coolant = G[gpu].have_coolant ? G[gpu].coolant : NAN, to_water = lift[2] / R[gpu][2];
    PredictGpuOutputs *o = &G[gpu].out;
    o->coolant_C = coolant;
    o->junction_model_C = coolant + lift[0];
    o->water_out_C = coolant + to_water / WATER_CAPACITY;
    o->flux_generated = G[gpu].power / DIE_AREA_CM2;
    o->flux_die_exit = (lift[0] - lift[1]) / R[gpu][0] / DIE_AREA_CM2;
    o->flux_to_water = to_water / DIE_AREA_CM2;
    return coolant;
}

/* The full set of outputs from the latest call for this GPU */
const PredictGpuOutputs *predict_gpu_outputs(int gpu) { return &G[gpu].out; }

/* Forget everything; the next call for each GPU starts over */
void predict_gpu_reset(void) { memset(G, 0, sizeof G); }

#ifdef PREDICT_GPU_MAIN
#include <stdio.h>
#include <stdlib.h>

/* Command-line program: ./predict_gpu GPU [samples.txt]. Reads one sample per line from a file or stdin:
 *     power_W  timestamp_s  junction_C      (use nan, or leave it off, when there's no reading)
 * separated by spaces or commas. Other lines (headers, comments) are skipped. Prints one CSV row per sample. */
int main(int argc, char **argv)
{
    FILE *in = stdin;
    if (argc < 2 || argc > 3 || atoi(argv[1]) < 0 || atoi(argv[1]) >= N_GPU || (argc == 3 && !(in = fopen(argv[2], "r")))) {
        fprintf(stderr, "usage: %s GPU [samples.txt]\n  GPU: 0 to %d\n  each line: power_W timestamp_s junction_C\n", argv[0], N_GPU - 1);
        return 1;
    }
    int gpu = atoi(argv[1]);
    char line[512];
    printf("timestamp_s,power_W,junction_C,coolant_C,junction_model_C,water_out_C,flux_generated_W_cm2,flux_die_exit_W_cm2,flux_to_water_W_cm2\n");
    while (fgets(line, sizeof line, in)) {
        double p, t, j = NAN;
        for (char *c = line; *c; c++) if (*c == ',') *c = ' ';
        if (sscanf(line, "%lf %lf %lf", &p, &t, &j) < 2) continue;
        predict_gpu_coolant(gpu, p, j, t);
        const PredictGpuOutputs *o = predict_gpu_outputs(gpu);
        printf("%.6f,%.3f,%.3f,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f\n", t, p, j, o->coolant_C, o->junction_model_C, o->water_out_C, o->flux_generated, o->flux_die_exit, o->flux_to_water);
    }
    if (in != stdin) fclose(in);
    return 0;
}
#endif
