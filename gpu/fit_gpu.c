/*
 * Calibrate: find each GPU's numbers from a power trace and its junction sensor. Nothing else is measured.
 *
 *     cc -O2 fit_gpu.c -lm -o fit_gpu
 *     ./fit_gpu gpu_fit.h gpu0_samples.txt gpu1_samples.txt ...
 *
 * Each line of a samples file: power_W timestamp_s junction_C. Use nan for the junction on lines with no reading.
 * Use a run with big power changes, both fast and slow, that holds high power for a couple of minutes.
 *
 * How the numbers are found:
 *     1. Start from a guess, sized from how far the junction moved per watt in this run.
 *     2. Run the model (the same code as gpu_model.h) over the logged power.
 *     3. Compare with the logged junction.
 *     4. Nudge each number to see how the error responds, then step to reduce it (Levenberg-Marquardt).
 *     5. Repeat until the error stops falling.
 *
 * Fit on the first 70% of the run, and score on the rest. That score is printed, so every calibration reports how well it predicts data it didn't see.
 * It also decides the water: if the run is long enough to show the coolant moving, try the 3 water numbers, and keep them only if they predict the held-back part better.
 * The final numbers then come from the whole run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_model.h"

#define LAYER_NUMBERS 7          /* R x 3, C x 3, coolant */
#define ALL_NUMBERS 10           /* and the 3 water numbers */
#define PEAK_WINDOW_S 0.2        /* shorter spikes come mostly from counter timing */
#define FIT_SHARE 0.7            /* the fraction of the run to fit on when scoring */
#define WATER_RUN_S 300.0        /* a shorter run can't show the water's response, which takes minutes */
#define WATER_MUST_IMPROVE 0.9   /* keep the water numbers only if the held-back error falls below this fraction */
#define SMALLEST_SWING_C 15.0    /* the sensor reads in 1 °C steps, so a smaller swing can't pin down the layers */

typedef struct {
    double *power, *time, *junction;
    int rows;
    int readings;       /* rows that have a junction reading */
} Trace;

static Trace read_trace(const char *path)
{
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "can't open %s\n", path);
        exit(1);
    }
    int capacity = 1 << 20;
    Trace trace = {malloc(capacity * sizeof(double)), malloc(capacity * sizeof(double)), malloc(capacity * sizeof(double)), 0, 0};
    char line[256];
    double power, time, junction;
    while (fgets(line, sizeof line, file)) {
        if (sscanf(line, "%lf %lf %lf", &power, &time, &junction) != 3) continue;
        if (trace.rows == capacity) {
            capacity *= 2;
            trace.power = realloc(trace.power, capacity * sizeof(double));
            trace.time = realloc(trace.time, capacity * sizeof(double));
            trace.junction = realloc(trace.junction, capacity * sizeof(double));
        }
        trace.power[trace.rows] = power;
        trace.time[trace.rows] = time;
        trace.junction[trace.rows] = junction;
        trace.rows++;
        if (isfinite(junction)) trace.readings++;
    }
    fclose(file);
    return trace;
}

/* R, C and the water numbers are searched as logarithms, so they stay positive and very different sizes behave alike. */
static GpuFit to_fit(const double *numbers, int count)
{
    GpuFit fit = {0};
    for (int i = 0; i < 3; i++) {
        fit.R[i] = exp(numbers[i]);
        fit.C[i] = exp(numbers[i + 3]);
    }
    fit.coolant = numbers[6];
    fit.water_gain = count > LAYER_NUMBERS ? exp(numbers[7]) : 0;
    fit.water_rise_s = count > LAYER_NUMBERS ? exp(numbers[8]) : 100;
    fit.water_settle_s = count > LAYER_NUMBERS ? exp(numbers[9]) : 100;
    return fit;
}

typedef struct {
    double fit_sum;     /* squared errors of the readings being fitted */
    int fit_count;
    double rest_sum;    /* squared errors of the readings held back */
    int rest_count;
} Errors;

/*
 * Model minus sensor at every junction reading, sorted by whether it comes before end_s.
 * One more error discourages a cold plate far slower than the part being fitted: nothing in the data could show it, so it would just be a way to absorb noise.
 */
static Errors compute_errors(const Trace *trace, const double *numbers, int count, double end_s, double *store)
{
    GpuFit fit = to_fit(numbers, count);
    GpuState state;
    gpu_start(&state, &fit, trace->power[0], 1.0, trace->time[0]);

    Errors e = {0, 0, 0, 0};
    for (int row = 0; row < trace->rows; row++) {
        double junction = gpu_predict(&state, &fit, trace->power[row], NAN, trace->time[row]).junction_C;
        if (!isfinite(trace->junction[row])) continue;
        double error = junction - trace->junction[row];
        if (trace->time[row] < end_s) {
            if (store) store[e.fit_count] = error;
            e.fit_sum += error * error;
            e.fit_count++;
        } else {
            e.rest_sum += error * error;
            e.rest_count++;
        }
    }
    double longest_s = (end_s < trace->time[trace->rows - 1] ? end_s : trace->time[trace->rows - 1]) - trace->time[0];
    double too_slow = fmax(log(fit.R[2] * fit.C[2] / (longest_s / 3)), 0) * sqrt(e.fit_count);
    if (store) store[e.fit_count] = too_slow;
    e.fit_sum += too_slow * too_slow;
    return e;
}

/* Solve A x = b by elimination. */
static void solve(double A[ALL_NUMBERS][ALL_NUMBERS], double *b, double *x, int count)
{
    for (int pivot = 0; pivot < count; pivot++) {
        for (int row = pivot + 1; row < count; row++) {
            double factor = A[row][pivot] / A[pivot][pivot];
            for (int col = pivot; col < count; col++)
                A[row][col] -= factor * A[pivot][col];
            b[row] -= factor * b[pivot];
        }
    }
    for (int row = count - 1; row >= 0; row--) {
        x[row] = b[row];
        for (int col = row + 1; col < count; col++)
            x[row] -= A[row][col] * x[col];
        x[row] /= A[row][row];
    }
}

/* Improve the numbers in place, using the readings before end_s. */
static void fit_trace(const Trace *trace, double *numbers, int count, double end_s)
{
    int readings = compute_errors(trace, numbers, count, end_s, NULL).fit_count + 1;   /* + 1 for the cold plate's error */
    double *errors = malloc(readings * sizeof(double));
    double *nudged_errors = malloc(readings * sizeof(double));
    double *slope = malloc(count * readings * sizeof(double));   /* how each error responds to each number */
    double caution = 1e-3;                                       /* larger means smaller steps */
    double cost = compute_errors(trace, numbers, count, end_s, errors).fit_sum;

    for (int iteration = 0; iteration < 200; iteration++) {
        /* Nudge each number */
        for (int j = 0; j < count; j++) {
            double original = numbers[j];
            double nudge = 1e-5 * fmax(fabs(original), 1e-2);
            numbers[j] = original + nudge;
            compute_errors(trace, numbers, count, end_s, nudged_errors);
            numbers[j] = original;
            for (int i = 0; i < readings; i++)
                slope[j * readings + i] = (nudged_errors[i] - errors[i]) / nudge;
        }

        double slope_product[ALL_NUMBERS][ALL_NUMBERS], downhill[ALL_NUMBERS];
        for (int a = 0; a < count; a++) {
            downhill[a] = 0;
            for (int i = 0; i < readings; i++)
                downhill[a] -= slope[a * readings + i] * errors[i];
            for (int b = 0; b < count; b++) {
                slope_product[a][b] = 0;
                for (int i = 0; i < readings; i++)
                    slope_product[a][b] += slope[a * readings + i] * slope[b * readings + i];
            }
        }

        /* Try a step. If it makes things worse, try a more cautious one. */
        double previous_cost = cost;
        for (int attempt = 0; attempt < 30; attempt++) {
            double A[ALL_NUMBERS][ALL_NUMBERS], b[ALL_NUMBERS], step[ALL_NUMBERS], trial[ALL_NUMBERS];
            memcpy(A, slope_product, sizeof A);
            memcpy(b, downhill, sizeof b);
            for (int a = 0; a < count; a++)
                A[a][a] *= 1 + caution;
            solve(A, b, step, count);
            for (int a = 0; a < count; a++)
                trial[a] = numbers[a] + step[a];

            double trial_cost = compute_errors(trace, trial, count, end_s, nudged_errors).fit_sum;
            if (trial_cost < cost) {
                memcpy(numbers, trial, count * sizeof(double));
                memcpy(errors, nudged_errors, readings * sizeof(double));
                cost = trial_cost;
                caution /= 3;
                break;
            }
            caution *= 4;
        }
        if (previous_cost - cost < 1e-10 * previous_cost) break;
    }

    free(errors);
    free(nudged_errors);
    free(slope);
}

static int compare_numbers(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The value that this fraction of the sorted numbers lie below. */
static double percentile(double *numbers, int count, double fraction)
{
    qsort(numbers, count, sizeof(double), compare_numbers);
    return numbers[(int)(fraction * (count - 1))];
}

/* Peak and idle power, over short windows. */
static void find_power_range(const Trace *trace, double *peak_W, double *idle_W)
{
    double *windows = malloc(trace->rows * sizeof(double));
    int count = 0, end = 0;
    double energy = 0;
    for (int start = 0; start < trace->rows; start++) {
        while (end < trace->rows - 1 && trace->time[end] - trace->time[start] < PEAK_WINDOW_S) {
            energy += trace->power[end] * (trace->time[end + 1] - trace->time[end]);
            end++;
        }
        if (trace->time[end] - trace->time[start] < PEAK_WINDOW_S) break;
        windows[count++] = energy / (trace->time[end] - trace->time[start]);
        energy -= trace->power[start] * (trace->time[start + 1] - trace->time[start]);
    }
    *peak_W = percentile(windows, count, 1.0);
    *idle_W = percentile(windows, count, 0.01);
    free(windows);
}

/* How far the sensor reads above the model. */
static double find_margin(const Trace *trace, const GpuFit *fit)
{
    double *above = malloc(trace->readings * sizeof(double));
    int reading = 0;
    GpuState state;
    gpu_start(&state, fit, trace->power[0], 1.0, trace->time[0]);
    for (int row = 0; row < trace->rows; row++) {
        double junction = gpu_predict(&state, fit, trace->power[row], NAN, trace->time[row]).junction_C;
        if (isfinite(trace->junction[row])) above[reading++] = trace->junction[row] - junction;
    }
    double margin = percentile(above, reading, 0.99);
    free(above);
    return margin;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s gpu_fit.h gpu0_samples.txt [gpu1_samples.txt ...]\n", argv[0]);
        return 1;
    }
    FILE *output = fopen(argv[1], "w");
    if (!output) {
        fprintf(stderr, "can't write %s\n", argv[1]);
        return 1;
    }
    fprintf(output, "/* Written by fit_gpu. Don't edit by hand. See gpu_model.h for what these mean. */\n");
    fprintf(output, "static const GpuFit GPU_FITS[] = {\n");

    for (int gpu = 0; gpu < argc - 2; gpu++) {
        Trace trace = read_trace(argv[gpu + 2]);
        double start_s = trace.time[0], seconds = trace.time[trace.rows - 1] - start_s;
        double fit_end_s = start_s + FIT_SHARE * seconds, all_s = start_s + 2 * seconds;

        double peak_W, idle_W;
        find_power_range(&trace, &peak_W, &idle_W);
        double lowest = INFINITY, highest = -INFINITY;
        for (int row = 0; row < trace.rows; row++) {
            if (!isfinite(trace.junction[row])) continue;
            lowest = fmin(lowest, trace.junction[row]);
            highest = fmax(highest, trace.junction[row]);
        }

        /* The first guess: the total R is this run's junction swing per watt, split across the layers, with typical response times */
        double total_R = fmax(highest - lowest, 1.0) / fmax(peak_W - idle_W, 1.0);
        double R0 = 0.3 * total_R, R1 = 0.4 * total_R, R2 = 0.3 * total_R;
        double guess[ALL_NUMBERS] = {log(R0), log(R1), log(R2), log(0.03 / R0), log(1.5 / R1), log(15 / R2), lowest - idle_W * total_R, log(0.02), log(75), log(150)};

        /* Score on the held-back part, with and without the water */
        double layers[ALL_NUMBERS], with_water[ALL_NUMBERS];
        memcpy(layers, guess, sizeof guess);
        fit_trace(&trace, layers, LAYER_NUMBERS, fit_end_s);
        Errors scored = compute_errors(&trace, layers, LAYER_NUMBERS, fit_end_s, NULL);
        double held_back = sqrt(scored.rest_sum / scored.rest_count);

        int count = LAYER_NUMBERS;
        double held_back_steady = held_back;
        if (seconds >= WATER_RUN_S) {
            memcpy(with_water, layers, LAYER_NUMBERS * sizeof(double));
            memcpy(with_water + LAYER_NUMBERS, guess + LAYER_NUMBERS, (ALL_NUMBERS - LAYER_NUMBERS) * sizeof(double));
            fit_trace(&trace, with_water, ALL_NUMBERS, fit_end_s);
            scored = compute_errors(&trace, with_water, ALL_NUMBERS, fit_end_s, NULL);
            double held_back_water = sqrt(scored.rest_sum / scored.rest_count);
            if (held_back_water < WATER_MUST_IMPROVE * held_back) {
                count = ALL_NUMBERS;
                held_back = held_back_water;
                memcpy(layers, with_water, sizeof with_water);
            }
        }

        /* The final numbers use the whole run */
        fit_trace(&trace, layers, count, all_s);
        scored = compute_errors(&trace, layers, count, all_s, NULL);
        GpuFit fit = to_fit(layers, count);
        fit.peak_power = peak_W;
        fit.idle_power = idle_W;
        fit.margin = find_margin(&trace, &fit);
        double slowest_s = fit.R[2] * fit.C[2];

        printf("GPU %d: R %.4f + %.4f + %.4f = %.4f K/W; C %.2f, %.1f, %.0f J/K; coolant %.1f °C; power %.0f to %.0f W; margin %.1f °C\n", gpu,
               fit.R[0], fit.R[1], fit.R[2], fit.R[0] + fit.R[1] + fit.R[2], fit.C[0], fit.C[1], fit.C[2], fit.coolant, idle_W, peak_W, fit.margin);
        printf("    error %.2f °C over %d readings; on the last %.0f%% when fitted only on the rest: %.2f °C\n", sqrt(scored.fit_sum / trace.readings), trace.readings, 100 * (1 - FIT_SHARE), held_back);
        if (count > LAYER_NUMBERS) printf("    water moved: gain %.4f °C/W, rise %.0f s, settle %.0f s (held-back error %.2f °C without it)\n", fit.water_gain, fit.water_rise_s, fit.water_settle_s, held_back_steady);
        if (seconds < 5 * slowest_s) printf("    warning: the run lasted %.0f s, but the cold plate takes about %.0f s to respond. Aim for at least %.0f s.\n", seconds, slowest_s, 5 * slowest_s);
        if (highest - lowest < SMALLEST_SWING_C) printf("    warning: the junction only moved %.0f °C, so these numbers may not carry over to other jobs. Use a run that swings at least %.0f °C.\n", highest - lowest, SMALLEST_SWING_C);
        if (held_back > 2.0) printf("    warning: the held-back error is %.1f °C, more than expected. Did the coolant change during the run?\n", held_back);

        fprintf(output, "    {{%.6g, %.6g, %.6g}, {%.6g, %.6g, %.6g}, %.6g, %.6g, %.6g, %.6g, %.6g, %.6g, %.6g},   /* GPU %d */\n",
                fit.R[0], fit.R[1], fit.R[2], fit.C[0], fit.C[1], fit.C[2], fit.coolant, fit.peak_power, fit.idle_power, fit.margin, fit.water_gain, fit.water_rise_s, fit.water_settle_s, gpu);
    }
    fprintf(output, "};\n");
    fclose(output);
    return 0;
}
