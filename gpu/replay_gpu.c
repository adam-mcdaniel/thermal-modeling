/*
 * Replay a power trace on a simulated GPU, with a controller inserting sleeps to hold the junction under a limit. The work is the same; sleeping makes it take longer.
 *
 *     cc -O2 replay_gpu.c -lm -o replay_gpu
 *     ./replay_gpu GPU samples.txt option=value ...
 *
 *     policy=none       never sleep
 *     policy=reactive   model-free: sleep the next period whenever the junction sensor reads at or above the limit
 *     policy=forecast   gpu_sleep_fraction, using the power the work has been drawing recently
 *     policy=worst      gpu_sleep_fraction, always assuming the GPU's peak power
 *     policy=cap        no sleeps: slow the GPU so its power meets gpu_allowed_power
 *     speed_power=N     for policy=cap: dynamic power goes as speed^N. An assumption: 1 means slowing is no better than sleeping, 2 to 3 is typical of lowering the clock
 *     limit=C           the junction limit [°C]
 *     horizon=S         how far the forecast looks ahead [s]
 *     period=S          how often a sleep can be inserted. Each period sleeps first, then works [s]
 *     group=1           the whole cooling group runs like this GPU, so its heat warms the water, and slowing down also cools the water
 *     jump=C            the coolant jumps by this much, unknown to the controller [°C]
 *     jump_at=S         when it jumps, in seconds of simulated time from the start. Default: halfway through the work
 *     csv=PATH          write every step's time, power and junction
 *
 * The simulated GPU uses the fit in gpu_fit.h. The controller uses the fit in CONTROLLER_FIT, so it can be tested with numbers from a different run.
 * It sees the power, and the junction sensor every 6.5 ms in 1 °C steps, as on the real GPU.
 *
 * Prints: seconds of work, seconds taken, energy [J].
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_model.h"
#ifndef CONTROLLER_FIT
#define CONTROLLER_FIT "gpu_fit.h"
#endif
#define GPU_FITS GPU_FITS_TRUTH
#include "gpu_fit.h"
#undef GPU_FITS
#define GPU_FITS CONTROLLER_FITS
#include CONTROLLER_FIT
#undef GPU_FITS

#define SENSOR_PERIOD_S 0.0065
#define WORK_POWER_SMOOTHING_S 1.0    /* "the power the work has been drawing recently" */

static double option(int argc, char **argv, const char *name, double fallback)
{
    size_t length = strlen(name);
    for (int i = 3; i < argc; i++)
        if (!strncmp(argv[i], name, length) && argv[i][length] == '=') return atof(argv[i] + length + 1);
    return fallback;
}

static const char *text_option(int argc, char **argv, const char *name, const char *fallback)
{
    size_t length = strlen(name);
    for (int i = 3; i < argc; i++)
        if (!strncmp(argv[i], name, length) && argv[i][length] == '=') return argv[i] + length + 1;
    return fallback;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s GPU samples.txt [policy=none|reactive|forecast|worst|cap] [speed_power=N] [limit=C] [horizon=S] [period=S] [group=1] [jump=C] [jump_at=S] [csv=PATH]\n", argv[0]);
        return 1;
    }
    int gpu = atoi(argv[1]);
    const GpuFit *truth_fit = &GPU_FITS_TRUTH[gpu];
    const GpuFit *controller_fit = &CONTROLLER_FITS[gpu];
    const char *policy = text_option(argc, argv, "policy", "none");
    double limit = option(argc, argv, "limit", 1000);
    double horizon = option(argc, argv, "horizon", 5);
    double period = option(argc, argv, "period", 0.001);
    double jump = option(argc, argv, "jump", 0);
    double group_share = option(argc, argv, "group", 0);
    double speed_power = option(argc, argv, "speed_power", 2);
    const char *csv_path = text_option(argc, argv, "csv", NULL);
    FILE *csv = csv_path ? fopen(csv_path, "w") : NULL;
    if (csv) fprintf(csv, "time_s,power_W,junction_C,sleep_fraction,work_s\n");

    /* The work: its power, as a function of how far through the work we are */
    FILE *file = fopen(argv[2], "r");
    if (!file) {
        fprintf(stderr, "can't open %s\n", argv[2]);
        return 1;
    }
    int capacity = 1 << 16, rows = 0;
    double *work_time = malloc(capacity * sizeof(double)), *work_power = malloc(capacity * sizeof(double));
    char line[256];
    while (fgets(line, sizeof line, file)) {
        double power, time;
        char reading[64] = "";
        if (sscanf(line, "%lf %lf %63s", &power, &time, reading) < 2) continue;
        if (reading[0] && strcmp(reading, "nan")) continue;    /* keep the power samples, skip the rows that only carry a sensor reading */
        if (rows == capacity) {
            capacity *= 2;
            work_time = realloc(work_time, capacity * sizeof(double));
            work_power = realloc(work_power, capacity * sizeof(double));
        }
        work_time[rows] = time;
        work_power[rows] = power;
        rows++;
    }
    fclose(file);

    GpuState truth, controller;
    GpuForecast forecast;
    double time = work_time[0], work = work_time[0], energy = 0;
    double sleep_idle = truth_fit->idle_power;
    gpu_start(&truth, truth_fit, work_power[0], group_share, time);
    gpu_start(&controller, controller_fit, work_power[0], group_share, time);
    gpu_forecast_start(&forecast, controller_fit, horizon);

    double jump_time = work_time[0] + option(argc, argv, "jump_at", (work_time[rows - 1] - work_time[0]) / 2);
    double recent_power = work_power[0];
    double next_sensor = time;
    double latest_reading = truth_fit->coolant + truth.die;
    double wake_time = time, next_period = time;
    double fraction = 0, speed = 1;
    int capped = !strcmp(policy, "cap");
    int row = 0, jumped = 0;

    while (row < rows - 1) {
        /* Time for a new decision? */
        if (time >= next_period - 1e-12) {
            if (!strcmp(policy, "reactive")) fraction = latest_reading >= limit ? 1 : 0;
            else if (!strcmp(policy, "forecast")) fraction = gpu_sleep_fraction(&controller, controller_fit, &forecast, limit, recent_power);
            else if (!strcmp(policy, "worst")) fraction = gpu_sleep_fraction(&controller, controller_fit, &forecast, limit, controller_fit->peak_power);
            else fraction = 0;
            if (capped) {
                double allowed = gpu_allowed_power(&controller, controller_fit, &forecast, limit);
                double share = (allowed - sleep_idle) / fmax(recent_power - sleep_idle, 1.0);
                speed = fmin(fmax(pow(fmax(share, 0), 1 / speed_power), 0.05), 1);
            }
            wake_time = time + fraction * period;
            next_period = time + period;
        }

        /* Move to the next thing that happens: waking, a new period, a sensor reading, or the work's power changing */
        int asleep = time < wake_time - 1e-12;
        double power = asleep ? sleep_idle : work_power[row];
        if (capped && power > sleep_idle) power = sleep_idle + (power - sleep_idle) * pow(speed, speed_power);
        double until = next_sensor;
        if (asleep) until = fmin(until, wake_time);
        else until = fmin(until, fmin(next_period, time + (work_time[row + 1] - work) / speed));
        double gap = fmax(until - time, 1e-9);

        /* gpu_predict holds the state's power over the gap, then takes the next power, so set the power for this gap first */
        truth.power = power;
        controller.power = power;
        energy += power * gap;
        time += gap;
        if (!asleep) {
            work += gap * speed;
            recent_power += (work_power[row] - recent_power) * (1 - exp(-gap * speed / WORK_POWER_SMOOTHING_S));
            while (row < rows - 1 && work >= work_time[row + 1] - 1e-12) row++;
        }

        if (jump != 0 && !jumped && time >= jump_time) {
            truth.steady_coolant += jump;
            jumped = 1;
        }
        GpuPrediction actual = gpu_predict(&truth, truth_fit, power, NAN, time);
        double reading = NAN;
        if (time >= next_sensor - 1e-12) {
            latest_reading = round(actual.junction_C);
            reading = latest_reading;
            next_sensor += SENSOR_PERIOD_S;
        }
        gpu_predict(&controller, controller_fit, power, reading, time);
        if (csv) fprintf(csv, "%.6f,%.2f,%.4f,%.4f,%.6f\n", time, power, actual.junction_C, fraction, work);
    }
    if (csv) fclose(csv);
    printf("%.3f %.3f %.0f\n", work_time[rows - 1] - work_time[0], time - work_time[0], energy);
    return 0;
}
