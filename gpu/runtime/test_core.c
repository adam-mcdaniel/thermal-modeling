/*
 * Test thermal_core.c without a GPU: a simulated GPU runs a workload, and the controller decides gaps and filler as it would for real.
 *
 *     cc -O2 test_core.c thermal_core.c -lm -o test_core
 *     ./test_core PATTERN off|on|nofill out.csv [setting=value ...]
 *
 * PATTERN is square (5 s of work, 5 s idle), hammer (50 ms work, 50 ms idle), random (work and idle of random length) or hpl
 * (the rocHPL power trace, replayed as operations). Settings are TgSettings' names without units: horizon, band, setpoint, rise, fall,
 * ceiling, max_gap, fill.
 *
 * The simulated GPU uses GPU 0's numbers from ../gpu_fit.h. The controller gets those numbers deliberately wrong (R +8%, C -15%, coolant +0.5 °C),
 * and sees power every millisecond and the junction every 6.5 ms in 1 °C steps.
 *
 * Each workload is a list of steps: run an operation (it takes as long as it takes), or think (the app leaves the GPU idle for a while).
 * The controller's gaps and filler come on top, so the run takes longer. Writes: time_s, power_W, junction_C, target_C.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "thermal_core.h"

#define STEP_S 0.00025
#define TELEMETRY_S 0.001
#define SENSOR_S 0.0065
#define WORK_W 560.0
#define FILLER_W 560.0

typedef struct {
    int think;              /* 1 = the app leaves the GPU idle */
    double seconds, power_W;
    uint64_t key;
} Step;

static Step *steps;
static int count, capacity;

static void add(int think, double seconds, double power_W, uint64_t key)
{
    if (count == capacity) {
        capacity = capacity ? 2 * capacity : 1024;
        steps = realloc(steps, capacity * sizeof(Step));
    }
    steps[count++] = (Step){think, seconds, power_W, key};
}

/* Work for `seconds` as 1 ms operations, or think. */
static void phase(int work, double seconds)
{
    if (!work) {
        add(1, seconds, 0, 0);
        return;
    }
    for (double t = 0; t < seconds - 1e-9; t += 0.001)
        add(0, 0.001, WORK_W, 1);
}

static void build(const char *pattern)
{
    if (!strcmp(pattern, "square")) {
        phase(0, 5);
        for (int i = 0; i < 12; i++) {
            phase(1, 5);
            phase(0, 5);
        }
    } else if (!strcmp(pattern, "hammer")) {
        phase(0, 5);
        for (int i = 0; i < 600; i++) {
            phase(1, 0.05);
            phase(0, 0.05);
        }
        phase(0, 5);
    } else if (!strcmp(pattern, "random")) {
        srand(7);
        phase(0, 5);
        for (double t = 0; t < 120;) {
            double seconds = 0.02 * pow(500, (double)rand() / RAND_MAX);     /* 20 ms to 10 s, evenly on a log scale */
            phase(rand() % 2, seconds);
            t += seconds;
        }
        phase(0, 5);
    } else if (!strcmp(pattern, "hpl")) {
        /* Each power sample becomes an operation of that length, keyed by its power to the nearest 50 W. Low power counts as thinking. */
        FILE *file = fopen("../samples/gpu0.txt", "r");
        if (!file) {
            fprintf(stderr, "can't open ../samples/gpu0.txt\n");
            exit(1);
        }
        double power, time, junction, previous_time = -1, previous_power = 0;
        char line[256];
        while (fgets(line, sizeof line, file)) {
            if (sscanf(line, "%lf %lf %lf", &power, &time, &junction) < 2 || isfinite(junction)) continue;
            if (previous_time >= 0 && time > previous_time) {
                double seconds = time - previous_time;
                if (previous_power < 150) add(1, seconds, 0, 0);
                else add(0, seconds, previous_power, 100 + (uint64_t)(previous_power / 50));
            }
            previous_time = time;
            previous_power = power;
        }
        fclose(file);
    } else {
        fprintf(stderr, "unknown pattern %s\n", pattern);
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s square|hammer|random|hpl off|on|nofill out.csv [setting=value ...]\n", argv[0]);
        return 1;
    }
    build(argv[1]);
    int control = strcmp(argv[2], "off") != 0;
    FILE *out = fopen(argv[3], "w");

    GpuFit truth, guess;
    if (tg_load_fit("../gpu_fit.h", 0, &truth)) {
        fprintf(stderr, "can't read ../gpu_fit.h\n");
        return 1;
    }
    guess = truth;
    for (int i = 0; i < 3; i++) {
        guess.R[i] *= 1.08;
        guess.C[i] *= 0.85;
    }
    guess.coolant += 0.5;

    TgSettings settings = TG_DEFAULTS;
    for (int i = 4; i < argc; i++) {
        char name[32];
        double value;
        if (sscanf(argv[i], "%31[^=]=%lf", name, &value) != 2) continue;
        if (!strcmp(name, "horizon")) settings.horizon_s = value;
        if (!strcmp(name, "band")) settings.band_C = value;
        if (!strcmp(name, "setpoint")) settings.setpoint_C = value;
        if (!strcmp(name, "rise")) settings.rise_C_per_s = value;
        if (!strcmp(name, "fall")) settings.fall_C_per_s = value;
        if (!strcmp(name, "ceiling")) settings.ceiling_C = value;
        if (!strcmp(name, "max_gap")) settings.max_gap_s = value;
        if (!strcmp(name, "fill")) settings.fill_chunk_s = value;
    }
    if (!strcmp(argv[2], "nofill")) settings.fill_chunk_s = 0;
    static Tg tg;
    tg_init(&tg, &guess, &settings);

    GpuState gpu;
    gpu_start(&gpu, &truth, truth.idle_power, 0.0, 0.0);
    double t = 0, energy = 0, telemetry_energy = 0, next_telemetry = 0, next_sensor = 0;
    double reading = NAN;

    /* The app's progress: which step, and when the current one ends. An operation first waits out its gap. */
    int step = 0, running = 0;
    double step_end = 0, gap_end = 0, op_start = 0, op_energy = 0;
    long op_id = -1;
    double filler_end = -1;
    long filler_id = -1;
    double filler_start = 0, filler_energy = 0;

    fprintf(out, "time_s,power_W,junction_C,target_C\n");
    while (step < count) {
        /* Start the next step when the current one is done */
        if (!running) {
            Step *s = &steps[step];
            if (s->think) {
                step_end = t + s->seconds;
            } else {
                double gap = 0;
                if (control) op_id = tg_before_op(&tg, t, s->key, 0, &gap);
                gap_end = fmax(t + gap, filler_end);     /* a running filler holds the GPU */
                step_end = gap_end + s->seconds;
                op_start = gap_end;
                op_energy = 0;
            }
            running = 1;
        }

        Step *s = &steps[step];
        int app_working = !s->think && t >= gap_end;
        int filling = t < filler_end;
        double power = app_working ? s->power_W : filling ? FILLER_W : truth.idle_power;

        gpu.power = power;
        double junction = gpu_predict(&gpu, &truth, power, NAN, t + STEP_S).junction_C;
        t += STEP_S;
        energy += power * STEP_S;
        telemetry_energy += power * STEP_S;
        if (app_working) op_energy += power * STEP_S;
        if (filling) filler_energy += power * STEP_S;

        if (t >= next_sensor) {
            reading = round(junction);
            next_sensor += SENSOR_S;
        }
        if (t >= next_telemetry - 1e-12) {
            if (control) {
                tg_observe(&tg, t, telemetry_energy / TELEMETRY_S, reading);
                reading = NAN;
            }
            telemetry_energy = 0;
            next_telemetry += TELEMETRY_S;
            fprintf(out, "%.4f,%.1f,%.3f,%.3f\n", t, power, junction, control ? tg.target : NAN);

            if (control && filler_id >= 0 && t >= filler_end) {
                tg_op_finished(&tg, filler_id, filler_start, filler_end, filler_energy);
                filler_id = -1;
            }
            double seconds;
            if (control && filler_id < 0 && (s->think || t < gap_end)) {
                long id = tg_fill(&tg, t, 2, &seconds);
                if (id >= 0) {
                    filler_id = id;
                    filler_start = t;
                    filler_end = t + seconds;
                    filler_energy = 0;
                }
            }
        }

        if (t >= step_end - 1e-12) {
            if (control && !s->think) tg_op_finished(&tg, op_id, op_start, step_end, op_energy);
            running = 0;
            step++;
        }
    }
    fclose(out);
    fprintf(stderr, "%s %s: %.2f s, %.0f J, %ld ops, %ld gaps (%.2f s), %ld fills (%.2f s)\n",
            argv[1], argv[2], t, energy, tg.ops, tg.gaps, tg.gap_s, tg.fills, tg.fill_s);
    return 0;
}
