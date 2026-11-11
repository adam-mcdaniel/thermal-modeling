/*
 * A rocBLAS workload that swings the junction temperature on purpose: square waves, sweeps through every time scale, random
 * bursts, power steps, and fit_gpu's calibration recipe. Use it to calibrate, to stress the cooling, and to test the governor.
 *
 *     cc -O2 thermal_stress.c -lrocblas -lamdhip64 -lm -lpthread -ldl -o thermal_stress     (with ROCm's lib directory on -L)
 *     ./thermal_stress square --period 10 --duty 0.5 --seconds 120 --samples samples.txt
 *
 * Patterns (the GPU idles for --idle seconds before and after each):
 *
 *     square    on for duty x period, off for the rest                       --period S --duty F
 *     chirp     a square wave whose period sweeps from --from to --to        --from S --to S --duty F
 *               over --seconds, evenly on a log scale. One run passes through the die's (~30 ms), the package's (~1.5 s)
 *               and the cold plate's (~16 s) response times, so it finds the period that swings the junction most.
 *     random    on and off, for random lengths between --shortest and        --shortest S --longest S --seed N
 *               --longest, evenly on a log scale: no period for anything to settle into
 *     steps     each level in --levels held for --hold seconds               --levels 0,0.25,0.5,1 --hold S
 *     recipe    fit_gpu's calibration run: 20 s idle, 120 s full, 50 s idle, 26 s of 1 s on, 1 s off, then 25 s of 0.1 s on, 0.1 s off
 *
 * A level between 0 and 1 runs that fraction of each 10 ms frame. "On" is DGEMMs of size --size back to back.
 *
 * Two clocks:
 *     (default)   follow the schedule by the clock. Use for calibration and for stressing the cooling.
 *     --work      do a fixed amount of work: each on-stretch runs the DGEMMs it would take at full speed, however long that takes,
 *                 and each off-stretch idles for its scheduled time. Under the governor the gaps then show up as a longer run, so
 *                 compare the runtime with and without it.
 *
 * Output:
 *     --samples FILE    power and junction every millisecond, in fit_gpu's format, from the energy counter and junction sensor
 *     --schedule FILE   what was asked for: time_s, level
 *     stdout            how long it took and how many DGEMMs ran
 */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpu_telemetry.h"

/* The few HIP and rocBLAS functions needed, declared here so no ROCm headers are needed to build */
typedef void *rocblas_handle;
int hipSetDevice(int device);
int hipGetDevice(int *device);
int hipDeviceGetPCIBusId(char *address, int length, int device);
int hipMalloc(void **pointer, size_t bytes);
int hipMemset(void *pointer, int value, size_t bytes);
int hipStreamSynchronize(void *stream);
int rocblas_create_handle(rocblas_handle *handle);
int rocblas_get_stream(rocblas_handle handle, void **stream);
int rocblas_dgemm(rocblas_handle handle, int trans_a, int trans_b, int m, int n, int k, const double *alpha, const double *a, int lda,
                  const double *b, int ldb, const double *beta, double *c, int ldc);
#define NO_TRANSPOSE 111

#define FRAME_S 0.01
#define MAX_SEGMENTS 1000000

typedef struct {
    double seconds, level;
} Segment;

static Segment *segments;
static int segment_count;

static void add(double seconds, double level)
{
    if (seconds <= 0) return;
    if (segment_count == MAX_SEGMENTS) {
        fprintf(stderr, "too many segments\n");
        exit(1);
    }
    segments[segment_count++] = (Segment){seconds, level};
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void sleep_until(double when)
{
    double left = when - now_s();
    if (left <= 0) return;
    struct timespec ts = {(time_t)left, (long)((left - (time_t)left) * 1e9)};
    nanosleep(&ts, NULL);
}

static const char *option(int argc, char **argv, const char *name, const char *fallback)
{
    for (int i = 2; i < argc - 1; i++)
        if (!strcmp(argv[i], name)) return argv[i + 1];
    return fallback;
}

static int flag(int argc, char **argv, const char *name)
{
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], name)) return 1;
    return 0;
}

/* --- Telemetry: a thread that logs power and junction every millisecond ---------------------------------------------------- */
static volatile int done;
static Telemetry telemetry;
static FILE *samples_file;

static void *log_telemetry(void *unused)
{
    (void)unused;
    double start = now_s(), next = start, last_time = NAN, last_energy = NAN;
    while (!done) {
        next += 0.001;
        sleep_until(next);
        double t = now_s(), energy, junction;
        tl_read(&telemetry, &energy, &junction);
        if (isfinite(last_energy) && isfinite(energy) && t > last_time) {
            double power = (energy - last_energy) / (t - last_time);
            fprintf(samples_file, "%.3f %.6f nan\n%.3f %.6f %.0f\n", power, last_time - start, power, t - start, junction);
        }
        last_time = t;
        last_energy = energy;
    }
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s square|chirp|random|steps|recipe [options]  (see the top of thermal_stress.c)\n", argv[0]);
        return 1;
    }
    const char *pattern = argv[1];
    double seconds = atof(option(argc, argv, "--seconds", "120"));
    double idle = atof(option(argc, argv, "--idle", "10"));
    double duty = atof(option(argc, argv, "--duty", "0.5"));
    int size = atoi(option(argc, argv, "--size", "4096"));
    int device = atoi(option(argc, argv, "--device", "0"));
    int work_clock = flag(argc, argv, "--work");
    segments = malloc(MAX_SEGMENTS * sizeof(Segment));

    /* The schedule */
    add(idle, 0);
    if (!strcmp(pattern, "square")) {
        double period = atof(option(argc, argv, "--period", "10"));
        for (double t = 0; t < seconds - 1e-9; t += period) {
            add(duty * period, 1);
            add((1 - duty) * period, 0);
        }
    } else if (!strcmp(pattern, "chirp")) {
        double from = atof(option(argc, argv, "--from", "0.01")), to = atof(option(argc, argv, "--to", "60"));
        for (double t = 0; t < seconds;) {
            double period = from * pow(to / from, t / seconds);
            add(duty * period, 1);
            add((1 - duty) * period, 0);
            t += period;
        }
    } else if (!strcmp(pattern, "random")) {
        double shortest = atof(option(argc, argv, "--shortest", "0.02")), longest = atof(option(argc, argv, "--longest", "20"));
        srand((unsigned)atoi(option(argc, argv, "--seed", "1")));
        for (double t = 0; t < seconds;) {
            double length = shortest * pow(longest / shortest, (double)rand() / RAND_MAX);
            add(length, rand() % 2);
            t += length;
        }
    } else if (!strcmp(pattern, "steps")) {
        char levels[1024];
        snprintf(levels, sizeof levels, "%s", option(argc, argv, "--levels", "0,0.25,0.5,0.75,1"));
        double hold = atof(option(argc, argv, "--hold", "30"));
        for (char *level = strtok(levels, ","); level; level = strtok(NULL, ","))
            add(hold, atof(level));
    } else if (!strcmp(pattern, "recipe")) {
        add(20, 0);
        add(120, 1);
        add(50, 0);
        for (int i = 0; i < 13; i++) {
            add(1, 1);
            add(1, 0);
        }
        for (int i = 0; i < 125; i++) {
            add(0.1, 1);
            add(0.1, 0);
        }
    } else {
        fprintf(stderr, "unknown pattern %s\n", pattern);
        return 1;
    }
    add(idle, 0);

    /* The GPU and the matrices */
    hipSetDevice(device);
    rocblas_handle handle;
    void *stream = NULL;
    rocblas_create_handle(&handle);
    rocblas_get_stream(handle, &stream);
    size_t bytes = (size_t)size * size * sizeof(double);
    double *a, *b, *c, one = 1, zero = 0;
    if (hipMalloc((void **)&a, bytes) || hipMalloc((void **)&b, bytes) || hipMalloc((void **)&c, bytes)) {
        fprintf(stderr, "can't allocate 3 matrices of size %d\n", size);
        return 1;
    }
    hipMemset(a, 0, bytes);
    hipMemset(b, 0, bytes);

    /* How long one DGEMM takes, after warming up */
    for (int i = 0; i < 5; i++)
        rocblas_dgemm(handle, NO_TRANSPOSE, NO_TRANSPOSE, size, size, size, &one, a, size, b, size, &zero, c, size);
    hipStreamSynchronize(stream);
    double before = now_s();
    for (int i = 0; i < 20; i++)
        rocblas_dgemm(handle, NO_TRANSPOSE, NO_TRANSPOSE, size, size, size, &one, a, size, b, size, &zero, c, size);
    hipStreamSynchronize(stream);
    double gemm_s = (now_s() - before) / 20;
    fprintf(stderr, "one DGEMM of size %d takes %.3f ms (%.1f TFLOP/s)\n", size, 1e3 * gemm_s, 2.0 * size * size * size / gemm_s * 1e-12);

    /* Telemetry, if asked for */
    pthread_t logger;
    const char *samples_path = option(argc, argv, "--samples", NULL);
    if (samples_path) {
        char address[64] = "";
        hipDeviceGetPCIBusId(address, sizeof address, device);
        samples_file = fopen(samples_path, "w");
        if (!samples_file || tl_open(&telemetry, address, atoi(option(argc, argv, "--smi", "-1")))) {
            fprintf(stderr, "can't log power and junction for GPU %s\n", address);
            return 1;
        }
        pthread_create(&logger, NULL, log_telemetry, NULL);
    }
    FILE *schedule = NULL;
    const char *schedule_path = option(argc, argv, "--schedule", NULL);
    if (schedule_path) {
        schedule = fopen(schedule_path, "w");
        fprintf(schedule, "time_s,level\n");
    }

    /* Run it: each segment in frames, each frame on for level x frame, then off */
    double start = now_s(), planned = start, owed = 0;
    long gemms = 0;
    for (int s = 0; s < segment_count; s++) {
        if (schedule) fprintf(schedule, "%.4f,%.3f\n", (work_clock ? now_s() : planned) - start, segments[s].level);
        double frames = ceil(segments[s].seconds / FRAME_S - 1e-9), frame = segments[s].seconds / frames;
        for (int f = 0; f < frames; f++) {
            double on = segments[s].level * frame;
            if (work_clock) {
                owed += on / gemm_s;                          /* the DGEMMs full speed would fit in, carried between frames */
                for (; owed >= 1; owed -= 1, gemms++)
                    rocblas_dgemm(handle, NO_TRANSPOSE, NO_TRANSPOSE, size, size, size, &one, a, size, b, size, &zero, c, size);
                hipStreamSynchronize(stream);
                sleep_until(now_s() + frame - on);
            } else {
                /* Against the schedule's own clock, so small delays don't add up */
                while (on > 0 && now_s() < planned + on - 0.5 * gemm_s) {
                    rocblas_dgemm(handle, NO_TRANSPOSE, NO_TRANSPOSE, size, size, size, &one, a, size, b, size, &zero, c, size);
                    hipStreamSynchronize(stream);
                    gemms++;
                }
                planned += frame;
                sleep_until(planned);
            }
        }
    }
    double took = now_s() - start;

    done = 1;
    if (samples_path) {
        pthread_join(logger, NULL);
        fclose(samples_file);
    }
    if (schedule) fclose(schedule);
    double scheduled = 0;
    for (int s = 0; s < segment_count; s++) scheduled += segments[s].seconds;
    printf("%s: %.2f s (scheduled %.2f s), %ld DGEMMs of size %d\n", pattern, took, scheduled, gemms, size);
    return 0;
}
