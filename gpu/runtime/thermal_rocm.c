/*
 * Connect thermal_core.c to a real AMD GPU: HIP, rocBLAS and rocm_smi, all looked up at run time, so this builds without ROCm.
 *
 *     thermal_start     once, before the first operation: load the GPU's numbers and start the telemetry thread
 *     thermal_before    just before a rocBLAS call: ask the core for a gap, and if there is one, let the GPU finish
 *                       what it has, then wait that long. Then mark the operation's start on its stream.
 *     thermal_after     just after the call returns (the GPU may still be running it): mark its end on the stream
 *     thermal_stop      at exit: stop the thread, print a summary
 *
 * The telemetry thread, every millisecond:
 *     1. read the energy counter and junction, and update the model (tg_observe)
 *     2. collect operations the GPU has finished: their times from HIP events, their energy from the counter (tg_op_finished)
 *     3. if the GPU is idle and the core asks for it, run filler work: DGEMMs on a low-priority stream (tg_fill)
 *     4. write a log row
 *
 * The filler calls rocBLAS through dlsym(RTLD_NEXT), which skips the tapir wrapper, so the filler isn't itself governed.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "thermal_core.h"
#include "gpu_telemetry.h"

typedef struct {
    TgSettings settings;
    const char *fit_path;    /* a gpu_fit.h written by fit_gpu */
    int fit_gpu;             /* which GPU's numbers in it; -1 = the device the program uses */
    const char *log_path;    /* NULL = no log */
    double log_every_s;
    int smi_index;           /* rocm_smi's number for the GPU, or -1 to find it from the PCI address */
    int fill_size;           /* filler DGEMMs are this size, n x n */
    int dry_run;             /* decide and log, but don't apply */
} ThermalOptions;

#define TICK_S 0.001
#define PENDING 8192          /* operations being timed at once */
#define ENERGY_HISTORY 60000  /* 60 s of telemetry at 1 ms */
#define FILLER_KEY 0x66696c6cULL

/* HIP and rocBLAS, as plain C function pointers */
typedef void *Handle;
static struct {
    int (*get_device)(int *);
    int (*set_device)(int);
    int (*pci_bus_id)(char *, int, int);
    int (*stream_synchronize)(void *);
    int (*stream_create_with_priority)(void **, unsigned, int);
    int (*stream_priority_range)(int *, int *);
    int (*event_create)(void **);
    int (*event_record)(void *, void *);
    int (*event_query)(void *);
    int (*event_synchronize)(void *);
    int (*event_elapsed)(float *, void *, void *);
    int (*device_malloc)(void **, size_t);
    int (*device_memset)(void *, int, size_t);
    int (*get_stream)(Handle, void **);
    int (*create_handle)(Handle *);
    int (*set_stream)(Handle, void *);
    int (*dgemm)(Handle, int, int, int, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
} api;

typedef struct {
    int state;               /* 0 free, 1 started, 2 ended */
    long id;
    void *start, *stop;      /* HIP events */
    int filler;              /* number of filler DGEMMs, or 0 for an app operation */
} Pending;

static struct {
    int on;
    ThermalOptions options;
    pthread_mutex_t lock;
    pthread_t thread;
    volatile int stopping;
    int device;
    Telemetry telemetry;
    Tg core;
    Pending pending[PENDING];
    long next_pending;
    void *reference;         /* a HIP event at a known host time: GPU times are measured from it */
    double reference_time;
    double energy_time[ENERGY_HISTORY], energy_J[ENERGY_HISTORY];
    long energy_samples;
    Handle filler_handle;
    void *filler_stream;
    double *filler_matrices;
    double filler_gemm_s;    /* measured time of one filler DGEMM */
    FILE *log;
} g = {.lock = PTHREAD_MUTEX_INITIALIZER};

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* Sleep until a moment, sleeping most of the way and spinning the last 50 us, so short gaps come out right. */
static void wait_until(double when)
{
    double left = when - now_s();
    if (left > 60e-6) {
        double sleep_s = left - 50e-6;
        struct timespec ts = {(time_t)sleep_s, (long)((sleep_s - (time_t)sleep_s) * 1e9)};
        nanosleep(&ts, NULL);
    }
    while (now_s() < when) {}
}

static int load_api(void)
{
#define FIND(field, name, where) if (!(api.field = (__typeof__(api.field))dlsym(where, name))) { fprintf(stderr, "thermal: %s not found\n", name); return -1; }
    FIND(get_device, "hipGetDevice", RTLD_DEFAULT)
    FIND(set_device, "hipSetDevice", RTLD_DEFAULT)
    FIND(pci_bus_id, "hipDeviceGetPCIBusId", RTLD_DEFAULT)
    FIND(stream_synchronize, "hipStreamSynchronize", RTLD_DEFAULT)
    FIND(stream_create_with_priority, "hipStreamCreateWithPriority", RTLD_DEFAULT)
    FIND(stream_priority_range, "hipDeviceGetStreamPriorityRange", RTLD_DEFAULT)
    FIND(event_create, "hipEventCreate", RTLD_DEFAULT)
    FIND(event_record, "hipEventRecord", RTLD_DEFAULT)
    FIND(event_query, "hipEventQuery", RTLD_DEFAULT)
    FIND(event_synchronize, "hipEventSynchronize", RTLD_DEFAULT)
    FIND(event_elapsed, "hipEventElapsedTime", RTLD_DEFAULT)
    FIND(device_malloc, "hipMalloc", RTLD_DEFAULT)
    FIND(device_memset, "hipMemset", RTLD_DEFAULT)
    FIND(get_stream, "rocblas_get_stream", RTLD_NEXT)
    FIND(create_handle, "rocblas_create_handle", RTLD_NEXT)
    FIND(set_stream, "rocblas_set_stream", RTLD_NEXT)
    FIND(dgemm, "rocblas_dgemm", RTLD_NEXT)
#undef FIND
    return 0;
}

/* A HIP event's time on the host clock. */
static double event_time(void *event)
{
    float ms = 0;
    api.event_elapsed(&ms, g.reference, event);
    return g.reference_time + 1e-3 * ms;
}

/* Energy used by time t [J], from the telemetry history, or NAN if t is outside it. */
static double energy_at(double t)
{
    long newest = g.energy_samples - 1, oldest = g.energy_samples > ENERGY_HISTORY ? g.energy_samples - ENERGY_HISTORY : 0;
    if (newest < 1 || t < g.energy_time[oldest % ENERGY_HISTORY] || t > g.energy_time[newest % ENERGY_HISTORY]) return NAN;
    for (long i = newest; i > oldest; i--) {
        double t0 = g.energy_time[(i - 1) % ENERGY_HISTORY], t1 = g.energy_time[i % ENERGY_HISTORY];
        if (t >= t0) {
            double e0 = g.energy_J[(i - 1) % ENERGY_HISTORY], e1 = g.energy_J[i % ENERGY_HISTORY];
            return t1 > t0 ? e0 + (e1 - e0) * (t - t0) / (t1 - t0) : e1;
        }
    }
    return NAN;
}

/* Take a free slot for timing an operation, or -1 if all are in use. */
static int take_slot(void)
{
    for (int tries = 0; tries < 16; tries++) {
        int slot = (int)(__atomic_fetch_add(&g.next_pending, 1, __ATOMIC_RELAXED) % PENDING);
        int free_state = 0;
        if (__atomic_compare_exchange_n(&g.pending[slot].state, &free_state, -1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            if (!g.pending[slot].start) {
                api.event_create(&g.pending[slot].start);
                api.event_create(&g.pending[slot].stop);
            }
            return slot;
        }
    }
    return -1;
}

static void launch_filler(double seconds, long id)
{
    int slot = take_slot();
    if (slot < 0) return;
    int n = g.options.fill_size, count = (int)fmax(1, round(seconds / g.filler_gemm_s));
    double one = 1, zero = 0, *a = g.filler_matrices, *b = a + (size_t)n * n, *c = b + (size_t)n * n;
    Pending *p = &g.pending[slot];
    p->id = id;
    p->filler = count;
    api.event_record(p->start, g.filler_stream);
    for (int i = 0; i < count; i++)
        api.dgemm(g.filler_handle, 111, 111, n, n, n, &one, a, n, b, n, &zero, c, n);   /* 111 = rocblas_operation_none */
    api.event_record(p->stop, g.filler_stream);
    __atomic_store_n(&p->state, 2, __ATOMIC_RELEASE);
}

/* Measure finished operations and teach the core what they cost. */
static void collect_finished(void)
{
    for (int slot = 0; slot < PENDING; slot++) {
        Pending *p = &g.pending[slot];
        if (__atomic_load_n(&p->state, __ATOMIC_ACQUIRE) != 2 || api.event_query(p->stop) != 0) continue;
        double start = event_time(p->start), end = event_time(p->stop);
        double energy = energy_at(end) - energy_at(start);
        if (p->filler && end > start) g.filler_gemm_s += ((end - start) / p->filler - g.filler_gemm_s) * 0.3;
        pthread_mutex_lock(&g.lock);
        tg_op_finished(&g.core, p->id, start, end, energy);
        pthread_mutex_unlock(&g.lock);
        p->filler = 0;
        __atomic_store_n(&p->state, 0, __ATOMIC_RELEASE);
    }
}

static void *telemetry_loop(void *unused)
{
    (void)unused;
    api.set_device(g.device);
    double next = now_s(), next_log = next, last_time = NAN, last_energy = NAN;
    while (!g.stopping) {
        next += TICK_S;
        wait_until(next);
        double t = now_s(), energy, junction;
        tl_read(&g.telemetry, &energy, &junction);
        if (isfinite(energy)) {
            g.energy_time[g.energy_samples % ENERGY_HISTORY] = t;
            g.energy_J[g.energy_samples % ENERGY_HISTORY] = energy;
            g.energy_samples++;
        }
        if (isfinite(last_energy) && isfinite(energy) && t > last_time) {
            pthread_mutex_lock(&g.lock);
            tg_observe(&g.core, t, (energy - last_energy) / (t - last_time), junction);
            pthread_mutex_unlock(&g.lock);
        }
        last_time = t;
        last_energy = energy;

        collect_finished();

        if (!g.options.dry_run && g.filler_handle) {
            double seconds;
            pthread_mutex_lock(&g.lock);
            long id = tg_fill(&g.core, t, FILLER_KEY, &seconds);
            pthread_mutex_unlock(&g.lock);
            if (id >= 0) launch_filler(seconds, id);
        }

        if (g.log && t >= next_log) {
            pthread_mutex_lock(&g.lock);
            Tg *c = &g.core;
            fprintf(g.log, "%.4f,%.1f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%ld,%.4f,%ld,%.4f\n", t, c->state.power, junction, tg_junction(c), c->target,
                    tg_band_top(c), tg_band_bottom(c), tg_worst_case(c), c->gaps, c->gap_s, c->fills, c->fill_s);
            pthread_mutex_unlock(&g.lock);
            next_log += g.options.log_every_s;
        }
    }
    return NULL;
}

/* Set up on the first operation, once HIP is ready. Returns 0 if governing is on. */
int thermal_start(const ThermalOptions *options)
{
    if (g.on) return 0;
    g.options = *options;
    GpuFit fit;
    if (load_api()) return -1;
    api.get_device(&g.device);
    int fit_gpu = options->fit_gpu >= 0 ? options->fit_gpu : g.device;
    if (tg_load_fit(options->fit_path, fit_gpu, &fit)) {
        fprintf(stderr, "thermal: can't read GPU %d from %s, so not governing\n", fit_gpu, options->fit_path);
        return -1;
    }
    char address[64] = "";
    api.pci_bus_id(address, sizeof address, g.device);
    if (tl_open(&g.telemetry, address, options->smi_index)) {
        fprintf(stderr, "thermal: can't read power and temperature for GPU %s, so not governing\n", address);
        return -1;
    }
    tg_init(&g.core, &fit, &options->settings);

    /* GPU times are measured from this event */
    api.event_create(&g.reference);
    api.event_record(g.reference, NULL);
    api.event_synchronize(g.reference);
    g.reference_time = now_s();

    if (options->settings.fill_chunk_s > 0 && options->fill_size > 0) {
        int least, greatest;
        size_t bytes = 3 * (size_t)options->fill_size * options->fill_size * sizeof(double);
        api.stream_priority_range(&least, &greatest);
        if (api.stream_create_with_priority(&g.filler_stream, 0, least) == 0 && api.create_handle(&g.filler_handle) == 0 &&
            api.device_malloc((void **)&g.filler_matrices, bytes) == 0) {
            api.device_memset(g.filler_matrices, 0, bytes);
            api.set_stream(g.filler_handle, g.filler_stream);
            g.filler_gemm_s = 2.0 * options->fill_size * options->fill_size * options->fill_size / options->settings.prior_flops_per_s;
        } else {
            g.filler_handle = NULL;
            fprintf(stderr, "thermal: can't set up filler work, so only gaps\n");
        }
    }

    if (options->log_path) {
        g.log = fopen(options->log_path, "w");
        if (g.log) fprintf(g.log, "time_s,power_W,junction_C,model_C,target_C,top_C,bottom_C,worst_case_C,gaps,gap_s,fills,fill_s\n");
    }
    g.on = 1;
    pthread_create(&g.thread, NULL, telemetry_loop, NULL);
    fprintf(stderr, "thermal: governing GPU %d (%s) with fit %d from %s%s\n", g.device, address, fit_gpu, options->fit_path, options->dry_run ? ", dry run" : "");
    return 0;
}

/* Before a rocBLAS call with this handle. Returns a token for thermal_after. */
long thermal_before(Handle handle, uint64_t key, double flops)
{
    if (!g.on) return -1;
    double gap;
    pthread_mutex_lock(&g.lock);
    long id = tg_before_op(&g.core, now_s(), key, flops, &gap);
    pthread_mutex_unlock(&g.lock);

    void *stream = NULL;
    api.get_stream(handle, &stream);
    if (gap > 0 && !g.options.dry_run) {
        api.stream_synchronize(stream);
        wait_until(now_s() + gap);
    }
    int slot = take_slot();
    if (slot < 0) return -1;
    Pending *p = &g.pending[slot];
    p->id = id;
    p->filler = 0;
    api.event_record(p->start, stream);
    __atomic_store_n(&p->state, 1, __ATOMIC_RELEASE);
    return slot;
}

void thermal_after(Handle handle, long token)
{
    if (!g.on || token < 0) return;
    void *stream = NULL;
    api.get_stream(handle, &stream);
    Pending *p = &g.pending[token];
    api.event_record(p->stop, stream);
    __atomic_store_n(&p->state, 2, __ATOMIC_RELEASE);
}

void thermal_stop(void)
{
    if (!g.on) return;
    g.stopping = 1;
    pthread_join(g.thread, NULL);
    if (g.log) fclose(g.log);
    fprintf(stderr, "thermal: %ld operations, %ld gaps (%.3f s), %ld filler pieces (%.3f s)\n",
            g.core.ops, g.core.gaps, g.core.gap_s, g.core.fills, g.core.fill_s);
    g.on = 0;
}
