/*
 * Keep a GPU's junction temperature steady. Before each operation, decide whether to hold it back for a moment.
 * While the GPU idles, decide whether to give it filler work.
 *
 * Plain C, no GPU code: the caller supplies the clock, the telemetry and the operations. thermal_rocm.c connects it to HIP and rocBLAS.
 *
 *     model      gpu_model.h, kept up to date from measured power and junction            tg_observe
 *     op costs   how long each kind of operation runs and how much power it draws,
 *                learned as each one finishes                                             tg_op_finished
 *     queue      operations already sent to the GPU, and when they should finish
 *     decide     before an operation: a gap, so the average power over the horizon
 *                stays under what keeps the junction below the top of the band           tg_before_op
 *                while idle: filler work, so the average power stays above what keeps
 *                it over the bottom                                                       tg_fill
 *     report     the hottest the junction could get by the horizon                         tg_worst_case
 *
 * The band is a target ± band_C. The target follows the junction, but it may only rise at rise_C_per_s and fall at fall_C_per_s,
 * and never above ceiling_C. So fast swings are removed, and slow changes, like a job starting or ending, become ramps.
 * Or fix the target with setpoint_C.
 *
 * How a decision works: the forecast turns "keep the junction under the top of the band by the horizon" into an allowed average power
 * (gpu_allowed_power). A following target can rise by rise_C_per_s x horizon_s meanwhile, so the top it aims for moves up by that much. An operation that draws more than that runs at a duty below 1, and the gap before it makes up the rest:
 *
 *     duty = (allowed - idle) / (operation's power - idle)        gap = duration x (1 - duty) / duty, at most max_gap_s
 *
 * Filler is the mirror image: the least average power that keeps the junction over the bottom, delivered as short chunks.
 */
#ifndef THERMAL_CORE_H
#define THERMAL_CORE_H

#include <stdint.h>
#include "../gpu_model.h"

typedef struct {
    double horizon_s;          /* how far ahead the forecast looks */
    double band_C;             /* how far the junction may stray either side of the target */
    double setpoint_C;         /* a fixed target, or NAN for one that follows the junction */
    double rise_C_per_s;       /* how fast a following target may rise */
    double fall_C_per_s;       /* how fast a following target may fall */
    double ceiling_C;          /* a following target never goes above this; NAN = no ceiling */
    double max_gap_s;          /* the longest gap before one operation. Longer gaps become thermal cycles of their own */
    double fill_chunk_s;       /* length of one piece of filler work; 0 = never fill */
    double prior_flops_per_s;  /* the first guess at a new kind of operation's speed */
} TgSettings;

/* Defaults: look 5 s ahead, ±1 °C around a target that rises at most 1 °C/s and falls at most 0.05 °C/s, gaps of at most 5 ms, filler in 1 ms pieces. */
static const TgSettings TG_DEFAULTS = {5.0, 1.0, NAN, 1.0, 0.05, NAN, 0.005, 0.001, 20e12};

typedef struct {
    uint64_t key;              /* the kind of operation: name and sizes */
    double duration_s;         /* learned run time */
    double power_W;            /* learned power while it runs */
    int finished;              /* how many have been measured */
} TgOpCost;

typedef struct {
    long id;
    int cost;                  /* index into costs */
    double start, end;         /* when it should run [s] */
    double power_W;
} TgQueued;

#define TG_COSTS 4096
#define TG_QUEUE 1024

typedef struct {
    TgSettings set;
    GpuFit fit;
    GpuState state;            /* the model, as of the latest telemetry */
    GpuForecast forecast;
    int started;
    double now;                /* time of the latest telemetry [s] */
    double recent_power;       /* measured power, averaged over about a second [W] */
    double target;             /* the middle of the band [°C] */
    double next_fill;          /* filler runs at a duty: the earliest time for the next piece [s] */
    TgOpCost costs[TG_COSTS];
    TgQueued queue[TG_QUEUE];
    int queued;
    long next_id;
    /* Totals, for the log */
    long ops, gaps, fills;
    double gap_s, fill_s;
} Tg;

void tg_init(Tg *tg, const GpuFit *fit, const TgSettings *settings);

/* Telemetry: the average power since the previous call, and the junction reading (NAN if none). Call about every millisecond. */
void tg_observe(Tg *tg, double now, double power_W, double junction_C);

/* An operation is about to be sent. Returns its id, and sets *gap_s: how long the GPU should idle before it starts.
   flops is a size hint for a kind of operation never seen before (0 if unknown). */
long tg_before_op(Tg *tg, double now, uint64_t key, double flops, double *gap_s);

/* An operation has finished on the GPU, between these times. energy_J is what the GPU used meanwhile, or NAN if unknown. */
void tg_op_finished(Tg *tg, long id, double start, double end, double energy_J);

/* While the GPU has nothing to do: returns a filler id and sets *seconds to the filler work to start now, or returns -1 when none is needed. */
long tg_fill(Tg *tg, double now, uint64_t key, double *seconds);

/* The hottest the junction could be by the horizon: peak power all the way, plus the model's margin. */
double tg_worst_case(const Tg *tg);

/* The model's junction now, and the band. */
double tg_junction(const Tg *tg);
double tg_band_top(const Tg *tg);
double tg_band_bottom(const Tg *tg);

/* Read one GPU's numbers from a gpu_fit.h written by fit_gpu. Returns 0 on success. */
int tg_load_fit(const char *path, int gpu, GpuFit *fit);

#endif
