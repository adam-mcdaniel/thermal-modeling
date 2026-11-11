/* See thermal_core.h. */
#include <stdio.h>
#include <string.h>
#include "thermal_core.h"

#define RECENT_POWER_WINDOW_S 1.0   /* "recent power" averages over about this long */
#define LEARNING_RATE 0.2           /* how much each finished operation moves its learned cost */
#define FIRST_GUESS_S 0.001         /* run time assumed for a new kind of operation with no size hint */
#define LOST_AFTER_S 2.0            /* forget an operation this long past its expected end, if it was never measured */

void tg_init(Tg *tg, const GpuFit *fit, const TgSettings *settings)
{
    memset(tg, 0, sizeof *tg);
    tg->set = *settings;
    tg->fit = *fit;
    gpu_forecast_start(&tg->forecast, fit, settings->horizon_s);
    tg->next_id = 1;
}

double tg_junction(const Tg *tg)
{
    return tg->state.steady_coolant + tg->state.rising - tg->state.settling + tg->state.die;
}

double tg_worst_case(const Tg *tg)
{
    return gpu_junction_after(&tg->state, &tg->fit, &tg->forecast, tg->fit.peak_power) + tg->fit.margin;
}

double tg_band_top(const Tg *tg)
{
    return tg->target + tg->set.band_C;
}

double tg_band_bottom(const Tg *tg)
{
    return tg->target - tg->set.band_C;
}

/* Where the band can be by the horizon: a following target may move by its rate meanwhile. */
static double top_by_horizon(const Tg *tg)
{
    if (isfinite(tg->set.setpoint_C)) return tg_band_top(tg);
    double target = tg->target + tg->set.rise_C_per_s * tg->set.horizon_s;
    if (isfinite(tg->set.ceiling_C)) target = fmin(target, tg->set.ceiling_C);
    return target + tg->set.band_C;
}

static double bottom_by_horizon(const Tg *tg)
{
    if (isfinite(tg->set.setpoint_C)) return tg_band_bottom(tg);
    return tg->target - tg->set.fall_C_per_s * tg->set.horizon_s - tg->set.band_C;
}

void tg_observe(Tg *tg, double now, double power_W, double junction_C)
{
    if (!tg->started) {
        gpu_start(&tg->state, &tg->fit, power_W, 0.0, now);
        gpu_predict(&tg->state, &tg->fit, power_W, junction_C, now);
        tg->recent_power = power_W;
        tg->target = isfinite(tg->set.setpoint_C) ? tg->set.setpoint_C : tg_junction(tg);
        tg->now = now;
        tg->started = 1;
        return;
    }
    double dt = now - tg->now;
    if (dt <= 0) return;

    /* The power just measured is what held since the previous sample */
    tg->state.power = power_W;
    gpu_predict(&tg->state, &tg->fit, power_W, junction_C, now);
    tg->recent_power += (power_W - tg->recent_power) * (1 - exp(-dt / RECENT_POWER_WINDOW_S));

    /* The target follows the model's junction, which the sensor keeps honest but which doesn't flicker in 1 °C steps */
    if (isfinite(tg->set.setpoint_C)) {
        tg->target = tg->set.setpoint_C;
    } else {
        double change = tg_junction(tg) - tg->target;
        tg->target += fmin(fmax(change, -tg->set.fall_C_per_s * dt), tg->set.rise_C_per_s * dt);
        if (isfinite(tg->set.ceiling_C)) tg->target = fmin(tg->target, tg->set.ceiling_C);
    }

    /* Forget operations whose measurement never arrived */
    for (int i = 0; i < tg->queued; i++)
        if (tg->queue[i].end < now - LOST_AFTER_S)
            tg->queue[i--] = tg->queue[--tg->queued];
    tg->now = now;
}

/* The learned cost of this kind of operation, adding a first guess for a new one. */
static int find_cost(Tg *tg, uint64_t key, double flops)
{
    int slot = (int)(key % TG_COSTS);
    for (int probe = 0; probe < TG_COSTS; probe++, slot = (slot + 1) % TG_COSTS) {
        TgOpCost *c = &tg->costs[slot];
        if (c->key == key && c->duration_s > 0) return slot;
        if (c->duration_s == 0) {
            c->key = key;
            c->duration_s = flops > 0 ? flops / tg->set.prior_flops_per_s : FIRST_GUESS_S;
            c->power_W = tg->fit.peak_power;     /* until measured, assume the worst */
            c->finished = 0;
            return slot;
        }
    }
    return (int)(key % TG_COSTS);   /* full: share a slot */
}

/* Power the GPU should draw at time t: idle, plus what each queued operation adds. */
static double power_at(const Tg *tg, double t)
{
    double power = tg->fit.idle_power;
    for (int i = 0; i < tg->queued; i++)
        if (tg->queue[i].start <= t && t < tg->queue[i].end)
            power += tg->queue[i].power_W - tg->fit.idle_power;
    return fmin(power, tg->fit.peak_power);
}

static int compare_times(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The model as it will be at time `until`, after the operations already queued. Steps from each start or end to the next. */
static GpuState state_at(const Tg *tg, double until)
{
    static double times[2 * TG_QUEUE + 1];
    int count = 0;
    times[count++] = until;
    for (int i = 0; i < tg->queued; i++) {
        times[count++] = tg->queue[i].start;
        times[count++] = tg->queue[i].end;
    }
    qsort(times, count, sizeof(double), compare_times);

    GpuState model = tg->state;
    double previous = tg->now;
    for (int i = 0; i < count && previous < until; i++) {
        double t = fmin(times[i], until);
        if (t <= previous) continue;
        double power = power_at(tg, 0.5 * (previous + t));
        model.power = power;
        gpu_predict(&model, &tg->fit, power, NAN, t);
        previous = t;
    }
    return model;
}

/* The average power, held from this state until the horizon, that would leave the junction at `limit_C`. */
static double allowed_power(const Tg *tg, const GpuState *model, double limit_C)
{
    return (limit_C - gpu_junction_after(model, &tg->fit, &tg->forecast, 0)) / tg->forecast.per_watt;
}

/* When the GPU will finish the operations already sent (fillers run alongside, so they don't count). */
static double busy_until(const Tg *tg, double now)
{
    double end = now;
    for (int i = 0; i < tg->queued; i++)
        if (tg->queue[i].cost >= 0)
            end = fmax(end, tg->queue[i].end);
    return end;
}

static long add_to_queue(Tg *tg, int cost, double start, double end, double power_W)
{
    if (tg->queued == TG_QUEUE) tg->queue[0] = tg->queue[--tg->queued];   /* full: drop the oldest */
    TgQueued *q = &tg->queue[tg->queued++];
    q->id = tg->next_id++;
    q->cost = cost;
    q->start = start;
    q->end = end;
    q->power_W = power_W;
    return q->id;
}

long tg_before_op(Tg *tg, double now, uint64_t key, double flops, double *gap_s)
{
    int cost = find_cost(tg, key, flops);
    double duration = tg->costs[cost].duration_s, power = tg->costs[cost].power_W;
    double start = busy_until(tg, now), gap = 0;

    if (tg->started && power > tg->fit.idle_power) {
        GpuState model = state_at(tg, start);
        double duty = (allowed_power(tg, &model, top_by_horizon(tg)) - tg->fit.idle_power) / (power - tg->fit.idle_power);
        if (duty <= 0) gap = tg->set.max_gap_s;
        else if (duty < 1) gap = fmin(duration * (1 - duty) / duty, tg->set.max_gap_s);
    }

    *gap_s = gap;
    tg->ops++;
    if (gap > 0) {
        tg->gaps++;
        tg->gap_s += gap;
    }
    return add_to_queue(tg, cost, start + gap, start + gap + duration, power);
}

long tg_fill(Tg *tg, double now, uint64_t key, double *seconds)
{
    *seconds = 0;
    if (tg->set.fill_chunk_s <= 0 || !tg->started || now < tg->next_fill || busy_until(tg, now) > now) return -1;
    for (int i = 0; i < tg->queued; i++)
        if (tg->queue[i].end > now) return -1;     /* filler already running */

    /* The least average power that keeps the junction over the bottom of the band, and the duty of filler that delivers it */
    int cost = find_cost(tg, key, 0);
    double filler_power = tg->costs[cost].power_W;
    GpuState model = state_at(tg, now);
    double duty = (allowed_power(tg, &model, bottom_by_horizon(tg)) - tg->fit.idle_power) / (filler_power - tg->fit.idle_power);
    if (duty <= 0 || filler_power <= tg->fit.idle_power) return -1;
    duty = fmin(duty, 1);

    *seconds = tg->set.fill_chunk_s;
    tg->next_fill = now + *seconds / duty;
    tg->fills++;
    tg->fill_s += *seconds;
    long id = add_to_queue(tg, cost, now, now + *seconds, filler_power);
    tg->queue[tg->queued - 1].cost = -1 - cost;     /* negative: a filler, which doesn't hold up the app's operations */
    return id;
}

void tg_op_finished(Tg *tg, long id, double start, double end, double energy_J)
{
    for (int i = 0; i < tg->queued; i++) {
        if (tg->queue[i].id != id) continue;
        int slot = tg->queue[i].cost >= 0 ? tg->queue[i].cost : -1 - tg->queue[i].cost;
        TgOpCost *c = &tg->costs[slot];
        double duration = end - start;
        if (duration > 0) {
            double rate = c->finished ? LEARNING_RATE : 1.0;
            c->duration_s += (duration - c->duration_s) * rate;
            if (isfinite(energy_J)) c->power_W += (energy_J / duration - c->power_W) * rate;
            c->finished++;
        }
        tg->queue[i] = tg->queue[--tg->queued];
        return;
    }
}

int tg_load_fit(const char *path, int gpu, GpuFit *fit)
{
    FILE *file = fopen(path, "r");
    if (!file) return -1;
    char line[1024];
    int index = 0;
    while (fgets(line, sizeof line, file)) {
        const char *row = strstr(line, "{{");
        if (!row) continue;
        if (index++ != gpu) continue;
        int read = sscanf(row, "{{%lf, %lf, %lf}, {%lf, %lf, %lf}, %lf, %lf, %lf, %lf, %lf, %lf, %lf}",
                          &fit->R[0], &fit->R[1], &fit->R[2], &fit->C[0], &fit->C[1], &fit->C[2], &fit->coolant,
                          &fit->peak_power, &fit->idle_power, &fit->margin, &fit->water_gain, &fit->water_rise_s, &fit->water_settle_s);
        fclose(file);
        return read == 13 ? 0 : -1;
    }
    fclose(file);
    return -1;
}
