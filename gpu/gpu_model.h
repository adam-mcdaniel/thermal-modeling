/*
 * One GPU on its cold plate.
 *
 *     power --> [ die ] --R1--> [ package ] --R2--> [ cold plate ] --R3--> coolant
 *                 C1               C2                  C3
 *
 * Heat flows from one layer to the next at (temperature difference) / R, and each layer warms by (heat in - heat out) / C.
 * From power alone, that gives how far each layer sits above the coolant: its "lift".
 *
 *     coolant  = measured junction - die lift
 *     junction = coolant + die lift
 *
 * The coolant moves too. When a whole cooling group runs the same job, the heat its GPUs put in warms the water before the CDU catches up,
 * and then the water settles back. The heat entering the water drives that directly:
 *
 *     rising   moves toward (water_gain x heat into the water), over water_rise_s
 *     settling moves toward rising, over water_settle_s
 *     coolant  = steady value + rising - settling
 *
 * With no junction reading, that alone sets the coolant. A junction reading also corrects the steady value.
 * All the numbers come from fit_gpu (gpu_fit.h), from power and the junction sensor. Its run must have the whole cooling group busy to show the water moving.
 * Otherwise water_gain is 0, and the water stays steady.
 *
 * Looking ahead: the layers are linear, so the junction a few seconds from now is a weighted sum of what's already there:
 *
 *     junction later = coolant + (a x die) + (b x package) + (c x plate) + (d x power held until then)
 *
 * gpu_forecast_start finds a, b, c, d once for a chosen horizon. Each forecast after that is those few multiplications.
 *     gpu_junction_after    the junction at the horizon, for a power you choose
 *     gpu_worst_case        the same at the highest power this GPU has sustained, plus the model's error margin
 *     gpu_allowed_power     the most power that keeps the junction under a limit
 *     gpu_sleep_fraction    how much of the time to sleep to get down to that power
 */
#ifndef GPU_MODEL_H
#define GPU_MODEL_H

#include <math.h>

/* Only flux_* and water_out_C use these two, so override them for another GPU with -D. The junction doesn't depend on them. */
#ifndef DIE_AREA_CM2
#define DIE_AREA_CM2 14.48         /* MI250X: 2 dies of 724 mm^2 each */
#endif
#ifndef WATER_CAPACITY
#define WATER_CAPACITY 223.0       /* power the water passing this GPU absorbs per 1 °C it warms [W/K]; an estimate, not fitted */
#endif
#define LONGEST_STEP_S 0.001       /* the layers move forward in steps no longer than this */
#define COOLANT_SMOOTHING_S 10.0   /* the junction sensor reads in 1 °C steps, so the coolant estimate is smoothed over this long */
#define RECENT_POWER_S 0.2         /* forecasts assume recent power over this long continues. The layers themselves always get every raw sample. */

typedef struct {
    double R[3];       /* die, package, cold plate: resistance to the next layer [K/W] */
    double C[3];       /* die, package, cold plate: heat capacity [J/K] */
    double coolant;    /* coolant during the fitted run: the starting estimate [°C] */
    double peak_power; /* highest power held for 200 ms during the fitted run [W] */
    double idle_power; /* power when nearly idle during the fitted run [W] */
    double margin;     /* the sensor read at most this far above the model on 99% of the fitted run's readings [°C] */
    double water_gain;     /* how far the coolant would move per watt into the water, before settling [°C/W]; 0 = it stays steady */
    double water_rise_s;   /* how fast the coolant responds to heat [s] */
    double water_settle_s; /* how fast it settles back as the cooling catches up [s] */
} GpuFit;

typedef struct {
    double die, package, plate;    /* how far each layer sits above the coolant [°C] */
    double power;                  /* the latest power, which holds until the next sample [W] */
    double recent_power;           /* power smoothed over RECENT_POWER_S: what a forecast assumes continues [W] */
    double time;                   /* the latest sample's time [s] */
    double steady_coolant;         /* the coolant's value when the water has settled [°C] */
    double steady_time;            /* when a junction reading last corrected it; negative infinity before the first reading [s] */
    double group_share;            /* fraction of the cooling group running like this GPU: 1 = all of it, 0 = only this GPU */
    double rising, settling;       /* the coolant's response to the GPU's heat [°C] */
} GpuState;

typedef struct {
    double junction_C;       /* junction temperature */
    double coolant_C;        /* coolant reaching this GPU's plate */
    double water_out_C;      /* coolant leaving this GPU's plate */
    double flux_generated;   /* heat made by the chip, power / die area [W/cm^2] */
    double flux_die_exit;    /* heat leaving the die into the package [W/cm^2] */
    double flux_to_water;    /* heat entering the coolant [W/cm^2 of die area] */
} GpuPrediction;

/* The forecast's weights for one horizon. */
typedef struct {
    double horizon_s;
    double from_die, from_package, from_plate;   /* how much of each layer's lift remains at the horizon */
    double per_watt;                             /* junction rise at the horizon per watt held from now until then [°C/W] */
} GpuForecast;

/* Move the layers and the water's response forward by dt, with the power held. */
static void gpu_move(GpuState *state, const GpuFit *fit, double dt)
{
    double die_to_package = (state->die - state->package) / fit->R[0];
    double package_to_plate = (state->package - state->plate) / fit->R[1];
    double plate_to_water = state->plate / fit->R[2];

    state->die += dt * (state->power - die_to_package) / fit->C[0];
    state->package += dt * (die_to_package - package_to_plate) / fit->C[1];
    state->plate += dt * (package_to_plate - plate_to_water) / fit->C[2];

    double target = state->group_share * fit->water_gain * plate_to_water;
    state->rising += (target - state->rising) * (1 - exp(-dt / fit->water_rise_s));
    state->settling += (state->rising - state->settling) * (1 - exp(-dt / fit->water_settle_s));
}

/* Start settled at this power: the same heat flows through every layer. */
static void gpu_start(GpuState *state, const GpuFit *fit, double power_W, double group_share, double time_s)
{
    state->plate = power_W * fit->R[2];
    state->package = state->plate + power_W * fit->R[1];
    state->die = state->package + power_W * fit->R[0];
    state->power = power_W;
    state->recent_power = power_W;
    state->time = time_s;
    state->steady_coolant = fit->coolant;
    state->steady_time = -INFINITY;
    state->group_share = group_share;
    state->rising = state->settling = group_share * fit->water_gain * power_W;
}

/* Give it the next sample, in time order. junction_C is the measured junction, or NAN when there's no reading. */
static GpuPrediction gpu_predict(GpuState *state, const GpuFit *fit, double power_W, double junction_C, double time_s)
{
    /* The previous power held from the previous sample until now */
    double gap = time_s - state->time;
    int steps = (int)ceil(gap / LONGEST_STEP_S);
    double dt = steps > 0 ? gap / steps : 0;

    for (int step = 0; step < steps; step++)
        gpu_move(state, fit, dt);
    state->recent_power += (state->power - state->recent_power) * (1 - exp(-gap / RECENT_POWER_S));
    state->power = power_W;
    state->time = time_s;

    /* The coolant's response to the GPU's heat */
    double response = state->rising - state->settling;

    /* The junction reading, minus the lift from the GPU's own power, is the coolant. Whatever the response doesn't explain corrects the steady value. */
    if (isfinite(junction_C)) {
        double reading = junction_C - state->die - response;
        double keep = exp(-(time_s - state->steady_time) / COOLANT_SMOOTHING_S);     /* the first reading replaces the fitted value outright: today's coolant may differ */
        state->steady_coolant = reading + (state->steady_coolant - reading) * keep;
        state->steady_time = time_s;
    }
    double coolant = state->steady_coolant + response;

    double heat_to_water = state->plate / fit->R[2];
    GpuPrediction prediction;
    prediction.junction_C = coolant + state->die;
    prediction.coolant_C = coolant;
    prediction.water_out_C = coolant + heat_to_water / WATER_CAPACITY;
    prediction.flux_generated = power_W / DIE_AREA_CM2;
    prediction.flux_die_exit = (state->die - state->package) / fit->R[0] / DIE_AREA_CM2;
    prediction.flux_to_water = heat_to_water / DIE_AREA_CM2;
    return prediction;
}

/* Run the model forward once per input, with that input set to 1 and everything else 0. What's left at the horizon is its weight. */
static void gpu_forecast_start(GpuForecast *forecast, const GpuFit *fit, double horizon_s)
{
    GpuState units[4] = {{.die = 1}, {.package = 1}, {.plate = 1}, {.power = 1}};
    int steps = (int)ceil(horizon_s / LONGEST_STEP_S);
    for (int unit = 0; unit < 4; unit++)
        for (int step = 0; step < steps; step++)
            gpu_move(&units[unit], fit, horizon_s / steps);
    forecast->horizon_s = horizon_s;
    forecast->from_die = units[0].die;
    forecast->from_package = units[1].die;
    forecast->from_plate = units[2].die;
    forecast->per_watt = units[3].die;
}

/* The junction at the horizon if this power holds from now until then. Call after gpu_predict. */
static double gpu_junction_after(const GpuState *state, const GpuFit *fit, const GpuForecast *forecast, double power_W)
{
    double plate_to_water = state->plate / fit->R[2];
    double rate = (state->group_share * fit->water_gain * plate_to_water - state->rising) / fit->water_rise_s - (state->rising - state->settling) / fit->water_settle_s;
    double coolant_later = state->steady_coolant + state->rising - state->settling + rate * forecast->horizon_s;
    return coolant_later + forecast->from_die * state->die + forecast->from_package * state->package + forecast->from_plate * state->plate + forecast->per_watt * power_W;
}

/*
 * The hottest the junction can be at the horizon. More power can only make the junction hotter, so holding the peak is the worst this GPU can do.
 * A heating run keeps climbing, so its hottest moment is at the horizon, and this bounds the whole time until then.
 */
static double gpu_worst_case(const GpuState *state, const GpuFit *fit, const GpuForecast *forecast)
{
    return gpu_junction_after(state, fit, forecast, fit->peak_power) + fit->margin;
}

/* The most power that, held from now until the horizon, leaves the junction at limit_C. Use it directly as a power cap. */
static double gpu_allowed_power(const GpuState *state, const GpuFit *fit, const GpuForecast *forecast, double limit_C)
{
    return (limit_C - fit->margin - gpu_junction_after(state, fit, forecast, 0)) / forecast->per_watt;
}

/*
 * The fraction of the next while to sleep to average down to the allowed power: 0 = run flat out, 1 = sleep the whole time.
 * work_power_W is what the GPU draws when it's awake. Each sleep must be much shorter than the die's response, or the sleeps become thermal cycles of their own: see README.
 */
static double gpu_sleep_fraction(const GpuState *state, const GpuFit *fit, const GpuForecast *forecast, double limit_C, double work_power_W)
{
    if (work_power_W <= fit->idle_power) return 0;
    double fraction = (work_power_W - gpu_allowed_power(state, fit, forecast, limit_C)) / (work_power_W - fit->idle_power);
    return fmin(fmax(fraction, 0), 1);
}

#endif
