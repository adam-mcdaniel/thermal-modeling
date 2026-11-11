/*
 * fit_cooling_model: estimate the constants used by predict_gpu.c and predict_cooling.c from logged data.
 * It only needs the C standard library.
 *
 *     cc -O2 fit_cooling_model.c -lm -o fit_cooling_model
 *
 * ONE GPU (needs only power and junction temperature)
 *
 *     ./fit_cooling_model gpu lux gpu0.csv gpu1.csv ...      writes gpu_lux.h
 *
 *     Columns: time_s, power_W, junction_C. Leave junction_C empty on rows with no reading.
 *     Use a run where the coolant stayed steady, because this fit assumes it did.
 *
 * A COOLING GROUP (needs cooling-system telemetry)
 *
 *     ./fit_cooling_model group lux log.csv [train_seconds] [heat_fraction]      writes cooling_lux.h
 *
 *     Columns, in any order:
 *         time_s                          seconds, or a date-time like 2026-02-24T20:14:13Z
 *         power_kW_1, power_kW_2, ...     one per cabinet; this sets the cabinet count
 *         supply_C_1, supply_C_2, ...     water entering each cabinet (or one shared supply_C)
 *         outlet_C_1, outlet_C_2, ...     water leaving each cabinet
 *         facility_in_C, facility_out_C   building water entering and leaving the CDU
 *         facility_gpm                    building-water flow
 *     The run needs a big power step, such as a job starting and stopping across the whole group.
 *
 * HOW THE NUMBERS ARE FOUND
 *
 *     1. Guess the numbers.
 *     2. Run the model over the logged inputs.
 *     3. Compare with the logged outputs.
 *     4. Nudge each number to see how the error responds, then step to reduce the error.
 *     5. Repeat until the error stops falling.
 *
 *     A cooling group is fitted in 3 stages (cabinets, heat exchanger, valve), so an error in one part can't hide in another.
 *     Only the first part of the log is used to search. The rest tests the result on data it never saw.
 */

#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WATTS_PER_KELVIN_PER_GPM (6.309e-5 * 998.0 * 4180.0)   /* heat that 1 gpm of water carries per 1 °C it warms */

#define MAX_COLUMNS 64
#define MAX_CABINETS 16
#define MAX_GPUS 64
#define MAX_SEARCH_NUMBERS 8


static void fail(const char *message, const char *detail)
{
    fprintf(stderr, "error: %s %s\n", message, detail ? detail : "");
    exit(1);
}

static double *new_array(int length)
{
    double *array = calloc((size_t)length, sizeof(double));
    if (!array) fail("out of memory", NULL);
    return array;
}

static double clamp(double value, double low, double high)
{
    return fmin(fmax(value, low), high);
}

static double rms_difference(const double *a, const double *b, int from, int to)
{
    if (to <= from) return NAN;
    double sum = 0;
    for (int i = from; i < to; i++) {
        double difference = a[i] - b[i];
        sum += difference * difference;
    }
    return sqrt(sum / (to - from));
}


/* ======================================== Reading a CSV by column name ======================================== */

typedef struct {
    int column_count;
    int row_count;
    char names[MAX_COLUMNS][48];
    double *values[MAX_COLUMNS];
} Table;

/* A number, or a date-time turned into seconds. Empty or unreadable cells become NAN. */
static double parse_cell(const char *text)
{
    char *end;
    double number = strtod(text, &end);
    while (isspace((unsigned char)*end)) end++;
    if (end != text && *end == 0) return number;

    struct tm date;
    memset(&date, 0, sizeof date);
    char *rest = strptime(text, "%Y-%m-%dT%H:%M:%S", &date);
    if (!rest) return NAN;
    double fraction_of_second = (*rest == '.') ? strtod(rest, NULL) : 0;
    return (double)timegm(&date) + fraction_of_second;
}

static Table read_csv(const char *path)
{
    Table table;
    memset(&table, 0, sizeof table);

    FILE *file = fopen(path, "r");
    if (!file) fail("can't open", path);

    static char line[1 << 16];
    if (!fgets(line, sizeof line, file)) fail("empty file", path);
    for (char *name = strtok(line, ",\r\n"); name && table.column_count < MAX_COLUMNS; name = strtok(NULL, ",\r\n"))
        snprintf(table.names[table.column_count++], 48, "%s", name);

    int capacity = 1 << 16;
    for (int c = 0; c < table.column_count; c++)
        table.values[c] = malloc(sizeof(double) * (size_t)capacity);

    while (fgets(line, sizeof line, file)) {
        if (table.row_count == capacity) {
            capacity *= 2;
            for (int c = 0; c < table.column_count; c++)
                table.values[c] = realloc(table.values[c], sizeof(double) * (size_t)capacity);
        }
        char *cell = line;
        for (int c = 0; c < table.column_count; c++) {
            char *comma = strchr(cell, ',');
            if (comma) *comma = 0;
            else cell[strcspn(cell, "\r\n")] = 0;
            table.values[c][table.row_count] = parse_cell(cell);
            cell = comma ? comma + 1 : cell + strlen(cell);
        }
        table.row_count++;
    }
    fclose(file);
    return table;
}

static double *find_column(const Table *table, const char *name)
{
    for (int c = 0; c < table->column_count; c++)
        if (!strcmp(table->names[c], name)) return table->values[c];
    return NULL;
}

/* Copy a column onto an even 1 s grid, by straight lines between rows. */
static double *to_one_second_grid(const double *times, const double *values, int row_count, int seconds, double scale)
{
    double *grid = new_array(seconds);
    int row = 0;
    for (int second = 0; second < seconds; second++) {
        while (row < row_count - 2 && times[row + 1] - times[0] <= second) row++;
        double fraction = (second - (times[row] - times[0])) / (times[row + 1] - times[row]);
        fraction = clamp(fraction, 0, 1);
        grid[second] = (values[row] * (1 - fraction) + values[row + 1] * fraction) * scale;
    }
    return grid;
}


/* ======================================== The search ========================================
 * Levenberg-Marquardt: finds the numbers that make a model's errors as small as possible, staying inside the given limits.
 * Every model below provides a function that fills in `errors` (model - measurement) for a given set of numbers. */

typedef void (*ErrorFunction)(const double *numbers, double *errors, void *model_data);

static double sum_of_squares(const double *values, int count)
{
    double sum = 0;
    for (int i = 0; i < count; i++) sum += values[i] * values[i];
    return sum;
}

/* Solve A x = b for a symmetric positive matrix A (Cholesky). Returns 0 if A isn't positive. */
static int solve_linear_system(int size, double A[MAX_SEARCH_NUMBERS][MAX_SEARCH_NUMBERS], const double *b, double *x)
{
    double L[MAX_SEARCH_NUMBERS][MAX_SEARCH_NUMBERS];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j <= i; j++) {
            double value = A[i][j];
            for (int k = 0; k < j; k++) value -= L[i][k] * L[j][k];
            if (i == j) {
                if (value <= 0) return 0;
                L[i][i] = sqrt(value);
            } else {
                L[i][j] = value / L[j][j];
            }
        }
    }
    double y[MAX_SEARCH_NUMBERS];
    for (int i = 0; i < size; i++) {
        double value = b[i];
        for (int k = 0; k < i; k++) value -= L[i][k] * y[k];
        y[i] = value / L[i][i];
    }
    for (int i = size - 1; i >= 0; i--) {
        double value = y[i];
        for (int k = i + 1; k < size; k++) value -= L[k][i] * x[k];
        x[i] = value / L[i][i];
    }
    return 1;
}

/* Improve `numbers` in place. Returns the final RMS error. */
static double search(ErrorFunction compute_errors, void *model_data, double *numbers, const double *low, const double *high, int number_count, int error_count)
{
    double *errors = new_array(error_count);
    double *trial_errors = new_array(error_count);
    double *sensitivity = new_array(number_count * error_count);     /* how each error responds to each number */
    double damping = 1e-3;                                           /* larger means smaller, more cautious steps */

    compute_errors(numbers, errors, model_data);
    double cost = sum_of_squares(errors, error_count);

    for (int iteration = 0; iteration < 200; iteration++) {
        /* Nudge each number to see how every error responds */
        for (int j = 0; j < number_count; j++) {
            double nudge = fmax(1e-5 * fabs(numbers[j]), 1e-7);
            double original = numbers[j];
            if (numbers[j] + nudge > high[j]) nudge = -nudge;
            numbers[j] = original + nudge;
            compute_errors(numbers, trial_errors, model_data);
            numbers[j] = original;
            for (int i = 0; i < error_count; i++)
                sensitivity[j * error_count + i] = (trial_errors[i] - errors[i]) / nudge;
        }

        double JTJ[MAX_SEARCH_NUMBERS][MAX_SEARCH_NUMBERS], gradient[MAX_SEARCH_NUMBERS];
        int held_at_limit[MAX_SEARCH_NUMBERS];
        for (int a = 0; a < number_count; a++) {
            gradient[a] = 0;
            for (int i = 0; i < error_count; i++) gradient[a] += sensitivity[a * error_count + i] * errors[i];
            for (int b = 0; b <= a; b++) {
                double sum = 0;
                for (int i = 0; i < error_count; i++) sum += sensitivity[a * error_count + i] * sensitivity[b * error_count + i];
                JTJ[a][b] = JTJ[b][a] = sum;
            }
            /* A number resting on a limit and pushing against it stays put */
            held_at_limit[a] = (numbers[a] <= low[a] && gradient[a] > 0) || (numbers[a] >= high[a] && gradient[a] < 0);
        }

        /* Try steps, more cautious each time, until one reduces the error */
        int improved = 0;
        for (int attempt = 0; attempt < 25 && !improved; attempt++) {
            double A[MAX_SEARCH_NUMBERS][MAX_SEARCH_NUMBERS], b[MAX_SEARCH_NUMBERS], step[MAX_SEARCH_NUMBERS], trial[MAX_SEARCH_NUMBERS];
            for (int a = 0; a < number_count; a++) {
                for (int c = 0; c < number_count; c++)
                    A[a][c] = (held_at_limit[a] || held_at_limit[c]) ? 0 : JTJ[a][c];
                double scale = JTJ[a][a] > 0 ? JTJ[a][a] : 1;
                A[a][a] = held_at_limit[a] ? 1 : JTJ[a][a] + damping * scale;
                b[a] = held_at_limit[a] ? 0 : -gradient[a];
            }
            if (!solve_linear_system(number_count, A, b, step)) {
                damping *= 4;
                continue;
            }
            for (int a = 0; a < number_count; a++)
                trial[a] = clamp(numbers[a] + step[a], low[a], high[a]);

            compute_errors(trial, trial_errors, model_data);
            double trial_cost = sum_of_squares(trial_errors, error_count);

            if (isfinite(trial_cost) && trial_cost < cost) {
                double relative_gain = (cost - trial_cost) / cost;
                memcpy(numbers, trial, sizeof(double) * (size_t)number_count);
                memcpy(errors, trial_errors, sizeof(double) * (size_t)error_count);
                cost = trial_cost;
                damping = fmax(damping / 3, 1e-12);
                improved = 1;
                if (relative_gain < 1e-10) iteration = 1000;         /* barely improving: done */
            } else {
                damping *= 4;
            }
        }
        if (!improved) break;
    }

    free(errors);
    free(trial_errors);
    free(sensitivity);
    return sqrt(cost / error_count);
}


/* ======================================== Cooling group: the log ======================================== */

typedef struct {
    int cabinet_count;
    int seconds;                          /* length of the log */
    int train_seconds;                    /* the first part, used for searching; the rest is for testing */
    double heat_fraction;                 /* fraction of electrical power that ends up in the water */
    double *power[MAX_CABINETS];          /* W, after heat_fraction */
    double *inlet[MAX_CABINETS];          /* °C */
    double *outlet[MAX_CABINETS];         /* °C */
    double *supply;                       /* °C: the average of the cabinet inlets */
    double *facility_in;                  /* °C */
    double *facility_out;                 /* °C */
    double *facility_flow;                /* gpm */
} GroupLog;

static GroupLog load_group_log(const char *path, double train_seconds, double heat_fraction)
{
    Table table = read_csv(path);
    const double *times = find_column(&table, "time_s");
    const double *facility_in = find_column(&table, "facility_in_C");
    const double *facility_out = find_column(&table, "facility_out_C");
    const double *facility_flow = find_column(&table, "facility_gpm");
    if (!times || !facility_in || !facility_out || !facility_flow)
        fail("need columns time_s, facility_in_C, facility_out_C, facility_gpm in", path);

    GroupLog log;
    memset(&log, 0, sizeof log);
    char name[48];
    while (log.cabinet_count < MAX_CABINETS) {
        snprintf(name, sizeof name, "power_kW_%d", log.cabinet_count + 1);
        if (!find_column(&table, name)) break;
        log.cabinet_count++;
    }
    if (log.cabinet_count == 0) fail("need columns power_kW_1, power_kW_2, ... in", path);

    int rows = table.row_count;
    log.seconds = (int)floor(times[rows - 1] - times[0]) + 1;
    if (train_seconds <= 0) train_seconds = log.seconds / 2.0;
    log.train_seconds = (int)fmin(ceil(train_seconds), log.seconds);
    log.heat_fraction = heat_fraction;
    log.supply = new_array(log.seconds);

    for (int cabinet = 0; cabinet < log.cabinet_count; cabinet++) {
        snprintf(name, sizeof name, "power_kW_%d", cabinet + 1);
        log.power[cabinet] = to_one_second_grid(times, find_column(&table, name), rows, log.seconds, 1e3 * heat_fraction);

        snprintf(name, sizeof name, "supply_C_%d", cabinet + 1);
        const double *inlet = find_column(&table, name);
        if (!inlet) inlet = find_column(&table, "supply_C");
        if (!inlet) fail("need supply_C columns in", path);
        log.inlet[cabinet] = to_one_second_grid(times, inlet, rows, log.seconds, 1);

        snprintf(name, sizeof name, "outlet_C_%d", cabinet + 1);
        const double *outlet = find_column(&table, name);
        if (!outlet) fail("missing column", name);
        log.outlet[cabinet] = to_one_second_grid(times, outlet, rows, log.seconds, 1);

        for (int k = 0; k < log.seconds; k++)
            log.supply[k] += log.inlet[cabinet][k] / log.cabinet_count;
    }
    log.facility_in = to_one_second_grid(times, facility_in, rows, log.seconds, 1);
    log.facility_out = to_one_second_grid(times, facility_out, rows, log.seconds, 1);
    log.facility_flow = to_one_second_grid(times, facility_flow, rows, log.seconds, 1);
    return log;
}

/* A signal's value `delay` seconds before second k, by straight lines between seconds. */
static double value_before(const double *signal, int length, int k, double delay)
{
    double time = k - delay;
    if (time <= 0) return signal[0];
    if (time >= length - 1) return signal[length - 1];
    int whole = (int)time;
    double fraction = time - whole;
    return signal[whole] * (1 - fraction) + signal[whole + 1] * fraction;
}


/* ======================================== Stage 1: one cabinet ========================================
 * Inputs: measured power and inlet. Output: outlet temperature. */

typedef struct {
    double heat_capacity;       /* Ch [J/K] */
    double conductance;         /* UA [W/K] */
    double stream_capacity;     /* mcp [W/K] */
    double inlet_delay;         /* [s] */
    double sensor_offset;       /* [°C] */
} Cabinet;

static void simulate_cabinet(const double *power, const double *inlet, int seconds, const Cabinet *cabinet, double *outlet)
{
    double effectiveness = 1 - exp(-cabinet->conductance / cabinet->stream_capacity);
    double removal = effectiveness * cabinet->stream_capacity;
    double keep_each_second = exp(-removal / cabinet->heat_capacity);

    double hardware = value_before(inlet, seconds, 0, cabinet->inlet_delay) + power[0] / removal;    /* start settled */
    double previous_inlet = 0;
    for (int k = 0; k < seconds; k++) {
        double water_in = value_before(inlet, seconds, k, cabinet->inlet_delay);
        if (k > 0) {
            double settling_point = previous_inlet + power[k - 1] / removal;
            hardware = settling_point + (hardware - settling_point) * keep_each_second;
        }
        outlet[k] = water_in + effectiveness * (hardware - water_in) + cabinet->sensor_offset;
        previous_inlet = water_in;
    }
}

typedef struct {
    const double *power, *inlet, *outlet;
    int seconds, train_seconds;
    double scale;               /* the search works on values relative to a rough stream capacity, so any machine size works */
} CabinetData;

static void cabinet_errors(const double *numbers, double *errors, void *model_data)
{
    CabinetData *data = model_data;
    static double *modelled = NULL;
    static int capacity = 0;
    if (data->seconds > capacity) {
        free(modelled);
        modelled = new_array(data->seconds);
        capacity = data->seconds;
    }
    Cabinet cabinet = {numbers[0] * data->scale, numbers[1] * data->scale, numbers[2] * data->scale, numbers[3], numbers[4]};
    simulate_cabinet(data->power, data->inlet, data->seconds, &cabinet, modelled);
    for (int k = 0; k < data->train_seconds; k++)
        errors[k] = modelled[k] - data->outlet[k];
}

static Cabinet fit_cabinet(const GroupLog *log, int index, double *fit_error)
{
    double mean_power = 0, mean_rise = 0;
    for (int k = 0; k < log->seconds; k++) {
        mean_power += log->power[index][k] / log->seconds;
        mean_rise += (log->outlet[index][k] - log->inlet[index][k]) / log->seconds;
    }
    CabinetData data = {log->power[index], log->inlet[index], log->outlet[index], log->seconds, log->train_seconds, mean_power / fmax(mean_rise, 0.5)};

    /*                  heat capacity, conductance, stream capacity (all / scale), delay, offset */
    const double low[5] = {1, 0.03, 0.2, 0, -3};
    const double high[5] = {2000, 20, 5, 60, 3};
    const double heat_capacity_guesses[2] = {20, 60};     /* try two starting guesses and keep the better */

    double best[5], best_error = INFINITY;
    for (int g = 0; g < 2; g++) {
        double numbers[5] = {heat_capacity_guesses[g], 1, 1, 3, 0};
        double error = search(cabinet_errors, &data, numbers, low, high, 5, log->train_seconds);
        if (error < best_error) {
            best_error = error;
            memcpy(best, numbers, sizeof numbers);
        }
    }
    *fit_error = best_error;
    return (Cabinet){best[0] * data.scale, best[1] * data.scale, best[2] * data.scale, best[3], best[4]};
}


/* ======================================== Stage 2: the heat exchanger ========================================
 * Inputs: modelled return water, measured building-water temperature and flow. Outputs: supply and building-water return. */

typedef struct {
    double conductance;         /* UA_hx [W/K] */
    double delay;               /* hx_delay [s] */
    double supply_smoothing;    /* tau_s [s] */
    double return_smoothing;    /* tau_p [s] */
    double kappa;               /* correction on the building water's temperature rise */
    double return_offset;       /* b_p [°C] */
} Exchanger;

/* Counterflow heat exchanger (effectiveness-NTU): heat moved per 1 °C between the two incoming streams [W/K]. */
static double exchanger_capacity(double conductance, double cabinet_stream, double building_stream)
{
    double smaller = fmin(cabinet_stream, building_stream);
    double ratio = smaller / fmax(cabinet_stream, building_stream);
    double x = exp(-conductance / smaller * (1 - ratio));
    return (1 - x) / (1 - ratio * x) * smaller;
}

typedef struct {
    const double *return_water, *facility_in, *facility_flow, *supply, *facility_out;
    int seconds, train_seconds;
    double total_stream;        /* stream capacity of all the cabinets together [W/K] */
} ExchangerData;

/* numbers: conductance / total_stream, delay, supply smoothing, return smoothing, kappa, return offset */
static void simulate_exchanger(const ExchangerData *data, const double *numbers, double *supply, double *facility_out)
{
    double keep_supply = exp(-1 / numbers[2]);
    double keep_return = exp(-1 / numbers[3]);
    double smoothed_supply = 0, smoothed_return = 0;

    for (int k = 0; k < data->seconds; k++) {
        double water_back = value_before(data->return_water, data->seconds, k, numbers[1]);
        double building_stream = data->facility_flow[k] * WATTS_PER_KELVIN_PER_GPM;
        double heat = exchanger_capacity(numbers[0] * data->total_stream, data->total_stream, building_stream) * (water_back - data->facility_in[k]);
        double supply_target = water_back - heat / data->total_stream;
        double return_target = data->facility_in[k] + numbers[4] * heat / building_stream;

        if (k == 0) {
            smoothed_supply = supply_target;
            smoothed_return = return_target;
        }
        supply[k] = smoothed_supply;
        facility_out[k] = smoothed_return + numbers[5];
        smoothed_supply = supply_target + (smoothed_supply - supply_target) * keep_supply;
        smoothed_return = return_target + (smoothed_return - return_target) * keep_return;
    }
}

static void exchanger_errors(const double *numbers, double *errors, void *model_data)
{
    ExchangerData *data = model_data;
    static double *supply = NULL, *facility_out = NULL;
    static int capacity = 0;
    if (data->seconds > capacity) {
        free(supply);
        free(facility_out);
        supply = new_array(data->seconds);
        facility_out = new_array(data->seconds);
        capacity = data->seconds;
    }
    simulate_exchanger(data, numbers, supply, facility_out);
    for (int k = 0; k < data->train_seconds; k++) {
        errors[k] = supply[k] - data->supply[k];
        errors[k + data->train_seconds] = facility_out[k] - data->facility_out[k];
    }
}


/* ======================================== Stage 3: the valve ========================================
 * Input: measured supply temperature. Output: building-water flow. */

typedef struct {
    double target;              /* T_set [°C] */
    double fast_gain;           /* Kp [gpm per °C] */
    double slow_gain;           /* Ki [gpm per °C per second] */
    double sensor_delay;        /* [s] */
    double flow_min, flow_max;  /* the lowest and highest flow in the log [gpm] */
} Valve;

typedef struct {
    const double *supply, *flow;
    int seconds, train_seconds;
    double flow_min, flow_max;
} ValveData;

/* numbers: target, fast gain, slow gain, sensor delay */
static void simulate_valve(const ValveData *data, const double *numbers, double *flow)
{
    double previous_error = value_before(data->supply, data->seconds, 0, numbers[3]) - numbers[0];
    flow[0] = data->flow[0];
    for (int k = 1; k < data->seconds; k++) {
        double error = value_before(data->supply, data->seconds, k, numbers[3]) - numbers[0];
        double change = numbers[1] * (error - previous_error) + numbers[2] * error;
        flow[k] = clamp(flow[k - 1] + change, data->flow_min, data->flow_max);
        previous_error = error;
    }
}

static void valve_errors(const double *numbers, double *errors, void *model_data)
{
    ValveData *data = model_data;
    static double *modelled = NULL;
    static int capacity = 0;
    if (data->seconds > capacity) {
        free(modelled);
        modelled = new_array(data->seconds);
        capacity = data->seconds;
    }
    simulate_valve(data, numbers, modelled);
    for (int k = 0; k < data->train_seconds; k++)
        errors[k] = modelled[k] - data->flow[k];
}

static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const double *values, int count)
{
    double *sorted = new_array(count);
    memcpy(sorted, values, sizeof(double) * (size_t)count);
    qsort(sorted, (size_t)count, sizeof(double), compare_doubles);
    double middle = sorted[count / 2];
    free(sorted);
    return middle;
}


/* ======================================== Final test: the whole model running by itself ========================================
 * The only inputs are power and building-water temperature. Everything else is the model's own prediction. */

static void simulate_whole_group(const GroupLog *log, const Cabinet *cabinets, const Exchanger *exchanger, const Valve *valve,
                                 double *supply_out, double *facility_out, double *flow_out, double **outlet_out)
{
    enum { RING = 512 };
    int count = log->cabinet_count;
    double effectiveness[MAX_CABINETS], hardware[MAX_CABINETS], total_stream = 0;
    int inlet_delay[MAX_CABINETS];
    double return_sum = 0;

    for (int i = 0; i < count; i++) {
        const Cabinet *cabinet = &cabinets[i];
        effectiveness[i] = 1 - exp(-cabinet->conductance / cabinet->stream_capacity);
        hardware[i] = log->supply[0] + (log->outlet[i][0] - cabinet->sensor_offset - log->supply[0]) / effectiveness[i];
        total_stream += cabinet->stream_capacity;
        inlet_delay[i] = (int)lround(cabinet->inlet_delay);
        return_sum += cabinet->stream_capacity * (log->outlet[i][0] - cabinet->sensor_offset);
    }

    int exchanger_delay = (int)lround(exchanger->delay);
    int sensor_delay = (int)lround(valve->sensor_delay);
    double supply_history[RING], return_history[RING];
    for (int i = 0; i < RING; i++) {
        supply_history[i] = log->supply[0];
        return_history[i] = return_sum / total_stream;
    }

    int newest = 0;
    double supply = log->supply[0];
    double facility_return = log->facility_out[0] - exchanger->return_offset;
    double flow = log->facility_flow[0];
    double previous_error = log->supply[0] - valve->target;
    double keep_supply = exp(-1 / exchanger->supply_smoothing);
    double keep_return = exp(-1 / exchanger->return_smoothing);

    for (int k = 0; k < log->seconds; k++) {
        /* The cabinets */
        double outlet_sum = 0;
        for (int i = 0; i < count; i++) {
            const Cabinet *cabinet = &cabinets[i];
            double water_in = supply_history[(newest - inlet_delay[i] + RING) % RING];
            double water_out = water_in + effectiveness[i] * (hardware[i] - water_in);
            outlet_out[i][k] = water_out + cabinet->sensor_offset;

            double settling_point = water_in + log->power[i][k] / (effectiveness[i] * cabinet->stream_capacity);
            hardware[i] = settling_point + (hardware[i] - settling_point) * exp(-effectiveness[i] * cabinet->stream_capacity / cabinet->heat_capacity);
            outlet_sum += cabinet->stream_capacity * water_out;
        }
        supply_out[k] = supply_history[newest];
        flow_out[k] = flow;
        facility_out[k] = facility_return + exchanger->return_offset;

        /* The heat exchanger */
        int next = (newest + 1) % RING;
        return_history[next] = outlet_sum / total_stream;
        double water_back = return_history[(next - exchanger_delay + RING) % RING];
        double building_stream = flow * WATTS_PER_KELVIN_PER_GPM;
        double heat = exchanger_capacity(exchanger->conductance, total_stream, building_stream) * (water_back - log->facility_in[k]);
        double supply_target = water_back - heat / total_stream;
        double return_target = log->facility_in[k] + exchanger->kappa * heat / building_stream;
        supply = supply_target + (supply - supply_target) * keep_supply;
        facility_return = return_target + (facility_return - return_target) * keep_return;
        supply_history[next] = supply;
        newest = next;

        /* The valve */
        double error = supply_history[(newest - sensor_delay + RING) % RING] - valve->target;
        flow = clamp(flow + valve->fast_gain * (error - previous_error) + valve->slow_gain * error, valve->flow_min, valve->flow_max);
        previous_error = error;
    }
}

/* Things that make the fitted numbers untrustworthy, in plain words. */
static void print_warnings(const GroupLog *log, const Exchanger *exchanger, const Valve *valve, double worst_error)
{
    double total_power = 0, total_heat = 0, lowest = INFINITY, highest = 0;
    for (int k = 0; k < log->seconds; k++) {
        double power_now = 0;
        for (int i = 0; i < log->cabinet_count; i++) power_now += log->power[i][k] / log->heat_fraction;
        lowest = fmin(lowest, power_now);
        highest = fmax(highest, power_now);
        total_power += power_now;
        total_heat += log->facility_flow[k] * WATTS_PER_KELVIN_PER_GPM * (log->facility_out[k] - log->facility_in[k]);
    }
    double heat_ratio = total_heat / total_power;

    printf("\nHeat in the building water / electrical power: %.2f\n\nCheck these:\n", heat_ratio);
    if (highest - lowest < 0.3 * highest)
        printf("  - Power only moved %.0f %% of its peak. Use a run with a big job start or stop, or the time-response numbers are guesses.\n", (highest - lowest) / highest * 100);
    if (log->seconds < 1200)
        printf("  - The run is only %.0f minutes. Aim for at least 30, so the system can settle before and after a step.\n", log->seconds / 60.0);
    if (valve->flow_max - valve->flow_min < 0.1 * median(log->facility_flow, log->seconds))
        printf("  - Building-water flow barely moved, so the valve controller can't be estimated from this run.\n");
    if (exchanger->kappa < 0.65 || exchanger->kappa > 1.35)
        printf("  - kappa = %.2f is far from 1: check the flow units (gpm expected) and building-water sensors.\n", exchanger->kappa);
    if (fabs(heat_ratio - log->heat_fraction) > 0.07)
        printf("  - Heat in the building water is %.2f x the electrical power, but the model assumes %.2f. If some heat leaves by air, rerun with heat_fraction %.2f. Otherwise check the flow units and sensors.\n", heat_ratio, log->heat_fraction, heat_ratio);
    if (worst_error > 1)
        printf("  - Running on its own, the model is off by up to %.1f °C (RMS), so it doesn't describe this machine well yet. Fix the warnings above first.\n", worst_error);
    printf("  - Flow limits (%.0f-%.0f gpm) are only the lowest and highest flow in this run, not known hardware limits.\n", valve->flow_min, valve->flow_max);
}

static void write_cooling_constants(const char *name, const GroupLog *log, const Cabinet *cabinets, const Exchanger *exchanger, const Valve *valve)
{
    char filename[256];
    snprintf(filename, sizeof filename, "cooling_%s.h", name);
    FILE *file = fopen(filename, "w");
    if (!file) fail("can't write", filename);

    int count = log->cabinet_count;
    int longest_delay = (int)fmax((int)lround(valve->sensor_delay), (int)lround(exchanger->delay));
    for (int i = 0; i < count; i++) longest_delay = (int)fmax(longest_delay, (int)lround(cabinets[i].inlet_delay));

    fprintf(file, "/* Cooling-model constants for %s, written by fit_cooling_model. See predict_cooling.c for what each one means. */\n", name);
    fprintf(file, "#define N_CAB %d\n#define HISTORY %d\n", count, longest_delay + 1);

    fprintf(file, "static const double Ch[N_CAB] = {");
    for (int i = 0; i < count; i++) fprintf(file, "%.6g%s", cabinets[i].heat_capacity, i < count - 1 ? ", " : "");
    fprintf(file, "};\nstatic const double UA[N_CAB] = {");
    for (int i = 0; i < count; i++) fprintf(file, "%.6g%s", cabinets[i].conductance, i < count - 1 ? ", " : "");
    fprintf(file, "};\nstatic const double mcp[N_CAB] = {");
    for (int i = 0; i < count; i++) fprintf(file, "%.6g%s", cabinets[i].stream_capacity, i < count - 1 ? ", " : "");
    fprintf(file, "};\nstatic const int inlet_delay[N_CAB] = {");
    for (int i = 0; i < count; i++) fprintf(file, "%d%s", (int)lround(cabinets[i].inlet_delay), i < count - 1 ? ", " : "");
    fprintf(file, "};\nstatic const double offset[N_CAB] = {");
    for (int i = 0; i < count; i++) fprintf(file, "%.6g%s", cabinets[i].sensor_offset, i < count - 1 ? ", " : "");
    fprintf(file, "};\n");

    fprintf(file, "static const double UA_hx = %.6g, tau_s = %.6g, tau_p = %.6g, kappa = %.6g, b_p = %.6g;\n", exchanger->conductance, exchanger->supply_smoothing, exchanger->return_smoothing, exchanger->kappa, exchanger->return_offset);
    fprintf(file, "static const int hx_delay = %d, sensor_delay = %d;\n", (int)lround(exchanger->delay), (int)lround(valve->sensor_delay));
    fprintf(file, "static const double T_set = %.6g, Kp = %.6g, Ki = %.6g, flow_min = %.6g, flow_max = %.6g;\n", valve->target, valve->fast_gain, valve->slow_gain, valve->flow_min, valve->flow_max);
    fprintf(file, "static const double heat_to_water = %.6g;\nstatic const double GPM = %.6g;\n", log->heat_fraction, WATTS_PER_KELVIN_PER_GPM);
    fclose(file);

    printf("\nWrote %s\nBuild with:  cc -DCOOLING_CONSTANTS='\"%s\"' predict_cooling.c yourcode.c -lm\n", filename, filename);
}

static int fit_cooling_group(const char *name, const char *path, double train_seconds, double heat_fraction)
{
    GroupLog log = load_group_log(path, train_seconds, heat_fraction);
    int seconds = log.seconds, train = log.train_seconds, count = log.cabinet_count;
    printf("%s: %d cabinets, %.0f minutes; searching on the first %d s, testing on the rest\n\n", name, count, seconds / 60.0, train);

    /* ---------- Stage 1 ---------- */
    printf("Stage 1: each cabinet (power and inlet -> outlet)\n");
    Cabinet cabinets[MAX_CABINETS];
    double *modelled = new_array(seconds);
    double total_stream = 0;
    for (int i = 0; i < count; i++) {
        double fit_error;
        cabinets[i] = fit_cabinet(&log, i, &fit_error);
        simulate_cabinet(log.power[i], log.inlet[i], seconds, &cabinets[i], modelled);
        double effectiveness = 1 - exp(-cabinets[i].conductance / cabinets[i].stream_capacity);
        printf("  cabinet %d: Ch %.2f MJ/K, UA %.1f kW/K, mcp %.1f kW/K, delay %.1f s, offset %+.2f °C, time constant %.0f s | error %.2f / %.2f °C (fit / test)\n",
               i + 1, cabinets[i].heat_capacity / 1e6, cabinets[i].conductance / 1e3, cabinets[i].stream_capacity / 1e3, cabinets[i].inlet_delay, cabinets[i].sensor_offset,
               cabinets[i].heat_capacity / (effectiveness * cabinets[i].stream_capacity), fit_error, rms_difference(modelled, log.outlet[i], train, seconds));
        total_stream += cabinets[i].stream_capacity;
    }

    /* ---------- Stage 2 ---------- */
    printf("\nStage 2: heat exchanger (return water and building water -> supply and building-water return)\n");
    double *return_water = new_array(seconds);          /* the cabinets' outlets mixed together, driven by the measured supply */
    for (int i = 0; i < count; i++) {
        Cabinet without_offset = cabinets[i];
        without_offset.sensor_offset = 0;
        simulate_cabinet(log.power[i], log.supply, seconds, &without_offset, modelled);
        for (int k = 0; k < seconds; k++)
            return_water[k] += without_offset.stream_capacity * modelled[k] / total_stream;
    }
    ExchangerData exchanger_data = {return_water, log.facility_in, log.facility_flow, log.supply, log.facility_out, seconds, train, total_stream};
    /*                       conductance / total_stream, delay, supply smoothing, return smoothing, kappa, offset */
    double exchanger_numbers[6] = {3.5, 10, 10, 10, 0.9, 0};
    const double exchanger_low[6] = {0.1, 0, 0.5, 0.5, 0.6, -5};
    const double exchanger_high[6] = {60, 90, 300, 300, 1.4, 5};
    search(exchanger_errors, &exchanger_data, exchanger_numbers, exchanger_low, exchanger_high, 6, 2 * train);
    Exchanger exchanger = {exchanger_numbers[0] * total_stream, exchanger_numbers[1], exchanger_numbers[2], exchanger_numbers[3], exchanger_numbers[4], exchanger_numbers[5]};

    double *modelled_supply = new_array(seconds), *modelled_return = new_array(seconds);
    simulate_exchanger(&exchanger_data, exchanger_numbers, modelled_supply, modelled_return);
    printf("  UA_hx %.0f kW/K, delay %.1f s, tau_s %.1f s, tau_p %.1f s, kappa %.3f, b_p %+.2f °C | supply error %.2f / %.2f, return error %.2f / %.2f °C\n",
           exchanger.conductance / 1e3, exchanger.delay, exchanger.supply_smoothing, exchanger.return_smoothing, exchanger.kappa, exchanger.return_offset,
           rms_difference(modelled_supply, log.supply, 0, train), rms_difference(modelled_supply, log.supply, train, seconds),
           rms_difference(modelled_return, log.facility_out, 0, train), rms_difference(modelled_return, log.facility_out, train, seconds));

    /* ---------- Stage 3 ---------- */
    printf("\nStage 3: valve controller (supply temperature -> flow)\n");
    double flow_min = log.facility_flow[0], flow_max = log.facility_flow[0], supply_min = log.supply[0], supply_max = log.supply[0];
    for (int k = 0; k < seconds; k++) {
        flow_min = fmin(flow_min, log.facility_flow[k]);
        flow_max = fmax(flow_max, log.facility_flow[k]);
        supply_min = fmin(supply_min, log.supply[k]);
        supply_max = fmax(supply_max, log.supply[k]);
    }
    ValveData valve_data = {log.supply, log.facility_flow, seconds, train, flow_min, flow_max};
    /*                       target, fast gain, slow gain, sensor delay */
    const double valve_low[4] = {supply_min - 1, 0, 0, 0};
    const double valve_high[4] = {supply_max + 1, 50, 5, 120};
    const double delay_guesses[3] = {10, 40, 80};         /* the delay is the hard one, so try three starting guesses */
    double best[4], best_error = INFINITY;
    for (int g = 0; g < 3; g++) {
        double numbers[4] = {median(log.supply, seconds), 2, 0.1, delay_guesses[g]};
        double error = search(valve_errors, &valve_data, numbers, valve_low, valve_high, 4, train);
        if (error < best_error) {
            best_error = error;
            memcpy(best, numbers, sizeof numbers);
        }
    }
    Valve valve = {best[0], best[1], best[2], best[3], flow_min, flow_max};
    simulate_valve(&valve_data, best, modelled);
    printf("  target %.2f °C, Kp %.2f, Ki %.3f, delay %.0f s, flow %.0f-%.0f gpm | error %.1f / %.1f gpm\n",
           valve.target, valve.fast_gain, valve.slow_gain, valve.sensor_delay, flow_min, flow_max, best_error, rms_difference(modelled, log.facility_flow, train, seconds));

    /* ---------- The final test ---------- */
    double *whole_supply = new_array(seconds), *whole_return = new_array(seconds), *whole_flow = new_array(seconds);
    double *whole_outlet[MAX_CABINETS];
    for (int i = 0; i < count; i++) whole_outlet[i] = new_array(seconds);
    simulate_whole_group(&log, cabinets, &exchanger, &valve, whole_supply, whole_return, whole_flow, whole_outlet);

    printf("\nFinal test: the whole model on its own, from power and building-water temperature only (fit / test)\n");
    printf("  %-22s %5.2f / %5.2f °C\n", "supply", rms_difference(whole_supply, log.supply, 0, train), rms_difference(whole_supply, log.supply, train, seconds));
    printf("  %-22s %5.2f / %5.2f °C\n", "building-water return", rms_difference(whole_return, log.facility_out, 0, train), rms_difference(whole_return, log.facility_out, train, seconds));
    printf("  %-22s %5.2f / %5.2f gpm\n", "valve flow", rms_difference(whole_flow, log.facility_flow, 0, train), rms_difference(whole_flow, log.facility_flow, train, seconds));

    double worst_error = fmax(rms_difference(whole_supply, log.supply, 0, seconds), rms_difference(whole_return, log.facility_out, 0, seconds));
    for (int i = 0; i < count; i++) {
        printf("  outlet %-15d %5.2f / %5.2f °C\n", i + 1, rms_difference(whole_outlet[i], log.outlet[i], 0, train), rms_difference(whole_outlet[i], log.outlet[i], train, seconds));
        worst_error = fmax(worst_error, rms_difference(whole_outlet[i], log.outlet[i], 0, seconds));
    }

    print_warnings(&log, &exchanger, &valve, worst_error);
    write_cooling_constants(name, &log, cabinets, &exchanger, &valve);
    return 0;
}


/* ======================================== One GPU ========================================
 * Three layers in a row: die, package, cold plate, then the coolant. Inputs: power. Output: junction temperature.
 * The search works on the logarithms of R and C, so they can't go negative and very different sizes behave alike.
 * numbers: log R1, log R2, log R3, log C1, log C2, log C3, coolant temperature. */

typedef struct {
    const double *time, *power, *junction;
    int row_count;
    int reading_count;
} GpuData;

/* The three layers affect each other. Rewrite them as three independent exponential decays (the eigenvectors of a symmetric
 * 3 x 3 matrix, found by Jacobi rotations), so each decay can be updated exactly for any time gap. */
static void find_layer_modes(const double R[3], const double C[3], double decay_rate[3], double modes[3][3])
{
    double link12 = 1 / (R[0] * sqrt(C[0] * C[1]));
    double link23 = 1 / (R[1] * sqrt(C[1] * C[2]));
    double a[3][3] = {
        {-1 / (R[0] * C[0]),  link12,                                   0},
        {link12,              -1 / (R[0] * C[1]) - 1 / (R[1] * C[1]),   link23},
        {0,                   link23,                                   -1 / (R[1] * C[2]) - 1 / (R[2] * C[2])},
    };
    double q[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    for (int sweep = 0; sweep < 60; sweep++) {
        for (int p = 0; p < 2; p++) {
            for (int k = p + 1; k < 3; k++) {
                if (fabs(a[p][k]) < 1e-14) continue;
                double angle = 0.5 * atan2(2 * a[p][k], a[k][k] - a[p][p]);
                double s = sin(angle), c = cos(angle);
                for (int i = 0; i < 3; i++) {
                    double u = a[i][p], w = a[i][k];
                    a[i][p] = c * u - s * w;
                    a[i][k] = s * u + c * w;
                }
                for (int i = 0; i < 3; i++) {
                    double u = a[p][i], w = a[k][i];
                    a[p][i] = c * u - s * w;
                    a[k][i] = s * u + c * w;
                }
                for (int i = 0; i < 3; i++) {
                    double u = q[i][p], w = q[i][k];
                    q[i][p] = c * u - s * w;
                    q[i][k] = s * u + c * w;
                }
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        decay_rate[i] = a[i][i];
        for (int j = 0; j < 3; j++) modes[i][j] = q[i][j];
    }
}

static void simulate_gpu(const GpuData *data, const double *numbers, double *junction)
{
    double R[3], C[3];
    for (int i = 0; i < 3; i++) {
        R[i] = exp(numbers[i]);
        C[i] = exp(numbers[i + 3]);
    }
    double coolant = numbers[6];

    double decay_rate[3], modes[3][3];
    find_layer_modes(R, C, decay_rate, modes);

    /* Start settled at the mean power of the first second */
    double start_power = 0;
    int start_rows = 0;
    for (int i = 0; i < data->row_count && data->time[i] < data->time[0] + 1; i++) {
        start_power += data->power[i];
        start_rows++;
    }
    start_power /= start_rows;

    double layer[3] = {coolant + start_power * (R[0] + R[1] + R[2]), coolant + start_power * (R[1] + R[2]), coolant + start_power * R[2]};
    double state[3];
    for (int m = 0; m < 3; m++) {
        state[m] = 0;
        for (int i = 0; i < 3; i++) state[m] += modes[i][m] * layer[i] * sqrt(C[i]);
    }

    double coolant_input = coolant / (R[2] * sqrt(C[2]));
    for (int i = 0; i < data->row_count; i++) {
        junction[i] = (modes[0][0] * state[0] + modes[0][1] * state[1] + modes[0][2] * state[2]) / sqrt(C[0]);
        if (i + 1 == data->row_count) break;

        double gap = data->time[i + 1] - data->time[i];
        double power_input = data->power[i] / sqrt(C[0]);
        for (int m = 0; m < 3; m++) {
            double settling_point = -(modes[0][m] * power_input + modes[2][m] * coolant_input) / decay_rate[m];
            state[m] = settling_point + (state[m] - settling_point) * exp(decay_rate[m] * gap);
        }
    }
}

static void gpu_errors(const double *numbers, double *errors, void *model_data)
{
    GpuData *data = model_data;
    static double *modelled = NULL;
    static int capacity = 0;
    if (data->row_count > capacity) {
        free(modelled);
        modelled = new_array(data->row_count);
        capacity = data->row_count;
    }
    simulate_gpu(data, numbers, modelled);
    int next_error = 0;
    for (int i = 0; i < data->row_count; i++)
        if (isfinite(data->junction[i])) errors[next_error++] = modelled[i] - data->junction[i];
}

static void write_gpu_constants(const char *name, int gpu_count, double R[][3], double C[][3])
{
    char filename[256];
    snprintf(filename, sizeof filename, "gpu_%s.h", name);
    FILE *file = fopen(filename, "w");
    if (!file) fail("can't write", filename);

    fprintf(file, "/* GPU-model constants for %s, written by fit_cooling_model from: power trace + junction sensor. Don't edit by hand.\n", name);
    fprintf(file, " * See predict_gpu.c for what these mean. */\n#define N_GPU %d\n\n", gpu_count);

    fprintf(file, "/* Thermal resistance [K/W]:             die,     package,  cold plate */\nstatic const double R[N_GPU][3] = {\n");
    for (int g = 0; g < gpu_count; g++)
        fprintf(file, "    {%.6g, %.6g, %.6g},   /* GPU %d: %.4f K/W in total */\n", R[g][0], R[g][1], R[g][2], g, R[g][0] + R[g][1] + R[g][2]);

    fprintf(file, "};\n\n/* Heat capacity [J/K]:                   die,     package,  cold plate */\nstatic const double C[N_GPU][3] = {\n");
    for (int g = 0; g < gpu_count; g++)
        fprintf(file, "    {%.6g, %.6g, %.6g},   /* GPU %d: reacts in about %.3f s, %.1f s, %.0f s */\n", C[g][0], C[g][1], C[g][2], g, R[g][0] * C[g][0], R[g][1] * C[g][1], R[g][2] * C[g][2]);
    fprintf(file, "};\n");
    fclose(file);

    printf("\nWrote %s\nBuild with:  cc -DGPU_CONSTANTS='\"%s\"' predict_gpu.c yourcode.c -lm\n", filename, filename);
}

static int fit_gpus(const char *name, int file_count, char **files)
{
    if (file_count > MAX_GPUS) fail("too many GPU files", NULL);
    double all_R[MAX_GPUS][3], all_C[MAX_GPUS][3];
    printf("%s: %d GPU(s), from power and junction temperature only\n\n", name, file_count);

    for (int g = 0; g < file_count; g++) {
        Table table = read_csv(files[g]);
        GpuData data = {find_column(&table, "time_s"), find_column(&table, "power_W"), find_column(&table, "junction_C"), table.row_count, 0};
        if (!data.time || !data.power || !data.junction) fail("need columns time_s, power_W, junction_C in", files[g]);

        double lowest_power = INFINITY, highest_power = 0, lowest_junction = INFINITY;
        for (int i = 0; i < table.row_count; i++) {
            if (isfinite(data.junction[i])) {
                data.reading_count++;
                lowest_junction = fmin(lowest_junction, data.junction[i]);
            }
            lowest_power = fmin(lowest_power, data.power[i]);
            highest_power = fmax(highest_power, data.power[i]);
        }
        if (data.reading_count < 10) fail("too few junction readings in", files[g]);

        /*                  R1, R2, R3 [K/W],                 C1, C2, C3 [J/K],                  coolant [°C] */
        const double low[7] = {log(1e-4), log(1e-4), log(1e-4), log(0.05), log(0.05), log(0.05), 0};
        const double high[7] = {log(0.5), log(0.5), log(0.5), log(1e5), log(1e5), log(1e5), 100};
        double numbers[7] = {log(0.015), log(0.02), log(0.015), log(1.5), log(60), log(1000), fmax(lowest_junction - 5, 1)};
        double error = search(gpu_errors, &data, numbers, low, high, 7, data.reading_count);

        double *R = all_R[g], *C = all_C[g];
        for (int i = 0; i < 3; i++) {
            R[i] = exp(numbers[i]);
            C[i] = exp(numbers[i + 3]);
        }
        printf("GPU %d (%s): R %.4f, %.4f, %.4f K/W (total %.4f); C %.2f, %.1f, %.0f J/K; time scales %.3f, %.2f, %.1f s; coolant %.1f °C | error %.2f °C from %d readings\n",
               g, files[g], R[0], R[1], R[2], R[0] + R[1] + R[2], C[0], C[1], C[2], R[0] * C[0], R[1] * C[1], R[2] * C[2], numbers[6], error, data.reading_count);

        if (highest_power - lowest_power < 0.3 * highest_power)
            printf("  - Power only moved %.0f %% of its peak. Use a run with a big job start, or the numbers are guesses.\n", (highest_power - lowest_power) / highest_power * 100);
        if (table.row_count && data.time[table.row_count - 1] - data.time[0] < 60)
            printf("  - The run is under a minute, too short to see the slowest layer.\n");
    }

    write_gpu_constants(name, file_count, all_R, all_C);
    return 0;
}


int main(int argc, char **argv)
{
    if (argc >= 4 && !strcmp(argv[1], "group")) {
        double train_seconds = argc > 4 ? atof(argv[4]) : 0;
        double heat_fraction = argc > 5 ? atof(argv[5]) : 1.0;
        return fit_cooling_group(argv[2], argv[3], train_seconds, heat_fraction);
    }
    if (argc >= 4 && !strcmp(argv[1], "gpu"))
        return fit_gpus(argv[2], argc - 3, argv + 3);

    fprintf(stderr, "usage:\n  %s gpu NAME gpu0.csv [gpu1.csv ...]\n  %s group NAME log.csv [train_seconds] [heat_fraction]\n", argv[0], argv[0]);
    return 1;
}
