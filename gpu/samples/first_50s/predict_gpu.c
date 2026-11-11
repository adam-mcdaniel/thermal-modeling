/*
 * Run one GPU's power trace through the model.
 *
 *     cc -O2 predict_gpu.c -lm -o predict_gpu
 *     ./predict_gpu GPU samples.txt [sensor] [group] [horizon=SECONDS]
 *
 *     sensor  also use the junction readings in the samples file
 *     group   the whole cooling group runs like this GPU, so its heat warms the water. Without this, it's only this GPU, and the water stays steady.
 *     horizon also forecast this far ahead: the junction if recent power continues, and the worst case at the GPU's peak power
 *
 * Each line of samples.txt: power_W timestamp_s [junction_C]. Use nan for the junction on lines with no reading.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_model.h"
#include "gpu_fit.h"

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "usage: %s GPU samples.txt [sensor] [group] [horizon=SECONDS]\n", argv[0]);
        return 1;
    }
    const GpuFit *fit = &GPU_FITS[atoi(argv[1])];
    int use_sensor = 0;
    double group_share = 0, horizon_s = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "sensor")) use_sensor = 1;
        if (!strcmp(argv[i], "group")) group_share = 1;
        if (!strncmp(argv[i], "horizon=", 8)) horizon_s = atof(argv[i] + 8);
    }
    FILE *file = fopen(argv[2], "r");
    if (!file) {
        fprintf(stderr, "can't open %s\n", argv[2]);
        return 1;
    }

    char line[256];
    GpuState state;
    int started = 0;
    GpuForecast forecast;
    if (horizon_s > 0) gpu_forecast_start(&forecast, fit, horizon_s);

    printf("time_s,power_W,junction_C,coolant_C,water_out_C,flux_generated_W_cm2,flux_die_exit_W_cm2,flux_to_water_W_cm2%s\n", horizon_s > 0 ? ",junction_if_held_C,worst_case_C" : "");
    while (fgets(line, sizeof line, file)) {
        double power, time, junction = NAN;
        if (sscanf(line, "%lf %lf %lf", &power, &time, &junction) < 2) continue;
        if (!use_sensor) junction = NAN;
        if (!started) {
            gpu_start(&state, fit, power, group_share, time);
            started = 1;
        }
        GpuPrediction p = gpu_predict(&state, fit, power, junction, time);
        printf("%.6f,%.3f,%.4f,%.4f,%.5f,%.4f,%.4f,%.4f", time, power, p.junction_C, p.coolant_C, p.water_out_C, p.flux_generated, p.flux_die_exit, p.flux_to_water);
        if (horizon_s > 0) printf(",%.4f,%.4f", gpu_junction_after(&state, fit, &forecast, state.recent_power), gpu_worst_case(&state, fit, &forecast));
        printf("\n");
    }
    fclose(file);
    return 0;
}
