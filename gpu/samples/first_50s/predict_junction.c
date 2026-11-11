/*
 * Predict one GPU's junction temperature from a power trace.
 *
 *     cc -O2 predict_junction.c -lm -o predict_junction
 *     ./predict_junction GPU samples.txt > predicted.csv
 *
 * Each line of samples.txt: power_W timestamp_s. Anything after those two columns is ignored.
 */
#include <stdio.h>
#include <stdlib.h>
#include "gpu_model.h"
#include "gpu_fit.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s GPU samples.txt\n", argv[0]);
        return 1;
    }
    const GpuFit *fit = &GPU_FITS[atoi(argv[1])];
    FILE *file = fopen(argv[2], "r");
    if (!file) {
        fprintf(stderr, "can't open %s\n", argv[2]);
        return 1;
    }

    char line[256];
    double power, time;
    GpuState state;
    int started = 0;

    printf("time_s,power_W,junction_C\n");
    while (fgets(line, sizeof line, file)) {
        if (sscanf(line, "%lf %lf", &power, &time) != 2) continue;
        if (!started) {
            gpu_start(&state, fit, power, time);
            started = 1;
        }
        printf("%.6f,%.3f,%.4f\n", time, power, gpu_junction(&state, fit, power, time));
    }
    fclose(file);
    return 0;
}
