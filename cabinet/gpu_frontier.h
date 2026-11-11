/* GPU-model constants for frontier, written by fit_cooling_model from: power trace + junction sensor. Don't edit by hand.
 * See predict_gpu.c for what these mean. */
#define N_GPU 4

/* Thermal resistance [K/W]:             die,     package,  cold plate */
static const double R[N_GPU][3] = {
    {0.0158626, 0.0235973, 0.015849},   /* GPU 0: 0.0553 K/W in total */
    {0.0170496, 0.0172841, 0.00965969},   /* GPU 1: 0.0440 K/W in total */
    {0.0126708, 0.0233166, 0.0180974},   /* GPU 2: 0.0541 K/W in total */
    {0.0156205, 0.0187679, 0.00746249},   /* GPU 3: 0.0419 K/W in total */
};

/* Heat capacity [J/K]:                   die,     package,  cold plate */
static const double C[N_GPU][3] = {
    {2.05164, 76.0569, 1060.49},   /* GPU 0: reacts in about 0.033 s, 1.8 s, 17 s */
    {1.92171, 69.7358, 1388.72},   /* GPU 1: reacts in about 0.033 s, 1.2 s, 13 s */
    {1.52889, 50.6139, 749.705},   /* GPU 2: reacts in about 0.019 s, 1.2 s, 14 s */
    {2.24498, 64.6036, 2040.18},   /* GPU 3: reacts in about 0.035 s, 1.2 s, 15 s */
};
