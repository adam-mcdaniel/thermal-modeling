/* Cooling-model constants for frontier, written by fit_cooling_model. See predict_cooling.c for what each one means. */
#define N_CAB 3
#define HISTORY 69
static const double Ch[N_CAB] = {1.61807e+06, 1.08262e+06, 763192};
static const double UA[N_CAB] = {35222.6, 20265.9, 15145.3};
static const double mcp[N_CAB] = {29758, 28872.9, 26950.5};
static const int inlet_delay[N_CAB] = {2, 0, 0};
static const double offset[N_CAB] = {0.269659, 0.624875, 0.621334};
static const double UA_hx = 335996, tau_s = 0.5, tau_p = 7.8589, kappa = 0.884233, b_p = 0.94092;
static const int hx_delay = 13, sensor_delay = 68;
static const double T_set = 28.6401, Kp = 5.57282, Ki = 0.122777, flow_min = 95.8, flow_max = 208.6;
static const double heat_to_water = 1;
static const double GPM = 263.189;
