/*
 * A GPU's energy counter and junction sensor, through rocm_smi, loaded at run time so nothing is needed to build.
 *
 *     Telemetry tl;
 *     if (tl_open(&tl, "0000:c1:00.0", -1) == 0)      // the GPU's PCI address, from hipDeviceGetPCIBusId
 *         tl_read(&tl, &energy_J, &junction_C);       // energy: a running total, so power = change / time
 *
 * The same counters the Score-P traces used: energy_count, and temp_current for sensor 1 (the junction).
 * On an MI250X, both GCDs share one package and one energy counter, which only the even-numbered device reports. For an odd GCD,
 * pass its partner's index as `index`.
 */
#ifndef GPU_TELEMETRY_H
#define GPU_TELEMETRY_H

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    void *library;
    uint32_t index;          /* rocm_smi's number for this GPU */
    int (*energy)(uint32_t, uint64_t *, float *, uint64_t *);
    int (*temperature)(uint32_t, uint32_t, int, int64_t *);
} Telemetry;

#define TL_JUNCTION 1        /* RSMI_TEMP_TYPE_JUNCTION */
#define TL_CURRENT 0         /* RSMI_TEMP_CURRENT */

/* rocm_smi's PCI id: domain << 32 | bus << 8 | device << 3 | function. */
static uint64_t tl_pci_id(const char *address)
{
    unsigned domain = 0, bus = 0, device = 0, function = 0;
    sscanf(address, "%x:%x:%x.%x", &domain, &bus, &device, &function);
    return ((uint64_t)domain << 32) | ((bus & 0xff) << 8) | ((device & 0x1f) << 3) | (function & 0x7);
}

/* Find the GPU with this PCI address, or use `index` if it's 0 or more. Returns 0 on success. */
static int tl_open(Telemetry *tl, const char *pci_address, int index)
{
    tl->library = dlopen("librocm_smi64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!tl->library) tl->library = dlopen("librocm_smi64.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!tl->library) return -1;
    int (*init)(uint64_t) = (int (*)(uint64_t))dlsym(tl->library, "rsmi_init");
    int (*devices)(uint32_t *) = (int (*)(uint32_t *))dlsym(tl->library, "rsmi_num_monitor_devices");
    int (*pci)(uint32_t, uint64_t *) = (int (*)(uint32_t, uint64_t *))dlsym(tl->library, "rsmi_dev_pci_id_get");
    tl->energy = (int (*)(uint32_t, uint64_t *, float *, uint64_t *))dlsym(tl->library, "rsmi_dev_energy_count_get");
    tl->temperature = (int (*)(uint32_t, uint32_t, int, int64_t *))dlsym(tl->library, "rsmi_dev_temp_metric_get");
    if (!init || !devices || !pci || !tl->energy || !tl->temperature || init(0) != 0) return -1;

    if (index >= 0) {
        tl->index = (uint32_t)index;
        return 0;
    }
    uint32_t count = 0;
    devices(&count);
    uint64_t wanted = tl_pci_id(pci_address);
    for (uint32_t i = 0; i < count; i++) {
        uint64_t id = 0;
        if (pci(i, &id) == 0 && id == wanted) {
            tl->index = i;
            return 0;
        }
    }
    return -1;
}

/* The energy used so far [J] and the junction [°C]. Either is NAN if it couldn't be read. Returns 0 if both were read. */
static int tl_read(Telemetry *tl, double *energy_J, double *junction_C)
{
    uint64_t count = 0, timestamp = 0;
    float resolution = 0;
    int64_t millidegrees = 0;
    int energy_failed = tl->energy(tl->index, &count, &resolution, &timestamp);
    int junction_failed = tl->temperature(tl->index, TL_JUNCTION, TL_CURRENT, &millidegrees);
    *energy_J = energy_failed ? NAN : count * (double)resolution * 1e-6;     /* counts of `resolution` microjoules */
    *junction_C = junction_failed ? NAN : millidegrees * 1e-3;
    return energy_failed || junction_failed;
}

#endif
