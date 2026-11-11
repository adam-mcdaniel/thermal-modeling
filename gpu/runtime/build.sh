#!/bin/sh
# Build everything here on a machine with ROCm, such as a Frontier login node (module load rocm), in one command:    ./build.sh
set -e
cd "$(dirname "$0")"
ROCM="${ROCM_PATH:-/opt/rocm}"
CC="${CC:-cc}"
"$CC" -O2 -I. thermal_stress.c -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lrocblas -lamdhip64 -lm -lpthread -ldl -o thermal_stress
"$CC" -O2 -Wno-unused-function test_core.c thermal_core.c -lm -o test_core
(cd tapir-thermal && cargo build --release)
cp tapir-thermal/target/release/libtapir_thermal.so .
echo "built thermal_stress, test_core and libtapir_thermal.so"
