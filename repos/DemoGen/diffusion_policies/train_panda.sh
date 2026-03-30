#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

demo=${1:-}
seed=${2:-}
run_name_suffix=${3:-}

if [ -z "${demo}" ] || [ -z "${seed}" ]; then
    echo "Usage: bash train_panda.sh <demo> <seed> [run_name_suffix]"
    echo "Example: bash train_panda.sh lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_9 0 fresh_a"
    exit 1
fi

if [ -n "${run_name_suffix}" ]; then
    export RUN_NAME_SUFFIX="${run_name_suffix}"
else
    auto_suffix="$(date +%Y%m%d_%H%M%S)"
    export RUN_NAME_SUFFIX="${auto_suffix}"
    echo "[INFO] RUN_NAME_SUFFIX not provided. Using auto suffix: ${RUN_NAME_SUFFIX}"
fi

echo "[INFO] Live output enabled via conda run --no-capture-output"
echo "[INFO] Fresh train run name suffix: ${RUN_NAME_SUFFIX}"

conda run --no-capture-output -n demogen \
    "${script_dir}/train_dp3_panda_explicit.sh" \
    "${demo}" \
    "${seed}"
