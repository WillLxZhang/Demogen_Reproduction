#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

demo=${1:-}
seed=${2:-}
ckpt_tag=${3:-}
control_steps=${4:-1}
run_name_suffix=${5:-}

if [ -z "${demo}" ] || [ -z "${seed}" ] || [ -z "${ckpt_tag}" ]; then
    echo "Usage: bash eval_panda.sh <demo> <seed> <ckpt_tag> [control_steps] [run_name_suffix]"
    echo "Example: bash eval_panda.sh lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_9 0 159 1 fresh_a"
    exit 1
fi

if [ -n "${run_name_suffix}" ]; then
    export RUN_NAME_SUFFIX="${run_name_suffix}"
    echo "[INFO] Evaluating renamed run with suffix: ${RUN_NAME_SUFFIX}"
else
    echo "[INFO] Evaluating default run name: ${demo}-dp3-seed${seed}"
fi

echo "[INFO] Live output enabled via conda run --no-capture-output"

conda run --no-capture-output -n demogen \
    "${script_dir}/eval_dp3_panda_explicit.sh" \
    "${demo}" \
    "${seed}" \
    "${ckpt_tag}" \
    "${control_steps}"
