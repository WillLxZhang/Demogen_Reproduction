#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

demo=${1:-}
seed=${2:-}
ckpt_tag=${3:-}
control_steps=${4:-1}

if [ -z "${demo}" ] || [ -z "${seed}" ] || [ -z "${ckpt_tag}" ]; then
    echo "Usage: bash eval_dp3_panda_explicit.sh <demo> <seed> <ckpt_tag> [control_steps]"
    echo "Example: bash eval_dp3_panda_explicit.sh lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_9 0 179 1"
    exit 1
fi

algo=dp3
task=panda

data_root="${repo_root}/data"
run_name_override=${RUN_NAME_OVERRIDE:-}
run_name_suffix=${RUN_NAME_SUFFIX:-}

generated_zarr_path=${data_root}/datasets/generated/${demo}.zarr
legacy_zarr_path=${data_root}/datasets/demogen/${demo}.zarr

if [ -e "${generated_zarr_path}" ]; then
    zarr_path=${generated_zarr_path}
elif [ -e "${legacy_zarr_path}" ]; then
    zarr_path=${legacy_zarr_path}
else
    echo "[ERROR] Dataset not found:"
    echo "  ${generated_zarr_path}"
    echo "  ${legacy_zarr_path}"
    exit 1
fi

source_dataset=${SOURCE_DATASET:-/home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_keyboard_1/1774355871_95818/demo.hdf5}
eval_episodes=${EVAL_EPISODES:-20}
save_video=${SAVE_VIDEO:-False}
n_gpu=${N_GPU:-1}
n_cpu_per_gpu=${N_CPU_PER_GPU:-2}
debug_action_print_steps=${DEBUG_ACTION_PRINT_STEPS:-0}

horizon=${HORIZON:-4}
n_obs_steps=${N_OBS_STEPS:-2}
max_supported_n_action_steps=$((horizon - n_obs_steps + 1))
if [ "${max_supported_n_action_steps}" -le 0 ]; then
    echo "[ERROR] Invalid protocol: horizon=${horizon}, n_obs_steps=${n_obs_steps} -> max_supported_n_action_steps=${max_supported_n_action_steps}"
    exit 1
fi

default_n_action_steps=${max_supported_n_action_steps}
n_action_steps=${N_ACTION_STEPS:-${default_n_action_steps}}
if [ "${n_action_steps}" -le 0 ]; then
    echo "[ERROR] n_action_steps must be positive, got ${n_action_steps}"
    exit 1
fi
if [ "${n_action_steps}" -gt "${max_supported_n_action_steps}" ]; then
    echo "[ERROR] Explicit panda protocol rejects requested n_action_steps=${n_action_steps} because horizon=${horizon}, n_obs_steps=${n_obs_steps} only support ${max_supported_n_action_steps}."
    echo "        Pick a compatible triple instead of relying on implicit clamping."
    exit 1
fi

down_dims=${DOWN_DIMS:-[128,256,384]}
encoder_output_dim=${ENCODER_OUTPUT_DIM:-64}
use_ema=${USE_EMA:-False}
dataloader_workers=${DATALOADER_WORKERS:-4}
val_dataloader_workers=${VAL_DATALOADER_WORKERS:-2}

if [ -n "${run_name_override}" ]; then
    exp_name=${run_name_override}
elif [ -n "${run_name_suffix}" ]; then
    exp_name=${demo}-${algo}-seed${seed}-${run_name_suffix}
else
    exp_name=${demo}-${algo}-seed${seed}
fi
run_dir=${data_root}/ckpts/${exp_name}

if [ ! -d "${run_dir}" ]; then
    echo "[ERROR] Run dir not found:"
    echo "  ${run_dir}"
    echo "        If training used a renamed run, pass the same RUN_NAME_SUFFIX=<name>."
    exit 1
fi

echo "[INFO] Using dataset: ${zarr_path}"
echo "[INFO] Using source dataset: ${source_dataset}"
echo "[INFO] Explicit panda eval protocol:"
echo "       checkpoint_tag=${ckpt_tag}, control_steps=${control_steps}, eval_episodes=${eval_episodes}, save_video=${save_video}"
echo "       horizon=${horizon}, n_obs_steps=${n_obs_steps}, n_action_steps=${n_action_steps}"
echo "       exp_name=${exp_name}"
echo "       run_dir=${run_dir}"
echo "       python_bin=${python_bin}"

export HYDRA_FULL_ERROR=1
cd "${script_dir}"
"${python_bin}" -W ignore eval.py --config-name=${algo}.yaml \
                         task=${task} \
                         hydra.run.dir=${run_dir} \
                         training.seed=${seed} \
                         training.device="cuda:0" \
                         exp_name=${exp_name} \
                         task.dataset.zarr_path=${zarr_path} \
                         horizon=${horizon} \
                         n_obs_steps=${n_obs_steps} \
                         n_action_steps=${n_action_steps} \
                         policy.down_dims=${down_dims} \
                         policy.encoder_output_dim=${encoder_output_dim} \
                         training.use_ema=${use_ema} \
                         dataloader.num_workers=${dataloader_workers} \
                         val_dataloader.num_workers=${val_dataloader_workers} \
                         task.env_runner.eval_episodes=${eval_episodes} \
                         eval.save_video=${save_video} \
                         eval.n_gpu=${n_gpu} \
                         eval.n_cpu_per_gpu=${n_cpu_per_gpu} \
                         eval.source_dataset=${source_dataset} \
                         ++eval.checkpoint_tag=${ckpt_tag} \
                         ++task.env_runner.n_control_steps=${control_steps} \
                         ++task.env_runner.debug_action_print_steps=${debug_action_print_steps}
