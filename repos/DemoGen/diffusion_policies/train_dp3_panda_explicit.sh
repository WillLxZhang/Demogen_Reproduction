#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

demo=${1:-}
seed=${2:-}

if [ -z "${demo}" ] || [ -z "${seed}" ]; then
    echo "Usage: bash train_dp3_panda_explicit.sh <demo> <seed>"
    echo "Example: bash train_dp3_panda_explicit.sh lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_9 0"
    exit 1
fi

algo=dp3
task=panda

data_root="${repo_root}/data"
run_name_override=${RUN_NAME_OVERRIDE:-}
run_name_suffix=${RUN_NAME_SUFFIX:-}
allow_dir_reuse=${ALLOW_DIR_REUSE:-False}

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

batch_size=${BATCH_SIZE:-4}
val_batch_size=${VAL_BATCH_SIZE:-4}
num_epochs=${NUM_EPOCHS:-160}
val_ratio=${VAL_RATIO:-0.1}
checkpoint_every=${CHECKPOINT_EVERY:-20}
val_every=${VAL_EVERY:-1}
sample_every=${SAMPLE_EVERY:-1}
save_last_ckpt=${SAVE_LAST_CKPT:-True}
save_ckpt=${SAVE_CKPT:-True}
resume=${RESUME:-False}

down_dims=${DOWN_DIMS:-[128,256,384]}
encoder_output_dim=${ENCODER_OUTPUT_DIM:-64}
use_ema=${USE_EMA:-False}
optimizer_foreach=${OPTIMIZER_FOREACH:-False}
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

if [ "${resume}" != "True" ] && [ -e "${run_dir}" ] && [ "${allow_dir_reuse}" != "True" ]; then
    echo "[ERROR] Refusing to write into existing run dir without RESUME=True:"
    echo "  ${run_dir}"
    echo "        Use RUN_NAME_SUFFIX=<new_name> to create a fresh run,"
    echo "        or set RESUME=True if you really want to continue that run."
    echo "        You can bypass this guard with ALLOW_DIR_REUSE=True, but that is not recommended."
    exit 1
fi

echo "[INFO] Using dataset: ${zarr_path}"
echo "[INFO] Explicit panda train protocol:"
echo "       horizon=${horizon}, n_obs_steps=${n_obs_steps}, n_action_steps=${n_action_steps}"
echo "       batch_size=${batch_size}, val_batch_size=${val_batch_size}, num_epochs=${num_epochs}, max_train_steps=null, val_ratio=${val_ratio}"
echo "       checkpoint_every=${checkpoint_every}, val_every=${val_every}, sample_every=${sample_every}, resume=${resume}"
echo "       exp_name=${exp_name}"
echo "       run_dir=${run_dir}"
echo "       python_bin=${python_bin}"

export HYDRA_FULL_ERROR=1
cd "${script_dir}"
"${python_bin}" -W ignore train.py --config-name=${algo}.yaml \
                          task=${task} \
                          hydra.run.dir=${run_dir} \
                          training.debug=False \
                          training.seed=${seed} \
                          training.device="cuda:0" \
                          exp_name=${exp_name} \
                          logging.mode=offline \
                          training.resume=${resume} \
                          training.num_epochs=${num_epochs} \
                          training.max_train_steps=null \
                          dataloader.batch_size=${batch_size} \
                          val_dataloader.batch_size=${val_batch_size} \
                          task.dataset.zarr_path=${zarr_path} \
                          horizon=${horizon} \
                          n_obs_steps=${n_obs_steps} \
                          n_action_steps=${n_action_steps} \
                          policy.down_dims=${down_dims} \
                          policy.encoder_output_dim=${encoder_output_dim} \
                          training.use_ema=${use_ema} \
                          +optimizer.foreach=${optimizer_foreach} \
                          dataloader.num_workers=${dataloader_workers} \
                          val_dataloader.num_workers=${val_dataloader_workers} \
                          task.dataset.val_ratio=${val_ratio} \
                          training.val_every=${val_every} \
                          training.sample_every=${sample_every} \
                          training.checkpoint_every=${checkpoint_every} \
                          checkpoint.save_last_ckpt=${save_last_ckpt} \
                          checkpoint.save_ckpt=${save_ckpt}
