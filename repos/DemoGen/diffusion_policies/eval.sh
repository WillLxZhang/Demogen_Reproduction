demo=${1}
algo=${2}
task=${3}
seed=${4}

data_root=../data
exp_name=${demo}-${algo}-seed${seed}
run_dir=${data_root}/ckpts/${exp_name}

generated_zarr_path=${data_root}/datasets/generated/${demo}.zarr
legacy_zarr_path=${data_root}/datasets/demogen/${demo}.zarr

if [ -e "${generated_zarr_path}" ]; then
    zarr_path=${generated_zarr_path}
elif [ -e "${legacy_zarr_path}" ]; then
    zarr_path=${legacy_zarr_path}
else
    zarr_path=${generated_zarr_path}
fi

if [ "${task}" = "panda" ]; then
    source_dataset_default=/home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_keyboard_1/1774355871_95818/demo.hdf5
else
    source_dataset_default=""
fi

source_dataset=${SOURCE_DATASET:-${source_dataset_default}}
eval_episodes=${EVAL_EPISODES:-20}
save_video=${SAVE_VIDEO:-False}
n_gpu=${N_GPU:-1}
n_cpu_per_gpu=${N_CPU_PER_GPU:-2}
control_steps=${CONTROL_STEPS:-5}
debug_action_print_steps=${DEBUG_ACTION_PRINT_STEPS:-0}
eval_ckpt_tag=${EVAL_CKPT_TAG:-latest}

echo "[INFO] Using dataset: ${zarr_path}"
echo "[INFO] Using source dataset: ${source_dataset}"
echo "[INFO] Eval episodes: ${eval_episodes}"
echo "[INFO] Save video: ${save_video}"
echo "[INFO] Checkpoint tag: ${eval_ckpt_tag}"
echo "[INFO] Control steps: ${control_steps}"
echo "[INFO] Debug action print steps: ${debug_action_print_steps}"

extra_args=()
if [ "${task}" = "panda" ] && [ "${algo}" = "dp3" ]; then
    down_dims=${DOWN_DIMS:-[128,256,384]}
    horizon=${HORIZON:-4}
    n_obs_steps=${N_OBS_STEPS:-2}
    requested_n_action_steps=${N_ACTION_STEPS:-4}
    max_supported_n_action_steps=$((horizon - n_obs_steps + 1))
    if [ "${max_supported_n_action_steps}" -le 0 ]; then
        echo "[ERROR] Invalid protocol: horizon=${horizon}, n_obs_steps=${n_obs_steps} -> max_supported_n_action_steps=${max_supported_n_action_steps}"
        exit 1
    fi
    n_action_steps=${requested_n_action_steps}
    if [ "${requested_n_action_steps}" -gt "${max_supported_n_action_steps}" ]; then
        echo "[WARN] Requested n_action_steps=${requested_n_action_steps} exceeds max_supported_n_action_steps=${max_supported_n_action_steps} for horizon=${horizon}, n_obs_steps=${n_obs_steps}. Clamping."
        n_action_steps=${max_supported_n_action_steps}
    fi
    encoder_output_dim=${ENCODER_OUTPUT_DIM:-64}
    use_ema=${USE_EMA:-False}
    dataloader_workers=${DATALOADER_WORKERS:-4}
    val_dataloader_workers=${VAL_DATALOADER_WORKERS:-2}

    echo "[INFO] Panda DP3 eval overrides: horizon=${horizon}, n_obs_steps=${n_obs_steps}, requested_n_action_steps=${requested_n_action_steps}, effective_n_action_steps=${n_action_steps}"

    extra_args+=(
        "horizon=${horizon}"
        "n_obs_steps=${n_obs_steps}"
        "n_action_steps=${n_action_steps}"
        "policy.down_dims=${down_dims}"
        "policy.encoder_output_dim=${encoder_output_dim}"
        "training.use_ema=${use_ema}"
        "dataloader.num_workers=${dataloader_workers}"
        "val_dataloader.num_workers=${val_dataloader_workers}"
        "task.env_runner.eval_episodes=${eval_episodes}"
        "eval.save_video=${save_video}"
        "eval.n_gpu=${n_gpu}"
        "eval.n_cpu_per_gpu=${n_cpu_per_gpu}"
        "eval.source_dataset=${source_dataset}"
        "++eval.checkpoint_tag=${eval_ckpt_tag}"
        "++task.env_runner.n_control_steps=${control_steps}"
        "++task.env_runner.debug_action_print_steps=${debug_action_print_steps}"
    )
fi

export HYDRA_FULL_ERROR=1
python -W ignore eval.py --config-name=${algo}.yaml \
                         task=${task} \
                         hydra.run.dir=${run_dir} \
                         training.seed=${seed} \
                         training.device="cuda:0" \
                         exp_name=${exp_name} \
                         task.dataset.zarr_path=${zarr_path} \
                         "${extra_args[@]}"
