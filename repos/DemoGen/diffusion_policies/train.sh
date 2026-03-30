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

echo "[INFO] Using dataset: ${zarr_path}"

if [ "${task}" = "panda" ]; then
    default_batch_size=4
    default_val_batch_size=4
    default_num_epochs=160
    default_val_ratio=0.1
    default_checkpoint_every=20
    default_val_every=1
    default_sample_every=1
    default_save_ckpt=True
else
    default_batch_size=512
    default_val_batch_size=64
    default_num_epochs=1
    default_val_ratio=0.0
    default_checkpoint_every=null
    default_val_every=5
    default_sample_every=5
    default_save_ckpt=False
fi

batch_size=${BATCH_SIZE:-${default_batch_size}}
val_batch_size=${VAL_BATCH_SIZE:-${default_val_batch_size}}
num_epochs=${NUM_EPOCHS:-${default_num_epochs}}
max_train_steps=${MAX_TRAIN_STEPS:-null}
val_ratio=${VAL_RATIO:-${default_val_ratio}}
checkpoint_every=${CHECKPOINT_EVERY:-${default_checkpoint_every}}
val_every=${VAL_EVERY:-${default_val_every}}
sample_every=${SAMPLE_EVERY:-${default_sample_every}}
save_last_ckpt=${SAVE_LAST_CKPT:-True}
save_ckpt=${SAVE_CKPT:-${default_save_ckpt}}

echo "[INFO] Training config: batch_size=${batch_size}, val_batch_size=${val_batch_size}, num_epochs=${num_epochs}, max_train_steps=${max_train_steps}, val_ratio=${val_ratio}, checkpoint_every=${checkpoint_every}"

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
    optimizer_foreach=${OPTIMIZER_FOREACH:-False}
    dataloader_workers=${DATALOADER_WORKERS:-4}
    val_dataloader_workers=${VAL_DATALOADER_WORKERS:-2}

    echo "[INFO] Panda DP3 lightweight overrides: down_dims=${down_dims}, horizon=${horizon}, n_obs_steps=${n_obs_steps}, requested_n_action_steps=${requested_n_action_steps}, effective_n_action_steps=${n_action_steps}, encoder_output_dim=${encoder_output_dim}, use_ema=${use_ema}"

    extra_args+=(
        "horizon=${horizon}"
        "n_obs_steps=${n_obs_steps}"
        "n_action_steps=${n_action_steps}"
        "policy.down_dims=${down_dims}"
        "policy.encoder_output_dim=${encoder_output_dim}"
        "training.use_ema=${use_ema}"
        "+optimizer.foreach=${optimizer_foreach}"
        "dataloader.num_workers=${dataloader_workers}"
        "val_dataloader.num_workers=${val_dataloader_workers}"
        "task.dataset.val_ratio=${val_ratio}"
        "training.val_every=${val_every}"
        "training.sample_every=${sample_every}"
        "training.checkpoint_every=${checkpoint_every}"
        "checkpoint.save_last_ckpt=${save_last_ckpt}"
        "checkpoint.save_ckpt=${save_ckpt}"
    )
fi

export HYDRA_FULL_ERROR=1
python -W ignore train.py --config-name=${algo}.yaml \
                          task=${task} \
                          hydra.run.dir=${run_dir} \
                          training.debug=False \
                          training.seed=${seed} \
                          training.device="cuda:0" \
                          exp_name=${exp_name} \
                          logging.mode=offline \
                          training.num_epochs=${num_epochs} \
                          training.max_train_steps=${max_train_steps} \
                          dataloader.batch_size=${batch_size} \
                          val_dataloader.batch_size=${val_batch_size} \
                          task.dataset.zarr_path=${zarr_path} \
                          "${extra_args[@]}"
