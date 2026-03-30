demo=${1}
algo=${2}
task=${3}
seed=${4}

SAVE_VIDEO=${SAVE_VIDEO:-True}
EVAL_EPISODES=${EVAL_EPISODES:-1}
N_GPU=${N_GPU:-1}
N_CPU_PER_GPU=${N_CPU_PER_GPU:-1}

export SAVE_VIDEO
export EVAL_EPISODES
export N_GPU
export N_CPU_PER_GPU

bash eval.sh ${demo} ${algo} ${task} ${seed}
