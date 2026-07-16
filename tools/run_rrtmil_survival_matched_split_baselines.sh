#!/usr/bin/env bash
set -u

ROOT="/data15/data15_5/fanhao/projects/MIRMIL"
PY="/data15/data15_5/fanhao/miniforge3/envs/rrtmil-surv/bin/python"
EXP_ROOT="/data15/data15_5/fanhao/experiments/RRTMIL_SURV_REFERENCE"
CSV_DIR="${ROOT}/reports/rrtmil_survival_reference/csv"
OUT_DIR="${EXP_ROOT}/matched_split"
LOG_DIR="${EXP_ROOT}/logs/matched_split"
STATUS="${STATUS:-${EXP_ROOT}/matched_split_status.tsv}"
RUN_ID="${RUN_ID:-matched_split_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

GPUS_CSV="${GPUS_CSV:-0,1,2,3,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
MAX_PARALLEL="${#GPUS[@]}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-4096}"
MODELS_CSV="${MODELS_CSV:-MeanMIL,MaxMIL,AttMIL,TransMIL,RRTMIL}"
DATASETS_CSV="${DATASETS_CSV:-KIRC_UNI_OS,KIRC_R50_OS,KIRC_UNI_PFS,KIRC_R50_PFS,BLCA_UNI_OS,BLCA_R50_OS}"
IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"

if [[ ! -f "${STATUS}" ]]; then
  echo -e "timestamp\trun_id\ttask_id\tdataset\tmodel\tgpu\tepochs\tseed\tmax_instances\tstatus\texit_code\tlog_path" > "${STATUS}"
fi

run_task() {
  local dataset="$1"
  local model="$2"
  local gpu="$3"
  local csv="${CSV_DIR}/${dataset}_rrtmil.csv"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local task_id="${dataset}_${model}_seed${SEED}_ep${EPOCHS}_max${MAX_INSTANCES}"
  local log="${LOG_DIR}/${task_id}_${ts}.log"

  if [[ ! -f "${csv}" ]]; then
    echo -e "$(date -Is)\t${RUN_ID}\t${task_id}\t${dataset}\t${model}\t${gpu}\t${EPOCHS}\t${SEED}\t${MAX_INSTANCES}\tmissing_csv\t127\t${log}" >> "${STATUS}"
    return 0
  fi

  echo -e "$(date -Is)\t${RUN_ID}\t${task_id}\t${dataset}\t${model}\t${gpu}\t${EPOCHS}\t${SEED}\t${MAX_INSTANCES}\trunning\tNA\t${log}" >> "${STATUS}"
  (
    echo "[task] ${task_id}"
    echo "[gpu] ${gpu}"
    echo "[csv] ${csv}"
    echo "[model] ${model}"
    echo "[epochs] ${EPOCHS}"
    echo "[seed] ${SEED}"
    echo "[max_instances] ${MAX_INSTANCES}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      "${PY}" "${ROOT}/tools/run_rrtmil_survival_matched_split.py" \
        --csv "${csv}" \
        --model "${model}" \
        --output-dir "${OUT_DIR}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --device cuda:0 \
        --num-workers 2 \
        --max-instances "${MAX_INSTANCES}"
    code=$?
    if [[ "${code}" == "0" ]]; then
      status="completed"
    else
      status="failed"
    fi
    echo -e "$(date -Is)\t${RUN_ID}\t${task_id}\t${dataset}\t${model}\t${gpu}\t${EPOCHS}\t${SEED}\t${MAX_INSTANCES}\t${status}\t${code}\t${log}" >> "${STATUS}"
    exit "${code}"
  ) > "${log}" 2>&1
}

TASKS_FILE="${EXP_ROOT}/${RUN_ID}_matched_tasks.tsv"
: > "${TASKS_FILE}"
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo -e "${dataset}\t${model}" >> "${TASKS_FILE}"
  done
done

worker() {
  local worker_idx="$1"
  local gpu="$2"
  local idx=0
  while IFS=$'\t' read -r dataset model; do
    if [[ $(( idx % MAX_PARALLEL )) -eq "${worker_idx}" ]]; then
      run_task "${dataset}" "${model}" "${gpu}" || true
    fi
    idx=$((idx + 1))
  done < "${TASKS_FILE}"
}

for worker_idx in "${!GPUS[@]}"; do
  worker "${worker_idx}" "${GPUS[$worker_idx]}" &
done

wait
echo "[done] all requested matched-split RRT-MIL survival baselines finished"
