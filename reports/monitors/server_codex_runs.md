# Server Codex run log

## 2026-07-16 01:55 CST

- Task: NSCLC HuggingFace snapshot download monitor
- Monitor PID: `3614892`
- Download PID being watched: `3566032`
- GPU: none
- Command: `reports/monitors/nsclc_download_watch.sh`
- Monitor log: `reports/monitors/nsclc_download_watch.log`
- Nohup log: `reports/monitors/nsclc_download_watch.nohup.log`
- PID file: `reports/monitors/nsclc_download_watch.pid`
- Data path: `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/nsclc`
- Source download log: `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_r50_uni_patches_20260716_002213.log`
- Current status at start: `patches=1046 files, r50=891 files, uni=226 files`; download still running, so NSCLC experiments not started.

## 2026-07-16 01:59 CST

- Task: BLCA R50 OS prognosis, RRT_MIL backbone, seed 2024
- PID: `3617537`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_BLCA_R50_OS ... Model.backbone=RRT_MIL Model.backbone_config=configs/RRT_MIL.yaml`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_rrt_seed2024_20260716_015944.log`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`

- Task: KIRC UNI OS prognosis, MIR_MIL backbone, seed 2024
- PID: `3617538`
- GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_UNI_OS ... Model.backbone=MIR_MIL Model.backbone_config=configs/MIR_MIL.yaml`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mir_seed2024_20260716_015944.log`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`

- Task: KIRC R50 OS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024
- PID: `3617539`
- GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_R50_OS ... Model.backbone=MIR_MIL Model.backbone_config=configs/releases/MIR_MIL_MT_V1.yaml`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mirmt_seed2024_20260716_015944.log`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`

- Task: KIRC UNI PFS prognosis, RRT_MIL backbone, seed 2024
- PID: `3617540`
- GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_UNI_PFS ... Model.backbone=RRT_MIL Model.backbone_config=configs/RRT_MIL.yaml`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_rrt_seed2024_20260716_015944.log`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`

- Task: KIRC R50 PFS prognosis, MIR_MIL backbone, seed 2024
- PID: `3617542`
- GPU: `5` via `CUDA_VISIBLE_DEVICES=5`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_R50_PFS ... Model.backbone=MIR_MIL Model.backbone_config=configs/MIR_MIL.yaml`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mir_seed2024_20260716_015944.log`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
- Status correction: the five non-`setsid` prognosis jobs above exited before producing epoch logs. Root causes found during debug: MIR backbones required `return_state=True` support in `SURVIVAL_MIL`; additionally, `setsid` is more reliable for persistent server-side launches from Codex tool sessions. Replaced by the `setsid` jobs below where applicable.

## 2026-07-16 02:07 CST

- Task: BLCA R50 OS prognosis, RRT_MIL backbone, seed 2024, retry with `setsid`
- PID/session leader: `3621779`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_BLCA_R50_OS ... Model.backbone=RRT_MIL ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_rrt_seed2024_setsid_20260716_020716.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_rrt_seed2024_setsid_20260716_020716.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`
- Verification: reached epoch 1 and continued training.
- Update 2026-07-16 02:11 CST: completed with early stopping at epoch 10; best checkpoint epoch 2; final test c-index `0.5426065162907269`.

## 2026-07-16 02:08 CST

- Task: KIRC UNI OS prognosis, MIR_MIL backbone, seed 2024, retry with `setsid`
- PID/session leader: `3623101`
- GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_UNI_OS ... Model.backbone=MIR_MIL ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mir_seed2024_setsid_20260716_020833.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mir_seed2024_setsid_20260716_020833.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
- Update 2026-07-16 02:11 CST: reached epoch 1; val c-index `0.6555944055944056`; still running.

- Task: KIRC UNI PFS prognosis, RRT_MIL backbone, seed 2024, retry with `setsid`
- PID/session leader: `3623102`
- GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_UNI_PFS ... Model.backbone=RRT_MIL ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_rrt_seed2024_setsid_20260716_020833.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_rrt_seed2024_setsid_20260716_020833.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
- Update 2026-07-16 02:11 CST: reached epoch 1; val c-index `0.7529527559055118`; still running.

## 2026-07-16 02:11 CST

- Task: NSCLC download status check
- Download PID: `3566032`
- Monitor PID: `3614892`
- Current status: `patches=1046 files, r50=892 files, uni=534 files`; download still running; NSCLC experiments still not started.

## 2026-07-16 02:13 CST

- Task: BLCA R50 OS prognosis, MIR_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3629597`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_BLCA_R50_OS ... Model.backbone=MIR_MIL Model.backbone_config=configs/MIR_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_mir_seed2024_setsid_20260716_021302.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_mir_seed2024_setsid_20260716_021302.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`
- Verification: initialized data/model and entered training.
- Update 2026-07-16 02:17 CST: completed with early stopping at epoch 11; best checkpoint epoch 3; final test c-index `0.5626566416040101`.

## 2026-07-16 02:14 CST

- Task: BLCA R50 OS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024, `setsid`
- PID/session leader: `3632623`
- GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_BLCA_R50_OS ... Model.backbone=MIR_MIL Model.backbone_config=configs/releases/MIR_MIL_MT_V1.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_mirmt_seed2024_setsid_20260716_021445.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_mirmt_seed2024_setsid_20260716_021445.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`
- Verification: initialized data/model and entered training.

## 2026-07-16 02:16 CST

- Task: KIRC R50 OS prognosis, RRT_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3638325`
- GPU: `5` via `CUDA_VISIBLE_DEVICES=5`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_R50_OS ... Model.backbone=RRT_MIL Model.backbone_config=configs/RRT_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_rrt_seed2024_setsid_20260716_021654.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_rrt_seed2024_setsid_20260716_021654.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
- Verification: initialized data/model and entered training.

## 2026-07-16 02:18 CST

- Task: BLCA R50 OS prognosis, AB_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3641481`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_BLCA_R50_OS ... Model.backbone=AB_MIL Model.backbone_config=configs/AB_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_ab_seed2024_setsid_20260716_021823.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_ab_seed2024_setsid_20260716_021823.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`
- Verification: initialized data/model and entered training.
