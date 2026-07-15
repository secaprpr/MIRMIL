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

## 2026-07-16 02:21 CST

- Task: NSCLC download status check and monitor restart
- Download PID: `3566032`
- Previous monitor PID: `3614892` had exited.
- First restart attempt PID `3649230` also exited after one check under the tool session.
- Persistent restarted monitor PID: `3651599`
- Command: `setsid bash -lc 'cd /data15/data15_5/fanhao/projects/MIRMIL; exec reports/monitors/nsclc_download_watch.sh' > reports/monitors/nsclc_download_watch.nohup.log 2>&1 < /dev/null &`
- Monitor log: `reports/monitors/nsclc_download_watch.log`
- PID file: `reports/monitors/nsclc_download_watch.pid`
- Current status: `patches=1046 files, r50=893 files, uni=763 files`; download still running with HuggingFace retry messages; NSCLC experiments still not started.

- Task: KIRC R50 PFS prognosis, MIR_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3650041`
- GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_R50_PFS ... Model.backbone=MIR_MIL Model.backbone_config=configs/MIR_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mir_seed2024_setsid_20260716_022150.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mir_seed2024_setsid_20260716_022150.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
- Verification: session started; detailed epoch output pending.

## 2026-07-16 02:22 CST

- Task: COADREAD feature location recheck
- Command: `find /data15/data15_5/fanhao/datasets/TCGA-COADREAD -maxdepth 6 -type f \( -name '*.pt' -o -name '*.h5' -o -iname '*.svs' \)` plus dataset-root COAD/READ feature search.
- Result: no `.pt`, `.h5`, or `.svs` files found under `/data15/data15_5/fanhao/datasets/TCGA-COADREAD`; dataset-root COAD/READ feature search also returned no candidates.
- Current available data: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata` only, size about `25M`.
- Consequence: COADREAD prognosis training cannot start yet; need actual feature/WSI location from user or a later WSI download + patch + R50/UNI feature extraction step.

## 2026-07-16 02:24 CST

- Task: Prognosis job status update
- Completed: BLCA R50 OS + AB_MIL, status `exit_code=0`; early stopped at epoch 13, best checkpoint epoch 5, final test c-index `0.5281954887218046`.
- Completed: BLCA R50 OS + MIR_MIL_MT_V1, status `exit_code=0`; early stopped at epoch 22, best checkpoint epoch 14, final test c-index `0.5789473684210527`.
- Completed: KIRC UNI OS + MIR_MIL, status `exit_code=0`; early stopped at epoch 14, best checkpoint epoch 6, final test c-index `0.7807953443258971`.
- Still running: KIRC R50 OS + RRT_MIL, PID/session `3638325`; reached epoch 2, current val c-index `0.5454545454545454`.
- Still running: KIRC R50 PFS + MIR_MIL, PID/session `3650041`; reached epoch 1, current val c-index `0.6505905511811023`.
- NSCLC download/monitor: download PID `3566032` and monitor PID `3651599` alive; status `patches=1046 files, r50=893 files, uni=820 files`; NSCLC experiments still not started.

## 2026-07-16 02:24 CST

- Task: BLCA R50 OS prognosis, MEAN_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3654881`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_BLCA_R50_OS ... Model.backbone=MEAN_MIL Model.backbone_config=configs/MEAN_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_mean_seed2024_setsid_20260716_022450.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_mean_seed2024_setsid_20260716_022450.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`

- Task: BLCA R50 OS prognosis, MAX_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3654882`
- GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_BLCA_R50_OS ... Model.backbone=MAX_MIL Model.backbone_config=configs/MAX_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_max_seed2024_setsid_20260716_022450.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_max_seed2024_setsid_20260716_022450.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`

- Task: KIRC R50 OS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024, `setsid`
- PID/session leader: `3654883`
- GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_R50_OS ... Model.backbone=MIR_MIL Model.backbone_config=configs/releases/MIR_MIL_MT_V1.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mirmt_seed2024_setsid_20260716_022450.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mirmt_seed2024_setsid_20260716_022450.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`

## 2026-07-16 02:28 CST

- Task: NSCLC protected auto-start watcher
- PID/session leader: `3664269`
- GPU: none while waiting; later waits for GPU0/GPU1 memory below 500 MiB before launching R50/UNI benchmarks.
- Command: `setsid bash -lc 'cd /data15/data15_5/fanhao/projects/MIRMIL; exec reports/monitors/nsclc_auto_start.sh'`
- Script: `reports/monitors/nsclc_auto_start.sh`
- Log: `reports/monitors/nsclc_auto_start.log`
- Nohup log: `reports/monitors/nsclc_auto_start.nohup.log`
- PID file: `reports/monitors/nsclc_auto_start.pid`
- Guard marker: `reports/monitors/nsclc_benchmark_started.marker`
- Behavior: waits until a HuggingFace download `done` marker exists, resumes download if the process exits without `done`, validates all R50/UNI split feature paths, runs dry-runs, then launches NSCLC R50 and UNI benchmarks with the handoff model list and seeds `2024/2025/2026`.
- Current status at start: `patches=1046 files, r50=893 files, uni=917 files`; download still running, so benchmark not started yet.

## 2026-07-16 02:29 CST

- Task: NSCLC auto-start watcher script update and restart
- Previous watcher PID: `3664269`
- New watcher PID/session leader: `3669077`
- Reason: updated `reports/monitors/nsclc_auto_start.sh` so future automatic writes to `reports/monitors/server_codex_runs.md` are committed and pushed to `origin/main`.
- Command: `setsid bash -lc 'cd /data15/data15_5/fanhao/projects/MIRMIL; exec reports/monitors/nsclc_auto_start.sh'`
- Current status at restart: `patches=1046 files, r50=893 files, uni=952 files`; download still running, NSCLC benchmarks not started yet.

## 2026-07-16 02:31 CST

- Task: Prognosis job status update
- Completed: BLCA R50 OS + MAX_MIL, status `exit_code=0`; early stopped at epoch 12, best checkpoint epoch 4, final test c-index `0.45551378446115287`.
- Still running: BLCA R50 OS + MEAN_MIL; reached epoch 14, current val c-index `0.5138713745271122`.
- Still running: KIRC R50 OS + RRT_MIL; reached epoch 9, current val c-index `0.5437062937062938`.
- Still running: KIRC R50 PFS + MIR_MIL; reached epoch 7, current val c-index `0.687007874015748`.
- Still running: KIRC R50 OS + MIR_MIL_MT_V1; reached epoch 5, current val c-index `0.5122377622377622`.
- NSCLC download/monitor: download PID `3566032`, monitor PID `3651599`, auto-start PID `3669077` alive; status `patches=1046 files, r50=893 files, uni=971 files`; NSCLC benchmarks still waiting for download `done`.

- Task: KIRC UNI OS prognosis, RRT_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3670893`
- GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_UNI_OS ... Model.backbone=RRT_MIL Model.backbone_config=configs/RRT_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_rrt_seed2024_setsid_20260716_023050.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_rrt_seed2024_setsid_20260716_023050.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`

## 2026-07-16 02:34 CST

- Task: Prognosis job status update
- Completed: KIRC R50 PFS + MIR_MIL, status `exit_code=0`; early stopped at epoch 11, best checkpoint epoch 3, final test c-index `0.6164383561643836`.
- Still running: BLCA R50 OS + MEAN_MIL; reached epoch 24, current val c-index `0.5138713745271122`.
- Still running: KIRC R50 OS + RRT_MIL; reached epoch 12, current val c-index `0.6713286713286714`.
- Still running: KIRC R50 OS + MIR_MIL_MT_V1; reached epoch 9, current val c-index `0.5506993006993007`.
- Still running: KIRC UNI OS + RRT_MIL; reached epoch 3, current val c-index `0.708916083916084`.
- NSCLC download/monitor: download PID `3566032`, monitor PID `3651599`, auto-start PID `3669077` alive; status `patches=1046 files, r50=893 files, uni=975 files`; NSCLC benchmarks still waiting for download `done`.

- Task: KIRC R50 PFS prognosis, RRT_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3678267`
- GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_R50_PFS ... Model.backbone=RRT_MIL Model.backbone_config=configs/RRT_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_rrt_seed2024_setsid_20260716_023410.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_rrt_seed2024_setsid_20260716_023410.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`

## 2026-07-16 02:38 CST

- Task: NSCLC download and auto-start status
- Download PID: `3566032`, alive for about `02:15:48`.
- Monitor PID: `3651599`, alive.
- Auto-start watcher PID: `3669077`, alive.
- Current data status: `patches=1046 files, 772M`; `r50=893 files, 41G`; `uni=1040 files, 48G`.
- Download completion marker: no `] done` line found yet in `download_nsclc*.log`.
- Current log state: HuggingFace requests are still retrying intermittent `[Errno 101] Network is unreachable`; do not start NSCLC benchmark manually until the done marker appears or the watcher validates completion.

- Task: BLCA R50 OS prognosis, MEAN_MIL backbone, seed 2024, `setsid`
- Status: completed, status file `exit_code=0`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_mean_seed2024_setsid_20260716_022450.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_r50_os_mean_seed2024_setsid_20260716_022450.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`
- Result from controller log: final test c-index `0.5789473684210527`, event_count `35`, sample_count `77`; checkpoint line reports `Best_EPOCH_23.pth`.
- Caveat: MEAN_MIL and MAX_MIL were launched in the same minute with identical dataset/seed naming, so their internal experiment output directory may collide. Treat controller logs/status files as authoritative until Best_Log files are audited.

- Task: KIRC UNI PFS prognosis, MIR_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3684131`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_UNI_PFS ... Model.backbone=MIR_MIL Model.backbone_config=configs/MIR_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mir_seed2024_setsid_20260716_023620.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mir_seed2024_setsid_20260716_023620.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
- Verification: training initialized and reached epoch 1; val c-index `0.6309055118110236`.

- Task: Prognosis job status update
- Still running: KIRC R50 OS + RRT_MIL, PID/session `3638325`; reached epoch 18; recent val c-index `0.6162587412587412`; best observed in log so far around epoch 12 with val c-index `0.6713286713286714`.
- Still running: KIRC R50 OS + MIR_MIL_MT_V1, PID/session `3654883`; reached epoch 14; recent val c-index `0.5708041958041958`.
- Still running: KIRC UNI OS + RRT_MIL, PID/session `3670893`; reached epoch 8; recent val c-index `0.7736013986013986`.
- Still running: KIRC R50 PFS + RRT_MIL, PID/session `3678267`; reached epoch 4; recent val c-index `0.6476377952755905`.
- Still running: KIRC UNI PFS + MIR_MIL, PID/session `3684131`; reached epoch 1; recent val c-index `0.6309055118110236`.

## 2026-07-16 02:40 CST

- Task: Handoff continuation status check
- NSCLC: download PID `3566032`, monitor PID `3651599`, and auto-start PID `3669077` still alive. Current status remains `patches=1046 files, 772M`; `r50=893 files, 41G`; `uni=1040 files, 48G`; no download `] done` marker found, so NSCLC benchmarks must still wait.
- Running prognosis jobs:
  - KIRC R50 OS + RRT_MIL, PID/session `3638325`, reached epoch 19; recent val c-index `0.6407342657342657`.
  - KIRC R50 OS + MIR_MIL_MT_V1, PID/session `3654883`, reached epoch 16; recent val c-index `0.5743006993006993`.
  - KIRC UNI OS + RRT_MIL, PID/session `3670893`, reached epoch 10; recent val c-index `0.708916083916084`.
  - KIRC R50 PFS + RRT_MIL, PID/session `3678267`, reached epoch 6; recent val c-index `0.6269685039370079`.
  - KIRC UNI PFS + MIR_MIL, PID/session `3684131`, reached epoch 3; recent val c-index `0.735236220472441`.
- GPU decision: GPUs `0/1/2/3/5` are occupied by current prognosis jobs; GPUs `4/6/7` appear occupied by other long-running workloads. No new training task started in this check.

## 2026-07-16 02:41 CST

- Task: COADREAD feature location recheck
- Commands:
  - `find /data15/data15_5/fanhao/datasets -maxdepth 7 -type f \( -iname '*coad*.pt' -o -iname '*read*.pt' -o -iname '*coad*.h5' -o -iname '*read*.h5' -o -iname '*coad*.svs' -o -iname '*read*.svs' \)`
  - `find /data15/data15_5/fanhao/datasets/TCGA-COADREAD -maxdepth 8 -type f \( -name '*.pt' -o -name '*.h5' -o -iname '*.svs' -o -iname '*.tif' -o -iname '*.mrxs' -o -iname '*.ndpi' \)`
- Result: no COADREAD/COAD/READ feature or WSI files found under the dataset root search. `/data15/data15_5/fanhao/datasets/TCGA-COADREAD` still contains only metadata, total size about `25M`.
- Consequence: COADREAD prognosis training still cannot start from existing data. Need actual R50/UNI feature location or WSI download + patch + feature extraction before generating patient-level feature split CSVs.

## 2026-07-16 02:46 CST

- Task: Prognosis job status update
- Completed: KIRC UNI OS + RRT_MIL, status `exit_code=0`; early stopped at epoch 16, loaded checkpoint `Best_EPOCH_8.pth`, final test c-index `0.7759456838021338` with CI `[0.6637659923506988, 0.8758569902319902]`, event_count `25`, sample_count `74`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_rrt_seed2024_setsid_20260716_023050.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_rrt_seed2024_setsid_20260716_023050.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`

- Task: KIRC UNI OS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024, `setsid`
- PID/session leader: `3702155`
- GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_UNI_OS ... Model.backbone=MIR_MIL Model.backbone_config=configs/releases/MIR_MIL_MT_V1.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mirmt_seed2024_setsid_20260716_024608.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mirmt_seed2024_setsid_20260716_024608.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
- Verification: training initialized successfully on CUDA-visible GPU1; survival cutpoints loaded.

- NSCLC status: download PID `3566032`, monitor PID `3651599`, and auto-start PID `3669077` remain alive. No `] done` marker yet; latest file counts observed remain `patches=1046`, `r50=893`, `uni=1040`.

## 2026-07-16 02:47 CST

- Task: Prognosis job status update
- Completed: KIRC UNI PFS + MIR_MIL, status `exit_code=0`; early stopped at epoch 10, loaded checkpoint `Best_EPOCH_2.pth`, final test c-index `0.700587084148728` with CI `[0.5910273574269768, 0.8017440859024401]`, event_count `22`, sample_count `72`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mir_seed2024_setsid_20260716_023620.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mir_seed2024_setsid_20260716_023620.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`

- Task: KIRC UNI PFS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024, `setsid`
- PID/session leader: `3703950`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_UNI_PFS ... Model.backbone=MIR_MIL Model.backbone_config=configs/releases/MIR_MIL_MT_V1.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mirmt_seed2024_setsid_20260716_024711.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mirmt_seed2024_setsid_20260716_024711.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
- Verification: training initialized successfully on CUDA-visible GPU0; survival cutpoints loaded.

## 2026-07-16 02:48 CST

- Task: Prognosis job status update
- Completed: KIRC R50 OS + RRT_MIL, status `exit_code=0`; early stopped at epoch 28, loaded checkpoint `Best_EPOCH_20.pth`, final test c-index `0.6110572259941804` with CI `[0.45814163459118257, 0.75278096839181]`, event_count `25`, sample_count `74`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_rrt_seed2024_setsid_20260716_021654.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_rrt_seed2024_setsid_20260716_021654.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`

- Task: KIRC R50 PFS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024, `setsid`
- PID/session leader: `3705356`
- GPU: `5` via `CUDA_VISIBLE_DEVICES=5`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_R50_PFS ... Model.backbone=MIR_MIL Model.backbone_config=configs/releases/MIR_MIL_MT_V1.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mirmt_seed2024_setsid_20260716_024806.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mirmt_seed2024_setsid_20260716_024806.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
- Verification: training initialized successfully on CUDA-visible GPU5; survival cutpoints loaded.

## 2026-07-16 02:52 CST

- Task: Prognosis job status update
- Completed: KIRC R50 OS + MIR_MIL_MT_V1, status `exit_code=0`; completed 30 epochs, loaded checkpoint `Best_EPOCH_29.pth`, final test c-index `0.6935014548981572` with CI `[0.5811142326159898, 0.791194095672032]`, event_count `25`, sample_count `74`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mirmt_seed2024_setsid_20260716_022450.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mirmt_seed2024_setsid_20260716_022450.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
- NSCLC status: download PID `3566032`, monitor PID `3651599`, and auto-start PID `3669077` remain alive; no `] done` marker yet; latest counts remain `patches=1046`, `r50=893`, `uni=1040`.

- Task: KIRC R50 OS prognosis, AB_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3713137`
- GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
- Command: `setsid bash -lc '... train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml ... Dataset.DATASET_NAME=TCGA_KIRC_R50_OS ... Model.backbone=AB_MIL Model.backbone_config=configs/AB_MIL.yaml ...'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_ab_seed2024_setsid_20260716_025225.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_ab_seed2024_setsid_20260716_025225.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
- Verification: training initialized successfully on CUDA-visible GPU3; survival cutpoints loaded.

## 2026-07-16 02:54 CST

- Task: Handoff continuation status check
- NSCLC: download PID `3566032`, monitor PID `3651599`, and auto-start PID `3669077` still alive. No `] done` marker found. Current counts: `patches=1046 files, 772M`; `r50=894 files, 41G`; `uni=1040 files, 48G`.
- Running prognosis jobs:
  - KIRC R50 PFS + RRT_MIL, PID/session `3678267`, reached epoch 22; recent val c-index `0.6988188976377953`.
  - KIRC UNI OS + MIR_MIL_MT_V1, PID/session `3702155`, reached epoch 8; recent val c-index `0.6818181818181818`.
  - KIRC UNI PFS + MIR_MIL_MT_V1, PID/session `3703950`, reached epoch 7; recent val c-index `0.7017716535433071`.
  - KIRC R50 PFS + MIR_MIL_MT_V1, PID/session `3705356`, reached epoch 6; recent val c-index `0.6909448818897638`.
  - KIRC R50 OS + AB_MIL, PID/session `3713137`, reached epoch 1; recent val c-index `0.4090909090909091`.
- GPU decision: all available non-external GPUs are currently occupied by the above prognosis jobs; no new task started in this check.

## 2026-07-16 02:56 CST

- Task: NSCLC download recovery
- Observation: original HuggingFace snapshot download PID `3566032` was no longer alive. The original log `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_r50_uni_patches_20260716_002213.log` ended with `RuntimeError: Cannot send a request, as the client has been closed.`
- Current data status before resume: `patches=1046 files, 772M`; `r50=895 files, 41G`; `uni=1040 files, 48G`; no `] done` marker found.
- Action: started resume download without deleting existing data.
- New resume PID/session leader: `3720292`
- Command: `setsid bash -lc 'DEST=/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature python -u -c "... snapshot_download(repo_id=\"Dearcat/CPathPatchFeature\", repo_type=\"dataset\", allow_patterns=[\"nsclc/r50/**\",\"nsclc/uni/**\",\"nsclc/patches/**\"], local_dir=dest, max_workers=6) ..."'`
- Log: `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_resume_20260716_025617.log`
- PID file: `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_resume_20260716_025617.pid`
- Verification: PID `3720292` alive after launch; log contains resume start line.

## 2026-07-16 03:00 CST

- Task: Handoff continuation status check
- NSCLC: resume download PID/session `3720292` is alive. Current counts: `patches=1046 files, 772M`; `r50=989 files, 45G`; `uni=1040 files, 48G`. Resume log is progressing through HuggingFace snapshot files and does not yet contain the final `] done` marker, so NSCLC benchmarks remain blocked by download completion.
- GPU snapshot before new prognosis launches: GPUs `0/1/2/5` were idle; GPU `3` was running KIRC R50 OS + AB_MIL; GPUs `4/6/7` remained occupied by other workloads.

- Task: KIRC R50 PFS prognosis, RRT_MIL backbone, seed 2024, `setsid`
- Status: completed, status file `exit_code=0`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_rrt_seed2024_setsid_20260716_023410.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_rrt_seed2024_setsid_20260716_023410.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
- Result from controller log: loaded `Best_EPOCH_19.pth`; final test c-index `0.6438356164383562`, CI `[0.5006338659736451, 0.7683694446054603]`, event_count `22`, sample_count `72`.

- Task: KIRC UNI OS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024, `setsid`
- Status: completed, status file `exit_code=0`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mirmt_seed2024_setsid_20260716_024608.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mirmt_seed2024_setsid_20260716_024608.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
- Result from controller log: loaded `Best_EPOCH_5.pth`; final test c-index `0.7866149369544132`, CI `[0.6937703803812092, 0.863822916729381]`, event_count `25`, sample_count `74`.

- Task: KIRC UNI PFS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024, `setsid`
- Status: completed, status file `exit_code=0`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mirmt_seed2024_setsid_20260716_024711.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mirmt_seed2024_setsid_20260716_024711.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
- Result from controller log: loaded `Best_EPOCH_5.pth`; final test c-index `0.7837573385518591`, CI `[0.6915862994631139, 0.8762020651661364]`, event_count `22`, sample_count `72`.

- Task: KIRC R50 PFS prognosis, MIR_MIL_MT_V1 backbone config, seed 2024, `setsid`
- Status: completed, status file `exit_code=0`.
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mirmt_seed2024_setsid_20260716_024806.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mirmt_seed2024_setsid_20260716_024806.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
- Result from controller log: loaded `Best_EPOCH_3.pth`; final test c-index `0.6115459882583171`, CI `[0.4947975085671284, 0.7167438833940656]`, event_count `22`, sample_count `72`.

- Task: KIRC R50 OS prognosis, AB_MIL backbone, seed 2024, `setsid`
- Status: still running; no status file yet.
- PID/session leader: `3713137`
- GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_ab_seed2024_setsid_20260716_025225.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_ab_seed2024_setsid_20260716_025225.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
- Recent validation: reached epoch 8 with val c-index `0.5`.

- Task: KIRC R50 PFS prognosis, AB_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3726147`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `setsid bash -lc 'cd /data15/data15_5/fanhao/projects/MIRMIL; CUDA_VISIBLE_DEVICES=0 /data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python -u train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options General.seed=2024 General.num_epochs=30 General.device=0 General.num_workers=2 General.best_model_metric=c_index General.earlystop.use=true General.earlystop.patience=8 General.earlystop.metric=c_index Dataset.DATASET_NAME=TCGA_KIRC_R50_PFS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv Logs.log_root_dir=/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS Model.backbone=AB_MIL Model.backbone_config=configs/AB_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.num_bins=4 Model.survival.time_column=time_months Model.survival.event_column=event Model.survival.patient_column=patient_id Model.survival.patient_level=true; code=$?; echo exit_code=$code > /data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_ab_seed2024_setsid_20260716_030040.status; exit $code'`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_ab_seed2024_setsid_20260716_030040.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_ab_seed2024_setsid_20260716_030040.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

- Task: KIRC UNI OS prognosis, AB_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3726148`
- GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_ab_seed2024_setsid_20260716_030040.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_ab_seed2024_setsid_20260716_030040.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

- Task: KIRC UNI PFS prognosis, AB_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3726149`
- GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_ab_seed2024_setsid_20260716_030040.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_ab_seed2024_setsid_20260716_030040.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

- Task: KIRC R50 OS prognosis, MEAN_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3726150`
- GPU: `5` via `CUDA_VISIBLE_DEVICES=5`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mean_seed2024_setsid_20260716_030040.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mean_seed2024_setsid_20260716_030040.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

## 2026-07-16 03:07 CST

- Task: NSCLC download completion and benchmark launch
- Download status: completed. Resume log `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_resume_20260716_025617.log` contains `[2026-07-16T03:02:01.454019] done`.
- Final observed data counts: `patches=1046 files, 772M`; `r50=1039 files, 49G`; `uni=1052 files, 49G`.
- Split validation: both `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_R50_split.csv` and `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_UNI_split.csv` passed path validation.
- Auto-start watcher note: watcher PID `3669077` detected the download completion and validated splits, then exited before creating dry-run logs or launching benchmarks. Manual dry-run and launch were used to avoid leaving NSCLC idle.
- Dry-run logs:
  - R50: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_dry_run_manual_20260716_030636.log`, exit code `0`.
  - UNI: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_dry_run_manual_20260716_030636.log`, exit code `0`.
- Duplicate-start marker written locally: `/data15/data15_5/fanhao/projects/MIRMIL/reports/monitors/nsclc_benchmark_started.marker`.

- Task: NSCLC LUAD vs LUSC benchmark, R50 features
- PID/session leader: `3737790`
- GPU: `6` via `CUDA_VISIBLE_DEVICES=6`
- Command: `experiments/run_benchmark.py --split /data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_R50_split.csv --dataset-name TCGA_NSCLC_LUAD_LUSC_R50 --num-classes 2 --models AB_MIL CLAM_SB_MIL CLAM_MB_MIL DS_MIL TRANS_MIL RRT_MIL WIKG_MIL AC_MIL MO_MIL MAMBA2D_MIL MIR_MIL MIR_MIL_MT_V1 --seeds 2024 2025 2026 --epochs 30 --feature r50 --protocol nsclc_luad_lusc --split-id r50_v1 --max-instances 4096`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_gpu6_manual_20260716_030702.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_gpu6_manual_20260716_030702.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_R50_split.csv`
- Verification: benchmark entered `AB_MIL` seed `2024` training; no immediate failure.

- Task: NSCLC LUAD vs LUSC benchmark, UNI features
- PID/session leader: `3737792`
- GPU: `0` via `CUDA_VISIBLE_DEVICES=0`
- Command: `experiments/run_benchmark.py --split /data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_UNI_split.csv --dataset-name TCGA_NSCLC_LUAD_LUSC_UNI --num-classes 2 --models AB_MIL CLAM_SB_MIL CLAM_MB_MIL DS_MIL TRANS_MIL RRT_MIL WIKG_MIL AC_MIL MO_MIL MAMBA2D_MIL MIR_MIL MIR_MIL_MT_V1 --seeds 2024 2025 2026 --epochs 30 --feature uni --protocol nsclc_luad_lusc --split-id uni_v1 --max-instances 4096`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_gpu0_manual_20260716_030702.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_gpu0_manual_20260716_030702.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_UNI_split.csv`
- Verification: benchmark entered `AB_MIL` seed `2024` training; no immediate failure.

## 2026-07-16 03:11 CST

- Task: NSCLC benchmark progress check
- R50 benchmark remains running.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_gpu6_manual_20260716_030702.log`
  - Progress: `AB_MIL`, seed `2024`, completed epoch `1`.
  - Epoch 1 validation: acc `0.605`, bacc `0.6141026924231808`, macro_auc `0.7419677709938945`.
  - Epoch 1 test snapshot: acc `0.6698113207547169`, bacc `0.6643518518518519`, macro_auc `0.7886396011396012`.
- UNI benchmark remains running.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_gpu0_manual_20260716_030702.log`
  - Progress: `AB_MIL`, seed `2024`, completed epoch `1`.
  - Epoch 1 validation: acc `0.88`, bacc `0.8825943349014113`, macro_auc `0.9611650485436893`.
  - Epoch 1 test snapshot: acc `0.9009433962264151`, bacc `0.899394586894587`, macro_auc `0.9824608262108262`.
- Decision: keep both NSCLC benchmark controllers running; no duplicate NSCLC job launched.

- Task: COADREAD feature/WSI availability recheck
- Dataset-root search result: still no COADREAD `.pt`, `.h5`, `.svs`, `.tif`, `.mrxs`, or `.ndpi` files found under `/data15/data15_5/fanhao/datasets/TCGA-COADREAD`.
- Alias directory search under `/data15/data15_5/fanhao/datasets` found only `/data15/data15_5/fanhao/datasets/TCGA-COADREAD` metadata and no alternate COAD/READ feature directory.
- Current COADREAD dataset size remains metadata-only, about `25M`.
- Consequence: COADREAD prognosis split generation/training is still blocked by missing feature or WSI paths. Do not start COADREAD prognosis until actual feature/WSI location is supplied or generated.

- Task: KIRC prognosis concurrency decision
- Current running KIRC jobs: R50 OS + AB_MIL, R50 PFS + AB_MIL, UNI OS + AB_MIL, UNI PFS + AB_MIL, R50 OS + MEAN_MIL.
- Decision: do not launch additional KIRC MEAN/MAX jobs in this checkpoint because two NSCLC benchmark controllers and five KIRC prognosis controllers are already active. Add the next KIRC MEAN/MAX jobs when one or more current controllers finish.

## 2026-07-16 03:14 CST

- Task: KIRC prognosis status update
- Completed: KIRC R50 PFS + AB_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_ab_seed2024_setsid_20260716_030040.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_ab_seed2024_setsid_20260716_030040.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_4.pth`; final test c-index `0.6252446183953033`, CI `[0.5050155588209871, 0.7358311502738747]`, event_count `22`, sample_count `72`.
- Completed: KIRC UNI PFS + AB_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_ab_seed2024_setsid_20260716_030040.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_ab_seed2024_setsid_20260716_030040.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_1.pth`; final test c-index `0.738747553816047`, CI `[0.6423269971719779, 0.8335540830446816]`, event_count `22`, sample_count `72`.
- Still running: KIRC R50 OS + AB_MIL, KIRC UNI OS + AB_MIL, KIRC R50 OS + MEAN_MIL.

- Task: KIRC R50 PFS prognosis, MEAN_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3759405`
- GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options General.seed=2024 General.num_epochs=30 General.best_model_metric=c_index Dataset.DATASET_NAME=TCGA_KIRC_R50_PFS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv Model.backbone=MEAN_MIL Model.backbone_config=configs/MEAN_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mean_seed2024_setsid_20260716_031420.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mean_seed2024_setsid_20260716_031420.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

## 2026-07-16 03:19 CST

- Task: KIRC prognosis status update
- Completed: KIRC R50 OS + AB_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_ab_seed2024_setsid_20260716_025225.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_ab_seed2024_setsid_20260716_025225.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_20.pth`; final test c-index `0.7604267701260912`, CI `[0.6660128720048588, 0.8486052542477222]`, event_count `25`, sample_count `74`.
- Still running: KIRC UNI OS + AB_MIL, KIRC R50 OS + MEAN_MIL, KIRC R50 PFS + MEAN_MIL, NSCLC R50 benchmark, NSCLC UNI benchmark.

- Task: KIRC UNI OS prognosis, MEAN_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3787234`
- GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options General.seed=2024 General.num_epochs=30 General.best_model_metric=c_index Dataset.DATASET_NAME=TCGA_KIRC_UNI_OS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv Model.backbone=MEAN_MIL Model.backbone_config=configs/MEAN_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mean_seed2024_setsid_20260716_031920.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mean_seed2024_setsid_20260716_031920.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

## 2026-07-16 03:21 CST

- Task: NSCLC benchmark progress check
- R50 benchmark remains running on `AB_MIL`, seed `2024`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_gpu6_manual_20260716_030702.log`
  - Recent progress: reached epoch `23`; epoch 23 validation macro_auc `0.9236312681413272`; test snapshot macro_auc `0.9365206552706552`.
- UNI benchmark remains running and has moved from `AB_MIL` to `CLAM_SB_MIL`, seed `2024`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_gpu0_manual_20260716_030702.log`
  - Recent progress: `AB_MIL` reached epoch `27`; `CLAM_SB_MIL` seed `2024` has started.

- Task: KIRC prognosis status update
- Completed: KIRC UNI OS + AB_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_ab_seed2024_setsid_20260716_030040.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_ab_seed2024_setsid_20260716_030040.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_6.pth`; final test c-index `0.7759456838021338`, CI `[0.6789173224808199, 0.867289886039886]`, event_count `25`, sample_count `74`.
- Still running: KIRC R50 OS + MEAN_MIL, KIRC R50 PFS + MEAN_MIL, KIRC UNI OS + MEAN_MIL, NSCLC R50 benchmark, NSCLC UNI benchmark.

- Task: KIRC UNI PFS prognosis, MEAN_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3795273`
- GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options General.seed=2024 General.num_epochs=30 General.best_model_metric=c_index Dataset.DATASET_NAME=TCGA_KIRC_UNI_PFS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv Model.backbone=MEAN_MIL Model.backbone_config=configs/MEAN_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mean_seed2024_setsid_20260716_032100.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mean_seed2024_setsid_20260716_032100.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

- Task: COADREAD feature/WSI availability recheck
- Current status remains metadata-only: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata` is about `25M`; no `.pt`, `.h5`, or WSI files were found under the COADREAD dataset directory.
- Consequence: COADREAD prognosis remains blocked by missing feature/WSI paths.

## 2026-07-16 03:23 CST

- Task: NSCLC benchmark partial results
- R50 benchmark status: controller still running; `AB_MIL` seed `2024` completed and controller moved to `CLAM_SB_MIL`.
  - Controller log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_gpu6_manual_20260716_030702.log`
  - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_R50/AB_MIL/time_2026-07-16-03-07_TCGA_NSCLC_LUAD_LUSC_R50_AB_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_R50_AB_MIL.csv`
  - Best epoch `29`; validation macro_auc `0.934740...`; test acc `0.8773584905660378`, test bacc `0.8757122507122507`, test macro_auc `0.9390135327635327`.
- UNI benchmark status: controller still running; `AB_MIL` seed `2024` completed and controller moved to `CLAM_SB_MIL`.
  - Controller log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_gpu0_manual_20260716_030702.log`
  - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_UNI/AB_MIL/time_2026-07-16-03-07_TCGA_NSCLC_LUAD_LUSC_UNI_AB_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_UNI_AB_MIL.csv`
  - Best epoch `19`; validation macro_auc `0.981433...`; test acc `0.9528301886792453`, test bacc `0.9528133903133903`, test macro_auc `0.9865562678062678`.

- Task: KIRC prognosis status checkpoint
- Still running: KIRC R50 OS + MEAN_MIL, KIRC R50 PFS + MEAN_MIL, KIRC UNI OS + MEAN_MIL, KIRC UNI PFS + MEAN_MIL.
- No new KIRC job launched in this checkpoint because the next MEAN tasks are already active and MAX jobs should wait for another controller to finish to avoid overloading concurrent CPU/I/O.

## 2026-07-16 03:29 CST

- Task: KIRC prognosis status update
- Completed: KIRC R50 OS + MEAN_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mean_seed2024_setsid_20260716_030040.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_mean_seed2024_setsid_20260716_030040.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_29.pth`; final test c-index `0.6042677012609118`, CI `[0.4957872743721191, 0.7066098365438]`, event_count `25`, sample_count `74`.
- Still running: NSCLC R50 benchmark, NSCLC UNI benchmark, KIRC R50 PFS + MEAN_MIL, KIRC UNI OS + MEAN_MIL, KIRC UNI PFS + MEAN_MIL.
- COADREAD status remains metadata-only: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata` is about `25M`; no `.pt`, `.h5`, or WSI files found under the dataset directory.

- Task: KIRC R50 OS prognosis, MAX_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3839797`
- GPU: `5` via `CUDA_VISIBLE_DEVICES=5`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options General.seed=2024 General.num_epochs=30 General.best_model_metric=c_index Dataset.DATASET_NAME=TCGA_KIRC_R50_OS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv Model.backbone=MAX_MIL Model.backbone_config=configs/MAX_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_max_seed2024_setsid_20260716_032906.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_max_seed2024_setsid_20260716_032906.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

## 2026-07-16 03:31 CST

- Task: KIRC prognosis status update
- Completed: KIRC UNI PFS + MEAN_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mean_seed2024_setsid_20260716_032100.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_mean_seed2024_setsid_20260716_032100.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_1.pth`; final test c-index `0.7553816046966731`, CI `[0.652123281768432, 0.843697948262632]`, event_count `22`, sample_count `72`.
- Still running: NSCLC R50 benchmark, NSCLC UNI benchmark, KIRC R50 PFS + MEAN_MIL, KIRC UNI OS + MEAN_MIL, KIRC R50 OS + MAX_MIL.

- Task: KIRC UNI PFS prognosis, MAX_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3849977`
- GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options General.seed=2024 General.num_epochs=30 General.best_model_metric=c_index Dataset.DATASET_NAME=TCGA_KIRC_UNI_PFS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv Model.backbone=MAX_MIL Model.backbone_config=configs/MAX_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_max_seed2024_setsid_20260716_033056.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_max_seed2024_setsid_20260716_033056.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

## 2026-07-16 03:32 CST

- Task: KIRC prognosis status update
- Completed: KIRC R50 PFS + MEAN_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mean_seed2024_setsid_20260716_031420.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_mean_seed2024_setsid_20260716_031420.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_9.pth`; final test c-index `0.6203522504892368`, CI `[0.5134583796503212, 0.725546510413039]`, event_count `22`, sample_count `72`.
- Completed: KIRC UNI OS + MEAN_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mean_seed2024_setsid_20260716_031920.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_mean_seed2024_setsid_20260716_031920.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_4.pth`; final test c-index `0.7478176527643065`, CI `[0.6412621595631525, 0.8409200606784958]`, event_count `25`, sample_count `74`.

- Task: KIRC R50 PFS prognosis, MAX_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3856454`
- GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options General.seed=2024 General.num_epochs=30 General.best_model_metric=c_index Dataset.DATASET_NAME=TCGA_KIRC_R50_PFS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv Model.backbone=MAX_MIL Model.backbone_config=configs/MAX_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_max_seed2024_setsid_20260716_033205.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_max_seed2024_setsid_20260716_033205.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

- Task: KIRC UNI OS prognosis, MAX_MIL backbone, seed 2024, `setsid`
- PID/session leader: `3856531`
- GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
- Command: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options General.seed=2024 General.num_epochs=30 General.best_model_metric=c_index Dataset.DATASET_NAME=TCGA_KIRC_UNI_OS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv Model.backbone=MAX_MIL Model.backbone_config=configs/MAX_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
- Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_max_seed2024_setsid_20260716_033206.log`
- Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_max_seed2024_setsid_20260716_033206.status`
- Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
- Verification: training initialized, survival cutpoints loaded.

## 2026-07-16 03:33 CST

- Task: NSCLC benchmark progress and partial results
- R50 benchmark remains running; current active child is `CLAM_SB_MIL`, seed `2024`.
  - Controller log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_gpu6_manual_20260716_030702.log`
  - Recent progress: `CLAM_SB_MIL` reached epoch `27`; no `Best_Log` was available yet for R50 `CLAM_SB_MIL` at this checkpoint.
- UNI benchmark remains running and has moved to `TRANS_MIL`, seed `2024`.
  - Controller log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_gpu0_manual_20260716_030702.log`
  - Completed UNI `CLAM_SB_MIL`, seed `2024`.
    - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_UNI/CLAM_SB_MIL/time_2026-07-16-03-20_TCGA_NSCLC_LUAD_LUSC_UNI_CLAM_SB_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_UNI_CLAM_SB_MIL.csv`
    - Best epoch `4`; validation macro_auc `0.9816835151636473`; test acc `0.9528301886792453`, test bacc `0.9528133903133903`, test macro_auc `0.9869123931623931`.
  - Completed UNI `CLAM_MB_MIL`, seed `2024`.
    - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_UNI/CLAM_MB_MIL/time_2026-07-16-03-24_TCGA_NSCLC_LUAD_LUSC_UNI_CLAM_MB_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_UNI_CLAM_MB_MIL.csv`
    - Best epoch `3`; validation macro_auc `0.9892903613251927`; test acc `0.9386792452830188`, test bacc `0.9391025641025641`, test macro_auc `0.9870014245014245`.
  - Completed UNI `DS_MIL`, seed `2024`.
    - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_UNI/DS_MIL/time_2026-07-16-03-29_TCGA_NSCLC_LUAD_LUSC_UNI_DS_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_UNI_DS_MIL.csv`
    - Best epoch `1`; validation macro_auc `0.9820838754879392`; test acc `0.9245283018867925`, test bacc `0.9239672364672364`, test macro_auc `0.9838853276353275`.

- Task: KIRC prognosis status checkpoint
- Four MAX_MIL jobs remain running and no final status file exists yet:
  - KIRC R50 OS + MAX_MIL, GPU `5`, PID/session `3839797`.
  - KIRC UNI PFS + MAX_MIL, GPU `1`, PID/session `3849977`.
  - KIRC R50 PFS + MAX_MIL, GPU `2`, PID/session `3856454`.
  - KIRC UNI OS + MAX_MIL, GPU `3`, PID/session `3856531`.

- Task: COADREAD feature/WSI availability recheck
- Dataset-root search result: still no COADREAD `.pt`, `.h5`, `.svs`, `.tif`, `.mrxs`, or `.ndpi` files found under `/data15/data15_5/fanhao/datasets/TCGA-COADREAD`.
- Current COADREAD dataset size remains metadata-only, about `25M`.
- Consequence: COADREAD prognosis remains blocked by missing feature/WSI paths.

## 2026-07-16 03:35 CST

- Task: NSCLC benchmark progress and partial results
- R50 benchmark remains running and has moved from `CLAM_SB_MIL` to `CLAM_MB_MIL`, seed `2024`.
  - Controller log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_gpu6_manual_20260716_030702.log`
  - Completed R50 `CLAM_SB_MIL`, seed `2024`.
    - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_R50/CLAM_SB_MIL/time_2026-07-16-03-22_TCGA_NSCLC_LUAD_LUSC_R50_CLAM_SB_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_R50_CLAM_SB_MIL.csv`
    - Best epoch `28`; validation macro_auc `0.9315383845460914`; test acc `0.8820754716981132`, test bacc `0.8812321937321937`, test macro_auc `0.9452457264957265`.
- UNI benchmark remains running; current active child is `TRANS_MIL`, seed `2024`.
- KIRC MAX_MIL jobs remain running; no MAX final status files were present at this checkpoint.
- COADREAD status unchanged: metadata-only, about `25M`, no `.pt`, `.h5`, or WSI files found under `/data15/data15_5/fanhao/datasets/TCGA-COADREAD`.

## 2026-07-16 03:37 CST

- Task: active run checkpoint
- NSCLC benchmark controllers remain running:
  - R50 controller PID/session `3737790`, current active child `CLAM_MB_MIL`, seed `2024`, GPU `6`.
  - UNI controller PID/session `3737792`, current active child `TRANS_MIL`, seed `2024`, GPU `0`.
- No additional NSCLC `Best_Log` files were present beyond the previously recorded R50 `AB_MIL`/`CLAM_SB_MIL` and UNI `AB_MIL`/`CLAM_SB_MIL`/`CLAM_MB_MIL`/`DS_MIL` seed `2024` results.
- KIRC MAX_MIL jobs remain running and no final status files were present:
  - R50 OS + MAX_MIL, GPU `5`, PID/session `3839797`; latest observed validation c-index sequence includes best-so-far `0.6494755244755245`.
  - UNI PFS + MAX_MIL, GPU `1`, PID/session `3849977`; latest observed validation c-index sequence includes best-so-far `0.7431102362204725`.
  - R50 PFS + MAX_MIL, GPU `2`, PID/session `3856454`; latest observed validation c-index sequence includes best-so-far `0.5334645669291339`.
  - UNI OS + MAX_MIL, GPU `3`, PID/session `3856531`; latest observed validation c-index sequence includes best-so-far `0.7263986013986014`.
- COADREAD recheck:
  - Command scope: searched `/data15/data15_5/fanhao/datasets/TCGA-COADREAD` to max depth 6 for `.pt`, `.h5`, `.svs`, `.tif`, `.mrxs`, `.ndpi`.
  - Result: no feature or WSI files found; dataset root remains about `25M`.
  - Consequence: COADREAD prognosis is still blocked by missing feature/WSI paths; no training or extraction was started.

## 2026-07-16 03:41 CST

- Task: active run checkpoint
- NSCLC benchmark controllers remain running; no new final `Best_Log` files were present at this checkpoint.
  - R50 current active child remains `CLAM_MB_MIL`, seed `2024`; latest observed progress reached epoch `13` with validation macro AUC around `0.9109198278450606` at epoch `13` and best observed validation macro AUC in this run around `0.9172255029526574`.
  - UNI current active child remains `TRANS_MIL`, seed `2024`; latest observed progress reached epoch `11` with validation macro AUC around `0.9772795515964369` at epoch `11`.
- KIRC MAX_MIL jobs remain running; no final status files were observed.
  - R50 OS + MAX_MIL reached at least epoch `11`, latest observed validation c-index `0.6625874125874126`.
  - UNI PFS + MAX_MIL reached at least epoch `8`, latest observed validation c-index `0.6958661417322834`; best-so-far remains `0.7431102362204725`.
  - R50 PFS + MAX_MIL reached at least epoch `8`, latest observed validation c-index `0.4409448818897638`; best-so-far remains `0.5334645669291339`.
  - UNI OS + MAX_MIL reached at least epoch `7`, latest observed validation c-index `0.7211538461538461`; best-so-far remains `0.7263986013986014`.
- GPU status check: GPUs `0/1/2/3/5/6` are being used by current monitored jobs; GPUs `4` and `7` remain occupied by other workloads. No additional jobs were launched.

## 2026-07-16 03:43 CST

- Task: KIRC prognosis status update
- Completed: KIRC R50 PFS + MAX_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_max_seed2024_setsid_20260716_033205.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_pfs_max_seed2024_setsid_20260716_033205.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_2.pth`; final test c-index `0.5391389432485323`, CI `[0.39253924672926926, 0.6648910254851739]`, event_count `22`, sample_count `72`.
- Completed: KIRC UNI PFS + MAX_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_max_seed2024_setsid_20260716_033056.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_pfs_max_seed2024_setsid_20260716_033056.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_4.pth`; final test c-index `0.7289628180039139`, CI `[0.6282518810013096, 0.8322681124744667]`, event_count `22`, sample_count `72`.
- Completed: KIRC UNI OS + MAX_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_max_seed2024_setsid_20260716_033206.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_uni_os_max_seed2024_setsid_20260716_033206.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_2.pth`; final test c-index `0.7187196896217265`, CI `[0.6073118426818581, 0.8244928602572488]`, event_count `25`, sample_count `74`.
- Still running:
  - KIRC R50 OS + MAX_MIL, GPU `5`, PID/session `3839797`; latest observed epoch `14`.
  - NSCLC R50 benchmark, current active child `CLAM_MB_MIL`, seed `2024`.
  - NSCLC UNI benchmark, current active child `TRANS_MIL`, seed `2024`.
- BLCA R50 OS prognosis baseline results were confirmed from existing controller logs, all status `exit_code=0`:
  - RRT_MIL: final test c-index `0.5426065162907269`, CI `[0.42865408922272136, 0.6446856291342209]`, events `35`, samples `77`.
  - MIR_MIL: final test c-index `0.5626566416040101`, CI `[0.44868439944711663, 0.6766564507733781]`, events `35`, samples `77`.
  - MIR_MIL_MT_V1: final test c-index `0.5789473684210527`, CI `[0.4560474863589133, 0.6875786749482402]`, events `35`, samples `77`.
  - AB_MIL: final test c-index `0.5281954887218046`, CI `[0.41153337643396415, 0.6381425404446016]`, events `35`, samples `77`.
  - MEAN_MIL: final test c-index `0.5789473684210527`, CI `[0.4522940843286445, 0.6887805183835615]`, events `35`, samples `77`.
  - MAX_MIL: final test c-index `0.45551378446115287`, CI `[0.35063739280386547, 0.5560360507782959]`, events `35`, samples `77`.
- BLCA data check:
  - Existing prognosis split found for R50 OS: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv`.
  - UNI `.pt` features exist under `/data15/data15_5/fanhao/datasets/TCGA-BLCA/CPathPatchFeature/blca/uni/pt_files`.
  - No existing BLCA UNI prognosis split was found in metadata at this checkpoint; next step is to create a UNI OS split by preserving the same R50 split assignment and mapping feature paths to existing UNI `.pt` files.

## 2026-07-16 03:47 CST

- Task: BLCA UNI OS prognosis preparation and launch
- Found two missing BLCA UNI `.pt` files when mapping the R50 OS split to UNI:
  - `TCGA-GV-A3JW-01Z-00-DX1.152AD6E5-D30A-4D94-A793-3DDF3828625C.pt`
  - `TCGA-GC-A3WC-01Z-00-DX1.D8F5CD43-7338-414C-ADE8-AC0BBC6A871C.pt`
- Action: downloaded only those two missing UNI features from `Dearcat/CPathPatchFeature` into `/data15/data15_5/fanhao/datasets/TCGA-BLCA/CPathPatchFeature`.
  - Download log: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/CPathPatchFeature/logs/download_blca_missing_uni_20260716_034520.log`
  - Verification: both files exist after download; sizes `53778021` and `8402533` bytes.
- Created BLCA UNI OS split by preserving the R50 OS train/val/test slide and patient assignment and replacing only feature paths with UNI `.pt` paths.
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv`
  - Manifest: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_manifest.json`
  - Rows: `259`; feature entries: `454`; missing features: `0`; split SHA256 `14512d848ef33d815b8b3647b5be16da66b229226923cae65bea25fad63ee05d`.
- Launched BLCA UNI OS prognosis, seed `2024`, using the same survival settings as BLCA R50 OS.
  - RRT_MIL:
    - PID/session leader: `3922282`
    - GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
    - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_rrt_seed2024_setsid_20260716_034634.log`
    - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_rrt_seed2024_setsid_20260716_034634.status`
    - Command core: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv Model.backbone=RRT_MIL Model.backbone_config=configs/RRT_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
  - MIR_MIL:
    - PID/session leader: `3922284`
    - GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
    - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mir_seed2024_setsid_20260716_034634.log`
    - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mir_seed2024_setsid_20260716_034634.status`
    - Command core: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv Model.backbone=MIR_MIL Model.backbone_config=configs/MIR_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
  - MIR_MIL_MT_V1:
    - Initial attempt `blca_uni_os_mirmt_seed2024_setsid_20260716_034634` failed with `exit_code=1` because `Model.backbone=MIR_MIL_MT_V1` is not a valid survival backbone name.
    - Corrected PID/session leader: `3924419`
    - GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
    - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mirmt_seed2024_setsid_20260716_034709.log`
    - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mirmt_seed2024_setsid_20260716_034709.status`
    - Command core: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv Model.backbone=MIR_MIL Model.backbone_config=configs/releases/MIR_MIL_MT_V1.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
- Verification: all three corrected/valid BLCA UNI OS jobs initialized, loaded survival cutpoints `[7.05, 13.34, 22.32]`, and entered `Train_Val_Test`.

## 2026-07-16 03:48 CST

- Task: KIRC prognosis and NSCLC benchmark status update
- Completed: KIRC R50 OS + MAX_MIL, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_max_seed2024_setsid_20260716_032906.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/kirc_r50_os_max_seed2024_setsid_20260716_032906.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_11.pth`; final test c-index `0.584384093113482`, CI `[0.4347511937672782, 0.7295231799753217]`, event_count `25`, sample_count `74`.
- Completed: NSCLC R50 `CLAM_MB_MIL`, seed `2024`.
  - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_R50/CLAM_MB_MIL/time_2026-07-16-03-34_TCGA_NSCLC_LUAD_LUSC_R50_CLAM_MB_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_R50_CLAM_MB_MIL.csv`
  - Best epoch `19`; validation macro_auc `0.9274346912220999`; test acc `0.9009433962264151`, test bacc `0.9006410256410255`, test macro_auc `0.9448005698005697`.
  - R50 benchmark controller remains running and has moved to `DS_MIL`, seed `2024`.
- BLCA UNI OS jobs:
  - RRT_MIL, MIR_MIL, and corrected MIR_MIL_MT_V1 jobs remain running.
  - First observed BLCA UNI OS epoch-1 validation c-index values:
    - RRT_MIL `0.5435056746532156`
    - MIR_MIL `0.4697351828499369`
    - MIR_MIL_MT_V1 `0.5441361916771753`
- Launched BLCA UNI OS + AB_MIL, seed `2024`.
  - PID/session leader: `3931478`
  - GPU: `5` via `CUDA_VISIBLE_DEVICES=5`
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_ab_seed2024_setsid_20260716_034833.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_ab_seed2024_setsid_20260716_034833.status`
  - Command core: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv Model.backbone=AB_MIL Model.backbone_config=configs/AB_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
  - Verification: initialized, loaded survival cutpoints `[7.05, 13.34, 22.32]`, and entered survival process.

## 2026-07-16 03:50 CST

- Task: NSCLC benchmark and BLCA UNI OS checkpoint
- Completed: NSCLC UNI `TRANS_MIL`, seed `2024`.
  - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_UNI/TRANS_MIL/time_2026-07-16-03-32_TCGA_NSCLC_LUAD_LUSC_UNI_TRANS_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_UNI_TRANS_MIL.csv`
  - Best epoch `19`; validation macro_auc `0.9845861275147633`; test acc `0.9622641509433962`, test bacc `0.9617165242165242`, test macro_auc `0.9867343304843306`.
  - UNI benchmark controller remains running and has moved to `RRT_MIL`, seed `2024`.
- NSCLC R50 benchmark remains running; current active child is `DS_MIL`, seed `2024`.
- BLCA UNI OS status:
  - Valid RRT_MIL, MIR_MIL, MIR_MIL_MT_V1, and AB_MIL jobs remain running.
  - The failed initial MIR_MIL_MT_V1 launch `blca_uni_os_mirmt_seed2024_setsid_20260716_034634` remains recorded as `exit_code=1`; corrected run `blca_uni_os_mirmt_seed2024_setsid_20260716_034709` is still active.
  - No BLCA UNI OS final status files were present yet for the valid runs.
- Scheduling decision: did not launch BLCA UNI OS MEAN_MIL/MAX_MIL at this checkpoint because existing BLCA UNI OS jobs plus NSCLC R50/UNI benchmark jobs are still active; wait for a BLCA or NSCLC GPU/CPU slot to finish before adding more prognosis runs.

## 2026-07-16 03:52 CST

- Task: BLCA UNI OS prognosis status update
- Completed: BLCA UNI OS + MIR_MIL_MT_V1 corrected run, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mirmt_seed2024_setsid_20260716_034709.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mirmt_seed2024_setsid_20260716_034709.status`
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv`
  - Result from controller log: loaded `Best_EPOCH_2.pth`; final test c-index `0.6372180451127819`, CI `[0.5394515504312384, 0.7455460091838602]`, event_count `35`, sample_count `77`.
- Still running:
  - BLCA UNI OS + RRT_MIL, GPU `1`, PID/session `3922282`; latest observed epoch `12`, best observed validation c-index `0.5983606557377049`.
  - BLCA UNI OS + MIR_MIL, GPU `2`, PID/session `3922284`; latest observed epoch `11`, best observed validation c-index `0.5542244640605296`.
  - BLCA UNI OS + AB_MIL, GPU `5`, PID/session `3931478`; latest observed epoch `9`, best observed validation c-index `0.5372005044136192`.
  - NSCLC R50 benchmark, current active child `DS_MIL`, seed `2024`.
  - NSCLC UNI benchmark, current active child `RRT_MIL`, seed `2024`.
- Launched: BLCA UNI OS + MEAN_MIL, seed `2024`, after MIR_MIL_MT_V1 released GPU `3`.
  - PID/session leader: `3958001`
  - GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mean_seed2024_setsid_20260716_035214.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mean_seed2024_setsid_20260716_035214.status`
  - Command core: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv Model.backbone=MEAN_MIL Model.backbone_config=configs/MEAN_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
  - Verification: initialized, loaded survival cutpoints `[7.05, 13.34, 22.32]`, and entered `Train_Val_Test`.

## 2026-07-16 03:55 CST

- Task: BLCA UNI OS prognosis issue and retry
- Invalid run detected: BLCA UNI OS + RRT_MIL initial run `blca_uni_os_rrt_seed2024_setsid_20260716_034634` exited with `exit_code=1`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_rrt_seed2024_setsid_20260716_034634.log`
  - Failure mode: after early stopping at epoch `17`, final test checkpoint loading failed with state-dict key mismatch.
  - Root cause: BLCA UNI OS RRT and MIR were launched within the same minute using the same `Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS`, `Model Info=SURVIVAL_MIL`, and seed. The survival log directory naming collided, causing final-test checkpoint loading to see a checkpoint from a different backbone.
  - Consequence: this RRT run is invalid and must be discarded.
- Risk also identified for the original BLCA UNI OS + MIR_MIL run `blca_uni_os_mir_seed2024_setsid_20260716_034634`, because it shared the same collided output namespace with the invalid RRT run. It should be treated as invalid/discarded even if it later writes a status.
- Corrective action: re-run the affected BLCA UNI OS backbones with identical data split and model configs, changing only `Dataset.DATASET_NAME` to isolate the output namespace.
- Launched isolated BLCA UNI OS + RRT_MIL retry:
  - Dataset namespace: `TCGA_BLCA_UNI_OS_RRT`
  - PID/session leader: `3976795`
  - GPU: `1` via `CUDA_VISIBLE_DEVICES=1`
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_rrt_isolated_seed2024_setsid_20260716_035508.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_rrt_isolated_seed2024_setsid_20260716_035508.status`
  - Command core: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS_RRT Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv Model.backbone=RRT_MIL Model.backbone_config=configs/RRT_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
  - Verification: initialized, loaded survival cutpoints `[7.05, 13.34, 22.32]`, and entered `Train_Val_Test`.
- Launched isolated BLCA UNI OS + MIR_MIL retry:
  - Dataset namespace: `TCGA_BLCA_UNI_OS_MIR`
  - PID/session leader: `3978584`
  - GPU: `2` via `CUDA_VISIBLE_DEVICES=2`
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mir_isolated_seed2024_setsid_20260716_035535.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mir_isolated_seed2024_setsid_20260716_035535.status`
  - Command core: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS_MIR Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv Model.backbone=MIR_MIL Model.backbone_config=configs/MIR_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
  - Verification: initialized, loaded survival cutpoints `[7.05, 13.34, 22.32]`, and entered `Train_Val_Test`.
- Scheduling: BLCA UNI OS + MAX_MIL still pending. No GPU was available after launching the isolated RRT/MIR retries; active jobs remain NSCLC R50/UNI plus BLCA UNI OS AB/MEAN/RRT-isolated/MIR-isolated.

## 2026-07-16 03:58 CST

- Task: BLCA UNI OS prognosis and NSCLC benchmark status update.
- Completed: BLCA UNI OS + AB_MIL, seed `2024`, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_ab_seed2024_setsid_20260716_034833.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_ab_seed2024_setsid_20260716_034833.status`
  - Result from controller log: loaded `Best_EPOCH_8.pth`; final test c-index `0.6322055137844611`, CI `[0.5323975981030817, 0.7336397456067255]`, event_count `35`, sample_count `77`.
- Completed: BLCA UNI OS + MEAN_MIL, seed `2024`, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mean_seed2024_setsid_20260716_035214.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mean_seed2024_setsid_20260716_035214.status`
  - Result from controller log: loaded `Best_EPOCH_2.pth`; final test c-index `0.6322055137844611`, CI `[0.5321516965595571, 0.7263577836005778]`, event_count `35`, sample_count `77`.
- Launched isolated BLCA UNI OS + MAX_MIL, seed `2024`.
  - Dataset namespace: `TCGA_BLCA_UNI_OS_MAX`
  - PID/session leader: `3990563`
  - GPU: `3` via `CUDA_VISIBLE_DEVICES=3`
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_max_isolated_seed2024_setsid_20260716_035806.log`
  - Status file: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_max_isolated_seed2024_setsid_20260716_035806.status`
  - Command core: `train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options Dataset.DATASET_NAME=TCGA_BLCA_UNI_OS_MAX Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_UNI_OS_split.csv Model.backbone=MAX_MIL Model.backbone_config=configs/MAX_MIL.yaml Model.in_dim=1024 Model.max_instances=4096 Model.survival.patient_level=true`
  - Verification: process started on GPU `3`.
- BLCA UNI OS isolated retries remain running:
  - RRT_MIL isolated retry: latest observed validation c-index sequence includes best observed `0.5983606557377049`; no status file yet at this checkpoint.
  - MIR_MIL isolated retry: latest observed validation c-index sequence includes best observed `0.542875157629256`; no status file yet at this checkpoint.
- Completed: NSCLC R50 `DS_MIL`, seed `2024`.
  - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_R50/DS_MIL/time_2026-07-16-03-44_TCGA_NSCLC_LUAD_LUSC_R50_DS_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_R50_DS_MIL.csv`
  - Best epoch `29`; validation macro_auc `0.8760884796316686`; test acc `0.8537735849056604`, test bacc `0.8532763532763532`, test macro_auc `0.9180911680911681`.
  - R50 benchmark controller remains running and has moved to `TRANS_MIL`, seed `2024`.
- Completed: NSCLC UNI `RRT_MIL`, seed `2024`.
  - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_UNI/RRT_MIL/time_2026-07-16-03-49_TCGA_NSCLC_LUAD_LUSC_UNI_RRT_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_UNI_RRT_MIL.csv`
  - Best epoch `7`; validation macro_auc `0.9822840556500851`; test acc `0.9575471698113207`, test bacc `0.9577991452991452`, test macro_auc `0.9895833333333333`.
  - UNI benchmark controller remains running and has moved to `WIKG_MIL`, seed `2024`.

## 2026-07-16 04:03 CST

- Task: COADREAD WSI preparation and BLCA UNI OS status update.
- COADREAD local data audit:
  - Dataset directory: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD`
  - Existing content before action: metadata only, about `25M`; no local `.pt`, `.h5`, `.svs`, `.tif`, `.mrxs`, or `.ndpi` feature/WSI files found under the dataset tree.
  - Dearcat/CPathPatchFeature audit: no `coad`, `coadread`, `colon`, or `rectum` entries; only `README.md` matched `read`, so there are no ready COADREAD R50/UNI features in that HF dataset.
  - Free space check: `/data15/data15_5` has about `3.2T` available.
- Created a reproducible GDC public slide manifest for COADREAD Primary Tumor Diagnostic Slide SVS files.
  - Script added: `experiments/download_gdc_slides.py`
  - Manifest: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/manifests/tcga_coadread_primary_tumor_diagnostic_slides.tsv`
  - Query projects: `TCGA-COAD`, `TCGA-READ`
  - Files: `624`; total bytes `361840115368`; total size about `336.99 GiB`.
  - Manifest rows include GDC UUID, filename, md5, size, project_id, case_submitter_id, data_format, experimental_strategy, sample_type, and slide_submitter_id.
- Launched COADREAD GDC WSI download.
  - PID/session leader: `4011479`
  - Child Python PID at checkpoint: `4011484`
  - Command core: `experiments/download_gdc_slides.py download --manifest /data15/data15_5/fanhao/datasets/TCGA-COADREAD/manifests/tcga_coadread_primary_tumor_diagnostic_slides.tsv --output-dir /data15/data15_5/fanhao/datasets/TCGA-COADREAD/raw_gdc --workers 4 --retries 8 --timeout 600`
  - Log: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/logs/download_coadread_gdc_wsi_20260716_040244.log`
  - Status file: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/logs/download_coadread_gdc_wsi_20260716_040244.status`
  - Initial checkpoint: raw download directory has `4` files and about `33M`, consistent with four concurrent in-progress downloads; no final status file yet.
- Completed: BLCA UNI OS + RRT_MIL isolated retry, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_rrt_isolated_seed2024_setsid_20260716_035508.log`
  - Result from controller log: loaded `Best_EPOCH_9.pth`; final test c-index `0.606516290726817`, CI `[0.5079400996825885, 0.7082144602305283]`, event_count `35`, sample_count `77`.
- Completed: BLCA UNI OS + MIR_MIL isolated retry, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_mir_isolated_seed2024_setsid_20260716_035535.log`
  - Result from controller log: loaded `Best_EPOCH_12.pth`; final test c-index `0.5795739348370927`, CI `[0.47530779405779405, 0.6828781982716761]`, event_count `35`, sample_count `77`.
- Still running:
  - BLCA UNI OS + MAX_MIL isolated run, GPU `3`, PID/session `3990563`; latest observed validation c-index includes `0.5208070617906684`; no status file yet.
  - NSCLC R50 benchmark controller, active child `TRANS_MIL`, seed `2024`.
  - NSCLC UNI benchmark controller, active child `WIKG_MIL`, seed `2024`.

## 2026-07-16 04:07 CST

- Task: COADREAD prognosis metadata preparation and active benchmark update.
- Added and verified COADREAD prognosis preparation script:
  - Script: `experiments/prepare_tcga_coadread_pipeline.py`
  - Syntax check: `python -m py_compile experiments/prepare_tcga_coadread_pipeline.py` passed under the `mirmil` environment.
  - Purpose: join the existing patient-level prognosis assignments in `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA-COADREAD-PROGNOSIS` to the GDC diagnostic-slide manifest, emit WSI source CSVs, and materialize expected R50/UNI survival split CSVs in the same wide format used by BLCA prognosis.
  - Supports `--require-wsi` after GDC download completion and `--require-features` after R50/UNI feature extraction.
- Ran COADREAD preparation in template mode, without requiring completed WSI/features yet.
  - Command core: `experiments/prepare_tcga_coadread_pipeline.py --manifest /data15/data15_5/fanhao/datasets/TCGA-COADREAD/manifests/tcga_coadread_primary_tumor_diagnostic_slides.tsv --raw-dir /data15/data15_5/fanhao/datasets/TCGA-COADREAD/raw_gdc --wsi-root /data15/data15_5/fanhao/datasets/TCGA-COADREAD/WSI --feature-root /data15/data15_5/fanhao/datasets/TCGA-COADREAD/features --prognosis-dir /data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA-COADREAD-PROGNOSIS --output-dir /data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata`
  - Output manifest: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/tcga_coadread_pipeline_manifest.json`
  - Source manifest: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/tcga_coadread_source_manifest.csv`
  - WSI source CSV for patching: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/tcga_coadread_wsi_paths.csv`
  - Slide labels: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/tcga_coadread_slide_labels.csv`
  - Generated split examples:
    - `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA_COADREAD_PROGNOSIS_R50_OS_split.csv`
    - `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA_COADREAD_PROGNOSIS_UNI_OS_split.csv`
  - Manifest-level counts: `624` slides, `616` patients in the diagnostic-slide manifest.
  - Template-mode WSI status: `0` size-matched complete slides, `624` missing/partial, because download is still in progress.
  - Prognosis join counts:
    - OS: R50/UNI each `583` slide rows, `576` patients, `14` assignment patients unmatched to diagnostic slides.
    - PFS: R50/UNI each `583` slide rows, `576` patients, `14` unmatched.
    - DSS: R50/UNI each `564` slide rows, `557` patients, `12` unmatched.
    - DFS: R50/UNI each `224` slide rows, `219` patients, `4` unmatched.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `553M`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `0.54 GiB / 336.99 GiB`.
- Completed: BLCA UNI OS + MAX_MIL isolated run, status `exit_code=0`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS/controller_logs/blca_uni_os_max_isolated_seed2024_setsid_20260716_035806.log`
  - Result from controller log: loaded `Best_EPOCH_15.pth`; final test c-index `0.6578947368421053`, CI `[0.556477054978573, 0.7573505876216806]`, event_count `35`, sample_count `77`.
  - This is the best BLCA UNI OS result observed so far among RRT, MIR, MIR-MT, AB, MEAN, and MAX.
- Completed: NSCLC UNI `WIKG_MIL`, seed `2024`.
  - Best log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_UNI/WIKG_MIL/time_2026-07-16-03-54_TCGA_NSCLC_LUAD_LUSC_UNI_WIKG_MIL_seed_2024/Best_Log_seed2024_TCGA_NSCLC_LUAD_LUSC_UNI_WIKG_MIL.csv`
  - Best epoch `8`; validation macro_auc `0.9867881092983686`; test acc `0.9056603773584906`, test bacc `0.9042022792022792`, test macro_auc `0.9876246438746439`.
  - UNI benchmark controller remains running and has moved to `AC_MIL`, seed `2024`.
- Still running:
  - NSCLC R50 benchmark controller, active child `TRANS_MIL`, seed `2024`.
  - NSCLC UNI benchmark controller, active child `AC_MIL`, seed `2024`.
  - COADREAD WSI download.

## 2026-07-16 04:09 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Raw download directory: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/raw_gdc`
  - Current raw size: about `737M`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `0.72 GiB / 336.99 GiB`.
  - The four current partial downloads are the first four manifest slides; each has written `192,937,984` bytes at this checkpoint.
- NSCLC benchmark status:
  - R50 benchmark controller remains running; active child is still `TRANS_MIL`, seed `2024`; no new R50 best-log result beyond `DS_MIL` was present at this checkpoint.
  - UNI benchmark controller remains running; active child is `AC_MIL`, seed `2024`; no new UNI best-log result beyond `WIKG_MIL` was present at this checkpoint.
- No new GPU job was launched at this checkpoint because COADREAD WSI download has not completed and NSCLC controllers are already advancing their model queues.

## 2026-07-16 04:10 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `881M`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `0.86 GiB / 336.99 GiB`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running; active child is still `TRANS_MIL`, seed `2024`.
  - UNI benchmark controller remains running; active child is still `AC_MIL`, seed `2024`.
  - No new `Best_Log` result beyond the 04:07 checkpoint was present.
- No COADREAD patch/feature task was launched because the WSI download is still incomplete.

## 2026-07-16 04:11 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `1001M`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `0.98 GiB / 336.99 GiB`.
  - The four partial downloads continue to grow; no download error was present in the checked log tail.
- NSCLC benchmark status:
  - R50 benchmark controller remains running; active child is still `TRANS_MIL`, seed `2024`.
  - UNI benchmark controller remains running; active child is still `AC_MIL`, seed `2024`.
  - No new `Best_Log` result beyond the prior checkpoint was present.

## 2026-07-16 04:12 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `1.1G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `1.08 GiB / 336.99 GiB`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running; active child is still `TRANS_MIL`, seed `2024`.
  - UNI benchmark controller remains running; active child is still `AC_MIL`, seed `2024`.
  - No new `Best_Log` result beyond `NSCLC UNI WIKG_MIL` was present.

## 2026-07-16 04:13 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `1.2G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `1.20 GiB / 336.99 GiB`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running; active child is still `TRANS_MIL`, seed `2024`.
  - UNI benchmark controller remains running; active child is still `AC_MIL`, seed `2024`.
  - No new `Best_Log` result beyond the prior checkpoint was present.

## 2026-07-16 04:15 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `1.5G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `1.47 GiB / 336.99 GiB`.
  - The first four downloads are still partial; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child advanced to `RRT_MIL`, seed `2024`.
  - UNI benchmark controller remains running on GPU0; active child remains `AC_MIL`, seed `2024`.
  - Newly observed R50 `TRANS_MIL` result: best epoch `30`, val macro AUC `0.9500550495445901`, test acc `0.8773584905660378`, test bacc `0.8760683760683761`, test macro AUC `0.9522792022792023`.
  - Latest UNI completed result remains `WIKG_MIL` at best epoch `8`, val macro AUC `0.9867881092983686`, test acc `0.9056603773584906`, test bacc `0.9042022792022792`, test macro AUC `0.9876246438746439`; `AC_MIL` is still active.

## 2026-07-16 04:16 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `1.7G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `1.62 GiB / 336.99 GiB`.
  - The same four partial downloads continue to grow; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `RRT_MIL`, seed `2024`.
  - UNI benchmark controller remains running on GPU0; active child remains `AC_MIL`, seed `2024`.
  - No additional `Best_Log` result was present beyond the 04:15 checkpoint.

## 2026-07-16 04:17 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `1.8G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `1.73 GiB / 336.99 GiB`.
  - The same four partial downloads continue to grow; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `RRT_MIL`, seed `2024`.
  - UNI benchmark controller remains running on GPU0; active child advanced to `MO_MIL`, seed `2024`.
  - Newly observed UNI `AC_MIL` result: best epoch `8`, val macro AUC `0.9842358122310079`, test acc `0.9292452830188679`, test bacc `0.9282407407407407`, test macro AUC `0.991898148148148`.

## 2026-07-16 04:18 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `1.9G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `1.85 GiB / 336.99 GiB`.
  - The same four partial downloads continue to grow; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `RRT_MIL`, seed `2024`.
  - UNI benchmark controller remains running on GPU0; active child remains `MO_MIL`, seed `2024`.
  - No additional `Best_Log` result was present beyond the 04:17 checkpoint.

## 2026-07-16 04:19 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `2.0G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `1.95 GiB / 336.99 GiB`.
  - The same four partial downloads continue to grow; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `RRT_MIL`, seed `2024`.
  - UNI benchmark controller remains running on GPU0; active child remains `MO_MIL`, seed `2024`.
  - No additional `Best_Log` result was present beyond the 04:17 checkpoint.

## 2026-07-16 04:20 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `2.1G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `2.04 GiB / 336.99 GiB`.
  - The same four partial downloads continue to grow; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `RRT_MIL`, seed `2024`.
  - UNI benchmark controller remains running on GPU0; active child remains `MO_MIL`, seed `2024`.
  - No additional `Best_Log` result was present beyond the 04:17 checkpoint.

## 2026-07-16 04:21 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `2.2G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `2.15 GiB / 336.99 GiB`.
  - The same four partial downloads continue to grow; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child advanced to `WIKG_MIL`, seed `2024`.
  - UNI benchmark controller remains running on GPU0; active child remains `MO_MIL`, seed `2024`.
  - Newly observed R50 `RRT_MIL` result: best epoch `12`, val macro AUC `0.9373436092483236`, test acc `0.8679245283018868`, test bacc `0.8678774928774928`, test macro AUC `0.93883547008547`.

## 2026-07-16 04:22 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `2.3G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `2.27 GiB / 336.99 GiB`.
  - The same four partial downloads continue to grow; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - UNI benchmark controller remains running on GPU0; active child remains `MO_MIL`, seed `2024`.
  - No additional `Best_Log` result was present beyond the 04:21 checkpoint.

## 2026-07-16 04:23 CST

- Task: active long-run status check and UNI benchmark recovery.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `2.4G`.
  - Size-matched complete files: `0 / 624`; partial files: `4`; missing files: `620`.
  - Byte progress estimate: `2.38 GiB / 336.99 GiB`.
  - The same four partial downloads continue to grow; no completion/status file exists yet.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - Original UNI benchmark controller exited with `exit_code=1` at `MAMBA2D_MIL`.
  - Newly observed UNI `MO_MIL` result before the controller exit: best epoch `4`, val macro AUC `0.9846862175958362`, test acc `0.9433962264150944`, test bacc `0.9437321937321937`, test macro AUC `0.9910078347578348`.
  - Failure reason for UNI `MAMBA2D_MIL`: `coordinate_dim=2 requires coordinates` for UNI `.pt` files; the downloaded NSCLC UNI features do not provide companion coordinates required by this 2D spatial baseline.
  - Recovery action: launched a separate UNI remaining benchmark for `MIR_MIL` and `MIR_MIL_MT_V1` only, since these are still required by the handoff and do not require coordinates.
    - PID: `4078329`.
    - GPU: `CUDA_VISIBLE_DEVICES=1`.
    - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_remaining_gpu1_20260716_042313.log`.
    - Status: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_remaining_gpu1_20260716_042313.status`.
    - Command: `experiments/run_benchmark.py --split /data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_UNI_split.csv --dataset-name TCGA_NSCLC_LUAD_LUSC_UNI --num-classes 2 --log-root /data15/data15_5/fanhao/experiments/MIRMIL_NSCLC --models MIR_MIL MIR_MIL_MT_V1 --seeds 2024 2025 2026 --epochs 30 --device 0 --num-workers 4 --in-dim 1024 --max-instances 4096 --feature uni --protocol nsclc_luad_lusc --split-id uni_v1`.

## 2026-07-16 04:24 CST

- Task: corrected UNI remaining benchmark launch.
- The first UNI recovery launch recorded at 04:23 did not enter training: PID `4078329` exited immediately, the log file stayed empty, and no status file was written.
- Relaunched UNI remaining benchmark with a direct `setsid bash -c` wrapper.
  - PID: `4080568`.
  - GPU: `CUDA_VISIBLE_DEVICES=1`.
  - Models: `MIR_MIL`, `MIR_MIL_MT_V1`.
  - Seeds: `2024`, `2025`, `2026`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_remaining_gpu1_20260716_042417.log`.
  - Status: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_remaining_gpu1_20260716_042417.status`.
  - Verified the log started correctly and printed the expected benchmark manifest for `TCGA_NSCLC_LUAD_LUSC_UNI`.

## 2026-07-16 04:25 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `2.7G`.
  - Size-matched complete files: `1 / 624`; partial files: `3`; missing files: `620`.
  - Byte progress estimate: `2.67 GiB / 336.99 GiB`.
  - First completed file: `TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - Corrected UNI remaining controller remains running on GPU1.
    - PID/session leader: `4080568`; child controller PID: `4080570`.
    - Active model: `MIR_MIL`, seed `2024`.
    - Log confirms training entered the loop and reached at least epoch `2`; no `Best_Log` result yet for the resumed MIR models.

## 2026-07-16 04:26 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `2.8G`.
  - Size-matched complete files: `1 / 624`; partial files: `4`; missing files: `619`.
  - Byte progress estimate: `2.78 GiB / 336.99 GiB`.
  - Completed file remains `TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs`; a new fourth partial download has started.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - Corrected UNI remaining controller remains running on GPU1.
    - PID/session leader: `4080568`; child controller PID: `4080570`.
    - Active model: `MIR_MIL`, seed `2024`.
    - Log confirms training reached at least epoch `5`; no new `Best_Log` result yet for the resumed MIR models.

## 2026-07-16 04:27 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `3.0G`.
  - Size-matched complete files: `1 / 624`; partial files: `4`; missing files: `619`.
  - Byte progress estimate: `2.91 GiB / 336.99 GiB`.
  - Completed file remains `TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - Corrected UNI remaining controller remains running on GPU1.
    - PID/session leader: `4080568`; child controller PID: `4080570`.
    - Active model: `MIR_MIL`, seed `2024`.
    - Log confirms training reached at least epoch `8`; no new `Best_Log` result yet for the resumed MIR models.

## 2026-07-16 04:28 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `3.1G`.
  - Size-matched complete files: `1 / 624`; partial files: `4`; missing files: `619`.
  - Byte progress estimate: `3.02 GiB / 336.99 GiB`.
  - Completed file remains `TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - Corrected UNI remaining controller remains running on GPU1.
    - PID/session leader: `4080568`; child controller PID: `4080570`.
    - Active model: `MIR_MIL`, seed `2024`.
    - Log confirms training reached at least epoch `10`; no new `Best_Log` result yet for the resumed MIR models.

## 2026-07-16 04:29 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `3.2G`.
  - Size-matched complete files: `1 / 624`; partial files: `4`; missing files: `619`.
  - Byte progress estimate: `3.13 GiB / 336.99 GiB`.
  - Completed file remains `TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - Corrected UNI remaining controller remains running on GPU1.
    - PID/session leader: `4080568`; child controller PID: `4080570`.
    - Active model: `MIR_MIL`, seed `2024`.
    - Log confirms training reached at least epoch `13`; no new `Best_Log` result yet for the resumed MIR models.

## 2026-07-16 04:30 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `3.3G`.
  - Size-matched complete files: `1 / 624`; partial files: `4`; missing files: `619`.
  - Byte progress estimate: `3.22 GiB / 336.99 GiB`.
  - Completed file remains `TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - Corrected UNI remaining controller remains running on GPU1.
    - PID/session leader: `4080568`; child controller PID: `4080570`.
    - Newly observed UNI `MIR_MIL` result: best epoch `6`, val macro AUC `0.9845861275147633`, test acc `0.9198113207547169`, test bacc `0.9211182336182336`, test macro AUC `0.9894943019943021`.
    - The UNI recovery task advanced to `MIR_MIL_MT_V1`, seed `2024`; log confirms it reached at least epoch `1`.

## 2026-07-16 04:33 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `3.7G`.
  - Size-matched complete files: `2 / 624`; partial files: `4`; missing files: `618`.
  - Byte progress estimate: `3.63 GiB / 336.99 GiB`.
  - Newly complete file: `TCGA-4N-A93T-01Z-00-DX1.82E240B1-22C3-46E3-891F-0DCE35C43F8B.svs`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6; active child remains `WIKG_MIL`, seed `2024`.
  - Corrected UNI remaining controller remains running on GPU1.
    - PID/session leader: `4080568`; child controller PID: `4080570`.
    - Active model: `MIR_MIL_MT_V1`, seed `2024`.
    - Log/process state confirms `MIR_MIL_MT_V1` is actively training; no `Best_Log` yet for `MIR_MIL_MT_V1`.

## 2026-07-16 04:35 CST

- Task: prognosis task audit before launching anything new.
- KIRC/BLCA prognosis jobs do not need duplicate launches at this checkpoint.
  - Found `36` prognosis `Best_Log*.csv` files under `/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS`.
  - Controller status files show the corrected BLCA/KIRC prognosis jobs have finished with `exit_code=0`.
  - Previously failed BLCA UNI OS attempts were documented and replaced by isolated dataset-name reruns to avoid log/checkpoint collisions.
  - Current process table shows no active `MIRMIL_PROGNOSIS`/`SURVIVAL_MIL` jobs.
- Existing completed prognosis coverage:
  - BLCA: R50 OS and UNI OS with RRT/MIR/MIR-MT plus AB/MEAN/MAX baselines where applicable.
  - KIRC: R50/UNI OS and PFS with RRT/MIR/MIR-MT plus AB/MEAN/MAX baselines.
- Decision: do not start more BLCA/KIRC prognosis jobs now; continue monitoring active NSCLC and COADREAD tasks.

## 2026-07-16 04:36 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `4.0G`.
  - Size-matched complete files: `2 / 624`; partial files: `4`; missing files: `618`.
  - Byte progress estimate: `3.94 GiB / 336.99 GiB`.
  - Latest complete files remain:
    - `TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs`
    - `TCGA-4N-A93T-01Z-00-DX1.82E240B1-22C3-46E3-891F-0DCE35C43F8B.svs`
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Active model: `WIKG_MIL`, seed `2024`.
    - Log shows training reached at least epoch `22`; no `Best_Log` yet for R50 `WIKG_MIL`.
  - Corrected UNI remaining controller remains running on GPU1.
    - PID/session leader: `4080568`; child controller PID: `4080570`.
    - Newly observed MT-config result for seed `2024`: best epoch `5`, val macro AUC `0.9885897307576819`, test acc `0.9528301886792453`, test bacc `0.9524572649572649`, test macro AUC `0.988425925925926`.
    - Note: the result directory/model label still appears as `MIR_MIL` even for the MT config launch; interpret this row by controller/config context (`configs/releases/MIR_MIL_MT_V1.yaml`), not the printed `Model Info`.
    - Controller advanced to `MIR_MIL`, seed `2025`; log confirms it reached at least epoch `2`.

## 2026-07-16 04:37 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and still making byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `4.1G`.
  - Size-matched complete files: `2 / 624`; partial files: `4`; missing files: `618`.
  - Byte progress estimate: `4.08 GiB / 336.99 GiB`.
  - No new complete file since the 04:36 check; partial file sizes increased.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Active model: `WIKG_MIL`, seed `2024`.
    - Log shows training reached epoch `24`; no R50 `WIKG_MIL` `Best_Log` yet.
  - Corrected UNI remaining controller remains running on GPU1.
    - Active model: `MIR_MIL`, seed `2025`.
    - Log shows training reached epoch `6`; no seed `2025` `Best_Log` yet.
- Decision: no recovery action needed at this checkpoint; continue polling.

## 2026-07-16 04:38 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `4.2G`.
  - Size-matched complete files: `2 / 624`; partial files: `4`; missing files: `618`.
  - Byte progress estimate: `4.19 GiB / 336.99 GiB`.
  - No new complete file since the 04:37 check; partial file sizes increased.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Active model: `WIKG_MIL`, seed `2024`.
    - Log shows training reached epoch `25`; no R50 `WIKG_MIL` `Best_Log` yet.
  - Corrected UNI remaining controller remains running on GPU1.
    - Active model: `MIR_MIL`, seed `2025`.
    - Log shows training reached epoch `9`; no seed `2025` `Best_Log` yet.
- Decision: no recovery action needed at this checkpoint; continue polling.

## 2026-07-16 04:39 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `4.4G`.
  - Size-matched complete files: `2 / 624`; partial files: `4`; missing files: `618`.
  - Byte progress estimate: `4.32 GiB / 336.99 GiB`.
  - No new complete file since the 04:38 check; partial file sizes increased.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Active model: `WIKG_MIL`, seed `2024`.
    - Log shows training reached epoch `27`; no R50 `WIKG_MIL` `Best_Log` yet.
  - Corrected UNI remaining controller remains running on GPU1.
    - Newly observed UNI `MIR_MIL` seed `2025` result: best epoch `3`, val macro AUC `0.9865879291362225`, test acc `0.9481132075471698`, test bacc `0.9481837606837606`, test macro AUC `0.9882478632478632`.
    - Controller advanced to `MIR_MIL_MT_V1`, seed `2025`.
    - Note: MT launch still prints `Model Info:MIR_MIL`; interpret by controller/config context (`configs/releases/MIR_MIL_MT_V1.yaml`).
- Decision: no recovery action needed at this checkpoint; continue polling.

## 2026-07-16 04:40 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `4.5G`.
  - Size-matched complete files: `2 / 624`; partial files: `4`; missing files: `618`.
  - Byte progress estimate: `4.45 GiB / 336.99 GiB`.
  - No new complete file since the 04:39 check; partial file sizes increased.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Active model: `WIKG_MIL`, seed `2024`.
    - Log shows training reached epoch `29`; no R50 `WIKG_MIL` `Best_Log` yet.
  - Corrected UNI remaining controller remains running on GPU1.
    - Active model: `MIR_MIL_MT_V1`, seed `2025`.
    - Log shows training reached epoch `3`; no seed `2025` MT `Best_Log` yet.
    - Note: MT launch still prints `Model Info:MIR_MIL`; interpret by controller/config context (`configs/releases/MIR_MIL_MT_V1.yaml`).
- Decision: no recovery action needed at this checkpoint; continue polling.

## 2026-07-16 04:41 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Current raw size: about `4.6G`.
  - Size-matched complete files: `2 / 624`; partial files: `4`; missing files: `618`.
  - Byte progress estimate: `4.58 GiB / 336.99 GiB`.
  - No new complete file since the 04:40 check; partial file sizes increased.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Newly observed R50 `WIKG_MIL` seed `2024` result: best epoch `30`, val macro AUC `0.9145230707636873`, test acc `0.8537735849056604`, test bacc `0.8554131054131053`, test macro AUC `0.9484508547008547`.
    - Controller advanced to `AC_MIL`, seed `2024`.
  - Corrected UNI remaining controller remains running on GPU1.
    - Active model: `MIR_MIL_MT_V1`, seed `2025`.
    - Log shows training reached epoch `5`; no seed `2025` MT `Best_Log` yet.
    - Note: MT launch still prints `Model Info:MIR_MIL`; interpret by controller/config context (`configs/releases/MIR_MIL_MT_V1.yaml`).
- Decision: no recovery action needed at this checkpoint; continue polling.

## 2026-07-16 04:45 CST

- Task: active long-run status check after server handoff.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `3 / 624`; partial files: `4`; missing files: `617`.
  - Byte progress estimate: `4.98 GiB / 336.99 GiB`.
  - Complete files now include:
    - `TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs`
    - `TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs`
    - `TCGA-4N-A93T-01Z-00-DX1.82E240B1-22C3-46E3-891F-0DCE35C43F8B.svs`
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Active model: `AC_MIL`, seed `2024`.
    - Log shows training reached epoch `6`; no R50 `AC_MIL` `Best_Log` yet.
  - Corrected UNI remaining controller remains running on GPU1.
    - Newly observed UNI MT-config seed `2025` result: best epoch `2`, val macro AUC `0.9857872084876388`, test acc `0.9386792452830188`, test bacc `0.9387464387464388`, test macro AUC `0.9886039886039887`.
    - Controller advanced to `MIR_MIL`, seed `2026`; log shows training reached epoch `5`.
    - Note: MT launch still prints `Model Info:MIR_MIL`; interpret MT rows by controller/config context (`configs/releases/MIR_MIL_MT_V1.yaml`).
- Decision: no recovery action needed at this checkpoint; continue polling active long tasks.

## 2026-07-16 04:48 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `3 / 624`; partial files: `4`; missing files: `617`.
  - Byte progress estimate: `5.27 GiB / 336.99 GiB`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Active model: `AC_MIL`, seed `2024`.
    - Log shows training reached epoch `10`; no R50 `AC_MIL` `Best_Log` yet.
  - Corrected UNI remaining controller remains running on GPU1.
    - Newly observed UNI `MIR_MIL` seed `2026` result: best epoch `2`, val macro AUC `0.9781803623260934`, test acc `0.9386792452830188`, test bacc `0.9391025641025641`, test macro AUC `0.9912749287749287`.
    - Controller advanced to `MIR_MIL_MT_V1`, seed `2026`; log shows training reached epoch `1`.
    - Note: MT launch still prints `Model Info:MIR_MIL`; interpret MT rows by controller/config context (`configs/releases/MIR_MIL_MT_V1.yaml`).
- Decision: no recovery action needed at this checkpoint; continue polling active long tasks.

## 2026-07-16 04:52 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `3 / 624`; partial files: `4`; missing files: `617`.
  - Byte progress estimate: `5.72 GiB / 336.99 GiB`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Active model: `AC_MIL`, seed `2024`.
    - Log shows training reached epoch `17`; no R50 `AC_MIL` `Best_Log` yet.
  - Corrected UNI remaining controller remains running on GPU1.
    - Active model: `MIR_MIL_MT_V1`, seed `2026`.
    - Log shows training reached epoch `11`; no seed `2026` MT `Best_Log` yet.
    - Note: MT launch still prints `Model Info:MIR_MIL`; interpret MT rows by controller/config context (`configs/releases/MIR_MIL_MT_V1.yaml`).
- Decision: no recovery action needed at this checkpoint; continue polling active long tasks.

## 2026-07-16 04:55 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `4 / 624`; partial files: `4`; missing files: `616`.
  - Byte progress estimate: `6.15 GiB / 336.99 GiB`.
  - Newly complete file:
    - `TCGA-4N-A93T-01Z-00-DX2.875E7F95-A6D4-4BEB-A331-F9D8080898C2.svs`
- NSCLC benchmark status:
  - Corrected UNI remaining controller finished successfully.
    - Status: `nsclc_uni_remaining_gpu1_20260716_042417.status` reports `exit_code=0`.
    - Newly observed UNI MT-config seed `2026` result: best epoch `11`, val macro AUC `0.9794815333800421`, test acc `0.9622641509433962`, test bacc `0.9622507122507122`, test macro AUC `0.9866452991452991`.
    - This completes the corrected UNI remaining MIRMIL/MIRMIL-MT seed set launched after the original UNI controller failed at coordinate-dependent `MAMBA2D_MIL`.
  - R50 benchmark controller remains running on GPU6.
    - Active model: `AC_MIL`, seed `2024`.
    - Log shows training reached epoch `23`; no R50 `AC_MIL` `Best_Log` yet.
- Decision: no UNI recovery needed; continue monitoring R50 controller and COADREAD download.

## 2026-07-16 05:00 CST

- Task: active long-run status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `4 / 624`; partial files: `4`; missing files: `616`.
  - Byte progress estimate: `6.65 GiB / 336.99 GiB`.
- NSCLC benchmark status:
  - R50 benchmark controller remains running on GPU6.
    - Newly observed R50 `AC_MIL` seed `2024` result: best epoch `25`, val macro AUC `0.9406465819237314`, test acc `0.8962264150943396`, test bacc `0.8952991452991452`, test macro AUC `0.9599358974358974`.
    - Controller advanced to `MO_MIL`, seed `2024`; current active training PID is `4180570`.
    - R50 `MAMBA2D_MIL` has not started yet at this checkpoint.
- Decision: no recovery action needed; continue monitoring R50 until `MAMBA2D_MIL` either runs or fails due missing coordinates.

## 2026-07-16 05:11 CST

- Task: handoff continuation status check.
- COADREAD GDC WSI download remains active and continues byte-level progress.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Command:
    `experiments/download_gdc_slides.py download --manifest /data15/data15_5/fanhao/datasets/TCGA-COADREAD/manifests/tcga_coadread_primary_tumor_diagnostic_slides.tsv --output-dir /data15/data15_5/fanhao/datasets/TCGA-COADREAD/raw_gdc --workers 4 --retries 8 --timeout 600`
  - Log: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/logs/download_coadread_gdc_wsi_20260716_040244.log`.
  - Status file: `/data15/data15_5/fanhao/datasets/TCGA-COADREAD/logs/download_coadread_gdc_wsi_20260716_040244.status` not present yet, so the download is still running.
  - Size-matched complete files: `4 / 624`; partial files: `4`; missing files: `616`.
  - Byte progress estimate: `8.01 GiB / 336.99 GiB`.
  - Current partial files:
    - `TCGA-4T-AA8H-01Z-00-DX1.A46C759C-74A2-4724-B6B5-DECA0D16E029.svs`: `1442840576 / 2145045839`
    - `TCGA-5M-AAT4-01Z-00-DX1.725C46CA-9354-43AC-AA81-3E5A66354D6B.svs`: `1191182336 / 2385119307`
    - `TCGA-5M-AAT5-01Z-00-DX1.548E7CEB-48FB-4037-A616-39AB025E7A73.svs`: `914358272 / 2497921503`
    - `TCGA-5M-AAT6-01Z-00-DX1.8834C952-14E3-4491-8156-52FC917BB014.svs`: `553648128 / 1766078883`
- NSCLC feature availability:
  - `patches`: `1046` files.
  - `r50`: `1039` files.
  - `uni`: `1052` files.
- NSCLC R50 original controller has finished with failure status.
  - Status: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_gpu6_manual_20260716_030702.status` reports `exit_code=1`.
  - Completed R50 non-coordinate baselines before failure: `9` Best_Log files (`AB_MIL`, `CLAM_SB_MIL`, `CLAM_MB_MIL`, `DS_MIL`, `TRANS_MIL`, `RRT_MIL`, `WIKG_MIL`, `AC_MIL`, `MO_MIL`) for seed `2024`.
  - Newly verified R50 `MO_MIL` seed `2024` result: best epoch `14`, val macro AUC `0.9350415373836453`, test acc `0.8962264150943396`, test bacc `0.8961894586894588`, test macro AUC `0.9505876068376069`.
  - Failure reason: `MAMBA2D_MIL` requires coordinates (`coordinate_dim=2`) but NSCLC R50 `.pt` features do not contain coordinates, e.g. `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/nsclc/r50/pt_files/TCGA-75-5125-01Z-00-DX1.2E419CAF-E248-4959-951E-498D8054433E.pt`.
  - Decision: document/skip R50 `MAMBA2D_MIL` for this feature set and relaunch remaining non-coordinate R50 models: `MIR_MIL` and `MIR_MIL_MT_V1` for seeds `2024`, `2025`, `2026`.
- NSCLC UNI corrected remaining controller has finished successfully.
  - Status: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_uni_remaining_gpu1_20260716_042417.status` reports `exit_code=0`.
  - UNI has `15` Best_Log files: original non-coordinate baselines seed `2024`, plus corrected `MIR_MIL` and `MIR_MIL_MT_V1` seeds `2024`, `2025`, `2026`.

## 2026-07-16 05:15 CST

- Task: launch corrected NSCLC R50 remaining non-coordinate MIRMIL runs.
- Reason: original R50 benchmark controller stopped at coordinate-dependent `MAMBA2D_MIL`; `MIR_MIL` and `MIR_MIL_MT_V1` had not yet run for R50.
- Failed launch attempts:
  - Two `nohup bash -lc` / direct `nohup env` attempts produced empty logs and did not persist; retained as empty controller logs for audit:
    - `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_20260716_051233.log`
    - `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_20260716_051315.log`
    - `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_direct_20260716_051411.log`
  - A foreground `timeout 20` diagnostic confirmed the R50 `MIR_MIL` training entrypoint is valid and reaches epoch progress; it was intentionally interrupted before completion.
- Active corrected launch:
  - tmux session: `nsclc_r50_remaining_20260716_051530`.
  - GPU: `6` via `CUDA_VISIBLE_DEVICES=6`, with internal `General.device=0`.
  - run_benchmark PID: `4247`.
  - Current train_mil PID at launch verification: `4250`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.log`.
  - Status: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.status`.
  - Split: `/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_R50_split.csv`.
  - Command:
    `CUDA_VISIBLE_DEVICES=6 /data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python -u experiments/run_benchmark.py --split /data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_R50_split.csv --dataset-name TCGA_NSCLC_LUAD_LUSC_R50 --num-classes 2 --log-root /data15/data15_5/fanhao/experiments/MIRMIL_NSCLC --models MIR_MIL MIR_MIL_MT_V1 --seeds 2024 2025 2026 --epochs 30 --device 0 --num-workers 4 --in-dim 1024 --max-instances 4096 --feature r50 --protocol nsclc_luad_lusc --split-id r50_v1`
  - Verified state: `MIR_MIL` seed `2024` started; log prints `MIL-yaml path: configs/MIR_MIL.yaml`, `Dataset Info:TCGA_NSCLC_LUAD_LUSC_R50`, `Device Info:CUDA:0`, and epoch progress began.

## 2026-07-16 05:17 CST

- Task: post-launch verification checkpoint.
- NSCLC R50 remaining run is active.
  - tmux session: `nsclc_r50_remaining_20260716_051530`.
  - run_benchmark PID: `4247`.
  - Active `train_mil.py`: `MIR_MIL`, seed `2024`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.log`.
  - Observed progress: reached epoch `3 / 30`; epoch 3 val macro AUC `0.7721949754779301`.
  - GPU6 memory at checkpoint: about `325 MiB`, utilization about `20%`.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `4 / 624`; partial files: `4`; missing files: `616`.
  - Byte progress estimate: `8.57 GiB / 336.99 GiB`.
  - Status file still absent, so the download has not completed yet.

## 2026-07-16 05:26 CST

- Task: NSCLC R50 remaining progress checkpoint.
- NSCLC R50 `MIR_MIL` seed `2024` completed successfully under the corrected tmux controller.
  - Run directory: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_R50/MIR_MIL/time_2026-07-16-05-15_TCGA_NSCLC_LUAD_LUSC_R50_MIR_MIL_seed_2024`.
  - Best epoch: `28`.
  - Val macro AUC: `0.9314382944650186`.
  - Test acc: `0.910377358490566`.
  - Test bacc: `0.9102564102564102`.
  - Test macro AUC: `0.9631410256410257`.
- Corrected R50 controller advanced to `MIR_MIL_MT_V1` seed `2024`.
  - Active `train_mil.py` PID at checkpoint: `22892`.
  - Note: MT config still prints `Model Info:MIR_MIL`; interpret this run by controller command/config context (`configs/releases/MIR_MIL_MT_V1.yaml` and `General.experiment_variant=MIR_MIL_MT_V1`).
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.log`.
  - Status file not present yet, so the corrected R50 remaining controller is still running.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `4 / 624`; partial files: `4`; missing files: `616`.
  - Byte progress estimate: `9.67 GiB / 336.99 GiB`.
  - Status file still absent, so the download has not completed yet.

## 2026-07-16 05:34 CST

- Task: active long-run checkpoint.
- NSCLC R50 corrected remaining controller remains active.
  - tmux session: `nsclc_r50_remaining_20260716_051530`.
  - run_benchmark PID: `4247`.
  - Active run: `MIR_MIL_MT_V1`, seed `2024`.
  - Active `train_mil.py` PID at checkpoint: `22892`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.log`.
  - Status file still absent, so the controller is still running.
  - Latest observed MT progress before this checkpoint: epoch `20 / 30`; no MT Best_Log yet.
  - Current completed corrected R50 remaining Best_Log count: `1` (`MIR_MIL` seed `2024`).
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `4 / 624`; partial files: `4`; missing files: `616`.
  - Byte progress estimate: `10.63 GiB / 336.99 GiB`.
  - One partial (`TCGA-4T-AA8H-01Z-00-DX1...`) is nearly complete at `2139095040 / 2145045839` bytes.
  - Status file still absent, so the download has not completed yet.

## 2026-07-16 05:38 CST

- Task: NSCLC R50 MT seed `2024` result checkpoint.
- NSCLC R50 `MIR_MIL_MT_V1` seed `2024` completed successfully under the corrected tmux controller.
  - Run directory: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_R50/MIR_MIL/time_2026-07-16-05-25_TCGA_NSCLC_LUAD_LUSC_R50_MIR_MIL_seed_2024`.
  - Important naming note: the run directory/model line prints `MIR_MIL`, but this run is `MIR_MIL_MT_V1` by controller order and config context (`configs/releases/MIR_MIL_MT_V1.yaml`, `General.experiment_variant=MIR_MIL_MT_V1`).
  - Best epoch: `26`.
  - Val macro AUC: `0.9294364928435592`.
  - Test acc: `0.9009433962264151`.
  - Test bacc: `0.900997150997151`.
  - Test macro AUC: `0.9594017094017093`.
- Corrected R50 controller advanced to `MIR_MIL` seed `2025`.
  - Active `train_mil.py` PID at checkpoint: `41845`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.log`.
  - Status file not present yet, so the corrected R50 remaining controller is still running.
- COADREAD GDC WSI download remains active.
  - Size-matched complete files: `5 / 624`; partial files: `4`; missing files: `615`.
  - Byte progress estimate: `10.74 GiB / 336.99 GiB`.
  - Newly complete file since previous checkpoint:
    - `TCGA-4T-AA8H-01Z-00-DX1.A46C759C-74A2-4724-B6B5-DECA0D16E029.svs`

## 2026-07-16 05:43 CST

- Task: active long-run checkpoint.
- NSCLC R50 corrected remaining controller remains active.
  - tmux session: `nsclc_r50_remaining_20260716_051530`.
  - run_benchmark PID: `4247`.
  - Active run: `MIR_MIL`, seed `2025`.
  - Active `train_mil.py` PID at checkpoint: `41845`.
  - Latest observed progress: epoch `13 / 30`; no seed `2025` Best_Log yet.
  - Current completed corrected R50 remaining Best_Log count: `2`:
    - `MIR_MIL` seed `2024`
    - `MIR_MIL_MT_V1` seed `2024` (directory/model name prints `MIR_MIL`; identify by config/controller context)
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.log`.
  - Status file not present yet, so the corrected R50 remaining controller is still running.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `5 / 624`; partial files: `4`; missing files: `615`.
  - Byte progress estimate: `11.28 GiB / 336.99 GiB`.
  - Status file still absent, so the download has not completed yet.

## 2026-07-16 05:49 CST

- Task: NSCLC R50 seed `2025` progress checkpoint.
- NSCLC R50 `MIR_MIL` seed `2025` completed successfully under the corrected tmux controller.
  - Run directory: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/TCGA_NSCLC_LUAD_LUSC_R50/MIR_MIL/time_2026-07-16-05-38_TCGA_NSCLC_LUAD_LUSC_R50_MIR_MIL_seed_2025`.
  - Best epoch: `21`.
  - Val macro AUC: `0.9337403663296968`.
  - Test acc: `0.9150943396226415`.
  - Test bacc: `0.9145299145299145`.
  - Test macro AUC: `0.9632300569800569`.
- Corrected R50 controller advanced to `MIR_MIL_MT_V1` seed `2025`.
  - Active `train_mil.py` PID at checkpoint: `60100`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.log`.
  - Status file not present yet, so the corrected R50 remaining controller is still running.
  - Current completed corrected R50 remaining Best_Log count: `3` of expected `6`.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `5 / 624`; partial files: `4`; missing files: `615`.
  - Byte progress estimate: `12.35 GiB / 336.99 GiB`.
  - Near-complete partials:
    - `TCGA-5M-AAT4-01Z-00-DX1...`: `2365587456 / 2385119307`
    - `TCGA-5M-AAT6-01Z-00-DX1...`: `1702887424 / 1766078883`

## 2026-07-16 05:53 CST

- Task: active long-run checkpoint.
- NSCLC R50 corrected remaining controller remains active.
  - tmux session: `nsclc_r50_remaining_20260716_051530`.
  - run_benchmark PID: `4247`.
  - Active run: `MIR_MIL_MT_V1`, seed `2025`.
  - Latest observed progress: epoch `10 / 30`; no seed `2025` MT Best_Log yet.
  - Current completed corrected R50 remaining Best_Log count: `3` of expected `6`.
  - Log: `/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC/controller_logs/nsclc_r50_remaining_gpu6_tmux_20260716_051530.log`.
  - Status file not present yet, so the corrected R50 remaining controller is still running.
- COADREAD GDC WSI download remains active.
  - PID/session leader: `4011479`; child Python PID: `4011484`.
  - Size-matched complete files: `7 / 624`; partial files: `4`; missing files: `613`.
  - Byte progress estimate: `12.79 GiB / 336.99 GiB`.
  - Newly complete files since previous checkpoint:
    - `TCGA-5M-AAT4-01Z-00-DX1.725C46CA-9354-43AC-AA81-3E5A66354D6B.svs`
    - `TCGA-5M-AAT6-01Z-00-DX1.8834C952-14E3-4491-8156-52FC917BB014.svs`
  - Status file still absent, so the download has not completed yet.
