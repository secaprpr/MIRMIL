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
