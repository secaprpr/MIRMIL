# 双 Codex 服务器交接文件

本文档给即将在服务器上运行的 Codex CLI 使用。当前写入方在 Mac，本文件落点应为：

`/data15/data15_5/fanhao/projects/MIRMIL/CODEX_SERVER_HANDOFF.md`

## 固定上下文

- 项目目录：`/data15/data15_5/fanhao/projects/MIRMIL`
- 数据根目录：`/data15/data15_5/fanhao/datasets`
- Python：`/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python`
- 进入环境：`mamba activate mirmil`
- NSCLC 下载进程：`PID=3566032`
- NSCLC 下载日志：`/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_r50_uni_patches_20260716_002213.log`
- NSCLC 数据目录：`/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/nsclc`
- COADREAD 预后 GT：`/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA-COADREAD-PROGNOSIS`
- KIRC 预后 GT：`/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS`
- BLCA 预后 GT：`/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata`

不要执行 `git reset --hard`、`git checkout -- .` 或删除用户未说明的文件。远端项目目录可能有大量未跟踪实验脚本和临时文件，视为用户资产。

## 总目标

1. 每半小时轮询 NSCLC 下载，等待 r50、uni、patches 全部下载完。
2. NSCLC 全部下载完毕后，启动所有 baseline 以及两个 MIRMIL 版本在 UNI 和 R50 特征上的实验。
3. NSCLC 下载完后，数据准备/下载任务可以和 GPU 训练任务并行推进：一边跑 NSCLC 实验，一边处理 COADREAD。用户说“我已经提前下载好了特征并放在 dataset 目录”，先查特征是否已存在；如果只有 WSI 或缺特征，再下载/处理 WSI。
4. 如果实际需要下载 COADREAD WSI，则启动 patches 生成和 UNI/R50 特征提取。
5. 对 COADREAD、KIRC、BLCA 启动预后任务，不是分期任务。

## 0. 开始前检查

```bash
cd /data15/data15_5/fanhao/projects/MIRMIL
source /data15/data15_5/fanhao/miniforge3/etc/profile.d/conda.sh
mamba activate mirmil
which python
git status --short
nvidia-smi
```

GPU 使用原则：先看 `nvidia-smi`。不要抢明显被其他用户长期占用的 GPU。若空闲卡充足，可以并行跑多个模型；若不确定，先用 0/1/2/3 分批。

## 1. 每半小时轮询 NSCLC 下载

### 单次检查

```bash
NSCLC=/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/nsclc
LOG=/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_r50_uni_patches_20260716_002213.log
date
ps -p 3566032 -o pid,etime,pcpu,pmem,cmd || true
du -sh /data15/data15_5/fanhao/datasets/TCGA-NSCLC /data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature 2>/dev/null || true
for sub in patches r50 uni; do
  [ -d "$NSCLC/$sub" ] && printf "%-8s files=%s size=%s\n" "$sub" "$(find "$NSCLC/$sub" -type f | wc -l)" "$(du -sh "$NSCLC/$sub" | awk '{print $1}')"
done
tail -40 "$LOG" 2>/dev/null || true
```

已知参考：`patches` 已经 1046 个文件完成；最近一次 r50 约 883 个文件，uni 可能尚未开始或尚未落盘。最终数量以 HuggingFace snapshot 完成日志和目录稳定为准，不要只按单次文件数判断。

### 半小时轮询

```bash
mkdir -p /data15/data15_5/fanhao/projects/MIRMIL/reports/monitors
while true; do
  {
    echo "===== $(date) ====="
    ps -p 3566032 -o pid,etime,pcpu,pmem,cmd || true
    NSCLC=/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/nsclc
    for sub in patches r50 uni; do
      [ -d "$NSCLC/$sub" ] && printf "%-8s files=%s size=%s\n" "$sub" "$(find "$NSCLC/$sub" -type f | wc -l)" "$(du -sh "$NSCLC/$sub" | awk '{print $1}')"
    done
    tail -20 /data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_r50_uni_patches_20260716_002213.log 2>/dev/null || true
  } | tee -a /data15/data15_5/fanhao/projects/MIRMIL/reports/monitors/nsclc_download_watch.log
  sleep 1800
done
```

如果 PID 退出但日志没有 `done`，优先用同一个 `snapshot_download` 命令续传，不要删除目录：

```bash
DEST=/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature
mkdir -p "$DEST/logs"
nohup /data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python -u - <<'PY' > "$DEST/logs/download_nsclc_resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
import datetime
from huggingface_hub import snapshot_download
repo = "Dearcat/CPathPatchFeature"
dest = "/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature"
allow = ["nsclc/r50/**", "nsclc/uni/**", "nsclc/patches/**"]
print(f"[{datetime.datetime.now().isoformat()}] resume repo={repo} dest={dest} allow={allow}", flush=True)
snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=allow, local_dir=dest, max_workers=6)
print(f"[{datetime.datetime.now().isoformat()}] done", flush=True)
PY
```

## 2. NSCLC 标签与实验

已有 NSCLC LUAD vs LUSC 标签：

```text
/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_R50_train_val.csv
/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_R50_split.csv
/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_UNI_train_val.csv
/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_UNI_split.csv
```

如果下载完成后发现这些 split 没覆盖新下载路径，重新生成/刷新。优先使用已有脚本：

```bash
cd /data15/data15_5/fanhao/projects/MIRMIL
python experiments/prepare_tcga_nsclc.py --help
```

### 推荐先跑 dry-run

`experiments/run_benchmark.py` 是当前最合适的统一入口。它支持 baseline 和 MIRMIL/MIRMIL-MT：

```bash
cd /data15/data15_5/fanhao/projects/MIRMIL
PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
LOGROOT=/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC

$PY experiments/run_benchmark.py \
  --split /data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_R50_split.csv \
  --dataset-name TCGA_NSCLC_LUAD_LUSC_R50 \
  --num-classes 2 \
  --log-root "$LOGROOT" \
  --models AB_MIL CLAM_SB_MIL CLAM_MB_MIL DS_MIL TRANS_MIL RRT_MIL WIKG_MIL AC_MIL MO_MIL MAMBA2D_MIL MIR_MIL MIR_MIL_MT_V1 \
  --seeds 2024 2025 2026 \
  --epochs 30 \
  --device 0 \
  --num-workers 4 \
  --in-dim 1024 \
  --max-instances 4096 \
  --feature r50 \
  --protocol nsclc_luad_lusc \
  --split-id r50_v1 \
  --dry-run
```

UNI 同理：

```bash
$PY experiments/run_benchmark.py \
  --split /data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata/TCGA_NSCLC_LUAD_LUSC_UNI_split.csv \
  --dataset-name TCGA_NSCLC_LUAD_LUSC_UNI \
  --num-classes 2 \
  --log-root "$LOGROOT" \
  --models AB_MIL CLAM_SB_MIL CLAM_MB_MIL DS_MIL TRANS_MIL RRT_MIL WIKG_MIL AC_MIL MO_MIL MAMBA2D_MIL MIR_MIL MIR_MIL_MT_V1 \
  --seeds 2024 2025 2026 \
  --epochs 30 \
  --device 1 \
  --num-workers 4 \
  --in-dim 1024 \
  --max-instances 4096 \
  --feature uni \
  --protocol nsclc_luad_lusc \
  --split-id uni_v1 \
  --dry-run
```

确认 dry-run 命令正确后，去掉 `--dry-run`。建议按 GPU 分批启动，每个命令写独立日志：

```bash
mkdir -p "$LOGROOT/controller_logs"
nohup $PY experiments/run_benchmark.py ...去掉dry-run... > "$LOGROOT/controller_logs/nsclc_r50_gpu0_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
nohup $PY experiments/run_benchmark.py ...UNI命令... > "$LOGROOT/controller_logs/nsclc_uni_gpu1_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

MAMBA2D 可能显存压力大；如果 OOM，单独降低：

```text
--models MAMBA2D_MIL --max-instances 2048
```

### 与 COADREAD 下载/准备并行

不要等 NSCLC 全部 GPU 实验跑完再处理 COADREAD。NSCLC 下载完成后可以立即并行启动两条线：

- 线 A：NSCLC r50/uni 训练，占用 GPU。
- 线 B：COADREAD 特征定位、WSI 下载、manifest 整理等 I/O/网络任务，通常不占 GPU。

如果 COADREAD 需要特征提取，特征提取也会占 GPU；这时根据 `nvidia-smi` 给它分配空闲 GPU，避免和 NSCLC 训练抢同一张卡。记录每条线的 PID、GPU 和日志。

## 3. NSCLC 完成后处理 COADREAD

用户说 COADREAD 特征已经提前下载并放在 dataset 目录。先查：

```bash
find /data15/data15_5/fanhao/datasets/TCGA-COADREAD -maxdepth 5 -type d | sort
find /data15/data15_5/fanhao/datasets/TCGA-COADREAD -maxdepth 6 -type f \( -name "*.pt" -o -name "*.h5" -o -name "*.svs" \) | head -50
du -sh /data15/data15_5/fanhao/datasets/TCGA-COADREAD/* 2>/dev/null
```

已准备好的 COADREAD 预后 GT 在：

```text
/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA-COADREAD-PROGNOSIS
```

但当前只有 GT/assignments/template，不含 feature path split。下载或定位特征后，需要按 `patient_id` join 出：

```text
TCGA_COADREAD_PROGNOSIS_R50_OS_split.csv
TCGA_COADREAD_PROGNOSIS_UNI_OS_split.csv
...
```

COADREAD 预后主终点建议：PFS、OS；DFS 事件少，只作为补充。

## 4. 如果需要下载 COADREAD WSI 并提取特征

### 下载 WSI

优先使用项目已有 TCGA 准备脚本，先看参数：

```bash
cd /data15/data15_5/fanhao/projects/MIRMIL
python experiments/prepare_tcga_projects.py --help
```

如果脚本不适配 COADREAD，就用 GDC manifest/API 获取 TCGA-COAD、TCGA-READ Primary Tumor Diagnostic Slide。保存 manifest 到：

```text
/data15/data15_5/fanhao/datasets/TCGA-COADREAD/manifests
```

WSI 建议落点：

```text
/data15/data15_5/fanhao/datasets/TCGA-COADREAD/WSI/COAD
/data15/data15_5/fanhao/datasets/TCGA-COADREAD/WSI/READ
```

### 生成 patches

先生成包含 `wsi_path` 的 CSV：

```bash
COADREAD=/data15/data15_5/fanhao/datasets/TCGA-COADREAD
find "$COADREAD/WSI" -type f \( -iname "*.svs" -o -iname "*.tif" -o -iname "*.mrxs" -o -iname "*.ndpi" \) \
  | awk 'BEGIN{print "wsi_path"} {print}' > "$COADREAD/metadata/coadread_wsi_paths.csv"
```

启动 patching：

```bash
cd /data15/data15_5/fanhao/projects/MIRMIL
PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
COADREAD=/data15/data15_5/fanhao/datasets/TCGA-COADREAD

nohup $PY feature_extractor/create_h5_patches.py \
  --source "$COADREAD/WSI" \
  --source_csv "$COADREAD/metadata/coadread_wsi_paths.csv" \
  --save_dir "$COADREAD/patches_level0_256" \
  --patch_size 256 \
  --step_size 256 \
  --level_or_magnification_control level \
  --patch_level 0 \
  --multiprocess_slide 8 \
  --no-stitch \
  --save_mask \
  > "$COADREAD/patches_level0_256/patching_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

### 提取 R50/UNI 特征

确认权重目录。若未知，先查：

```bash
find /data15/data15_5/fanhao -maxdepth 5 -type f \( -iname "*uni*.pth" -o -iname "*resnet*pth" -o -iname "*r50*pth" \) 2>/dev/null | head -50
```

R50：

```bash
COADREAD=/data15/data15_5/fanhao/datasets/TCGA-COADREAD
PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python

nohup $PY feature_extractor/create_pt_features.py \
  --data_h5_dir "$COADREAD/patches_level0_256" \
  --process_wsi_paths_csv "$COADREAD/metadata/coadread_wsi_paths.csv" \
  --feat_dir "$COADREAD/features/r50" \
  --backbone resnet50_imagenet \
  --batch_size 256 \
  --num_workers 4 \
  --num_shards 4 \
  --shard_index 0 \
  --failure_log "$COADREAD/features/r50/failures_shard0.csv" \
  > "$COADREAD/features/r50/extract_shard0.log" 2>&1 &
```

UNI：

```bash
nohup $PY feature_extractor/create_pt_features.py \
  --data_h5_dir "$COADREAD/patches_level0_256" \
  --process_wsi_paths_csv "$COADREAD/metadata/coadread_wsi_paths.csv" \
  --feat_dir "$COADREAD/features/uni" \
  --backbone uni \
  --batch_size 128 \
  --num_workers 4 \
  --num_shards 4 \
  --shard_index 0 \
  --failure_log "$COADREAD/features/uni/failures_shard0.csv" \
  > "$COADREAD/features/uni/extract_shard0.log" 2>&1 &
```

多卡并行时复制命令并改 `--shard_index 0/1/2/3`，同时用 `CUDA_VISIBLE_DEVICES=<gpu>` 控制每个进程使用的 GPU。

## 5. COADREAD、KIRC、BLCA 预后任务

### BLCA

BLCA 已有 R50 预后 GT。优先跑 OS；DFS 可以作为补充。常用路径：

```text
/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv
/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_train_val.csv
/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_DFS_split.csv
/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_DFS_train_val.csv
```

示例：用同一个 survival head/loss/cutpoints 包装不同 backbone。先从 RRT/MIR/MIR-MT 跑起，再扩展 AB/MEAN/MAX 等 baseline。

```bash
cd /data15/data15_5/fanhao/projects/MIRMIL
PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
LOGROOT=/data15/data15_5/fanhao/experiments/MIRMIL_PROGNOSIS

nohup $PY train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options \
  General.seed=2024 \
  General.num_epochs=30 \
  General.device=0 \
  General.best_model_metric=c_index \
  General.earlystop.use=true \
  General.earlystop.patience=8 \
  General.earlystop.metric=c_index \
  Dataset.DATASET_NAME=TCGA_BLCA_R50_OS \
  Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-BLCA/metadata/TCGA_BLCA_PROGNOSIS_R50_OS_split.csv \
  Logs.log_root_dir=$LOGROOT \
  Model.backbone=RRT_MIL \
  Model.backbone_config=configs/RRT_MIL.yaml \
  Model.in_dim=1024 \
  Model.max_instances=4096 \
  Model.survival.num_bins=4 \
  Model.survival.time_column=time_months \
  Model.survival.event_column=event \
  Model.survival.patient_column=patient_id \
  Model.survival.patient_level=true \
  > "$LOGROOT/blca_r50_os_rrt_seed2024.log" 2>&1 &
```

### KIRC

KIRC 已有 R50/UNI 预后 split，推荐主跑 OS/PFS，DSS 可补充，DFS 事件太少不建议做主结果。

```text
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_OS_split.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_R50_PFS_split.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_PFS_split.csv
```

示例：KIRC UNI OS + MIR_MIL。

```bash
nohup $PY train_mil.py --yaml_path configs/SURVIVAL_MIL.yaml --options \
  General.seed=2024 \
  General.num_epochs=30 \
  General.device=1 \
  General.best_model_metric=c_index \
  General.earlystop.use=true \
  General.earlystop.patience=8 \
  General.earlystop.metric=c_index \
  Dataset.DATASET_NAME=TCGA_KIRC_UNI_OS \
  Dataset.dataset_csv_path=/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA-KIRC-PROGNOSIS/TCGA_KIRC_PROGNOSIS_UNI_OS_split.csv \
  Logs.log_root_dir=$LOGROOT \
  Model.backbone=MIR_MIL \
  Model.backbone_config=configs/MIR_MIL.yaml \
  Model.in_dim=1024 \
  Model.max_instances=4096 \
  Model.survival.num_bins=4 \
  Model.survival.time_column=time_months \
  Model.survival.event_column=event \
  Model.survival.patient_column=patient_id \
  Model.survival.patient_level=true \
  > "$LOGROOT/kirc_uni_os_mir_seed2024.log" 2>&1 &
```

### COADREAD

COADREAD 当前已有 patient-level 预后 GT 和 assignments，但未必已有 feature path split；需要等 WSI/feature 完成或确认用户已放入特征后，再按 `patient_id` join 生成 `*_split.csv`。

```text
/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA-COADREAD-PROGNOSIS/TCGA_COADREAD_PROGNOSIS_OS_assignments.csv
/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA-COADREAD-PROGNOSIS/TCGA_COADREAD_PROGNOSIS_PFS_assignments.csv
/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA-COADREAD-PROGNOSIS/TCGA_COADREAD_PROGNOSIS_OS_feature_template_long.csv
/data15/data15_5/fanhao/datasets/TCGA-COADREAD/metadata/TCGA-COADREAD-PROGNOSIS/TCGA_COADREAD_PROGNOSIS_PFS_feature_template_long.csv
```

COADREAD 预后主终点建议：

```text
PFS: 590 patients, 150 events
OS:  590 patients, 120 events
DSS: 569 patients, 73 events
DFS: 223 patients, 30 events, 只做补充
```

生成 COADREAD feature split 后，按 BLCA/KIRC 的 `SURVIVAL_MIL` 命令模板启动。不要把分期分类命令用于这些预后任务。

## 6. 运行后汇总

训练过程看：

```bash
ps -u fanhao -f | egrep "train_mil|run_benchmark|create_pt_features|create_h5_patches" | grep -v egrep
nvidia-smi
tail -f /path/to/controller.log
```

结果汇总可先用：

```bash
cd /data15/data15_5/fanhao/projects/MIRMIL
python experiments/summarize_results.py --help
find /data15/data15_5/fanhao/experiments -name "Best_Log*.csv" | tail -50
```

如果启动任何长任务，请把：

- 命令
- PID
- GPU
- 日志路径
- 数据 split 路径

追加写入：

```text
/data15/data15_5/fanhao/projects/MIRMIL/reports/monitors/server_codex_runs.md
```

## 7. 重要提醒

- NSCLC 下载尚未完成时不要启动 NSCLC 实验。
- COADREAD 如果已有特征，就不要重复下载 WSI/重复提特征；先定位现有特征路径。
- 预后任务必须 patient-level split，不能 slide-level 泄漏。
- 预后任务优先 OS/PFS；COADREAD/KIRC 的 DFS 事件较少，不适合作主实验。
- 若远端项目 Git 状态很乱，不要清理；只在明确路径下新增日志/脚本/结果。
