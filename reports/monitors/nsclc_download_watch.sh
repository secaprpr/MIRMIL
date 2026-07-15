#!/usr/bin/env bash
set -u

NSCLC=/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/nsclc
LOG=/data15/data15_5/fanhao/datasets/TCGA-NSCLC/CPathPatchFeature/logs/download_nsclc_r50_uni_patches_20260716_002213.log
OUT=/data15/data15_5/fanhao/projects/MIRMIL/reports/monitors/nsclc_download_watch.log

while true; do
  {
    echo "===== $(date) ====="
    ps -p 3566032 -o pid,etime,pcpu,pmem,cmd || true
    for sub in patches r50 uni; do
      if [ -d "$NSCLC/$sub" ]; then
        files=$(find "$NSCLC/$sub" -type f | wc -l)
        size=$(du -sh "$NSCLC/$sub" 2>/dev/null | cut -f1)
        printf "%-8s files=%s size=%s\n" "$sub" "$files" "$size"
      else
        printf "%-8s missing\n" "$sub"
      fi
    done
    tail -20 "$LOG" 2>/dev/null || true
  } | tee -a "$OUT"
  sleep 1800
done
