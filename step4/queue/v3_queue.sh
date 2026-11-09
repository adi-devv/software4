#!/usr/bin/env bash
# V3 full queue: seeds 53 and 54 start now; 55 starts when one of them finishes COMPLETE (Kaggle allows 2 GPU sessions).
cd /d/Aadit/claudez/software4-v3-git
KR=/d/Aadit/claudez/software4/scripts/kaggle_run.sh
$KR step4/upload_v3_full53 NvidiaTeslaT4 14400 > step4/queue/s53.log 2>&1 &
sleep 20
$KR step4/upload_v3_full54 NvidiaTeslaT4 14400 > step4/queue/s54.log 2>&1 &
while true; do
  sleep 120
  for s in 53 54; do
    st=$(kaggle kernels status adidevv/software4-step4-v3-s$s 2>&1 | tail -1)
    case "$st" in
      *COMPLETE*) echo "$(date -u +%H:%M:%S)Z s$s COMPLETE -> starting s55"; $KR step4/upload_v3_full55 NvidiaTeslaT4 14400 > step4/queue/s55.log 2>&1; wait; exit 0;;
      *ERROR*|*CANCEL*) echo "$(date -u +%H:%M:%S)Z s$s failed/cancelled ($st); NOT starting s55 (stop rule); needs review"; wait; exit 1;;
    esac
  done
done
