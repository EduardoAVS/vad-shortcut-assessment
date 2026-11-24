#!/usr/bin/env bash
# Runs a job with nice, a limited number of threads and a memory watchdog.
# Usage: tools/run_guarded.sh <rss_limit_MB> <log_file> -- <command...>
# The watchdog kills the job (and its children) if their summed RSS exceeds the limit or if the
# available memory drops below MIN_AVAIL_MB (default 700). Written for WSL, where systemd's
# MemoryMax is not enforced. All experiments of the paper were run on CPU through this script.
set -u
LIMIT_MB="$1"; LOG="$2"; shift 2
[ "${1:-}" = "--" ] && shift
mkdir -p "$(dirname "$LOG")"
MIN_AVAIL_MB="${MIN_AVAIL_MB:-700}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
setsid nice -n 10 "$@" >>"$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  RSS_KB=$(ps -o rss= --sid "$PID" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$RSS_KB" -gt $((LIMIT_MB*1024)) ] || [ "$AVAIL_KB" -lt $((MIN_AVAIL_MB*1024)) ]; then
    echo "[watchdog] $(date +%T) job killed: RSS=$((RSS_KB/1024)) MB, available=$((AVAIL_KB/1024)) MB" >>"$LOG"
    kill -TERM -- -"$PID" 2>/dev/null; sleep 3; kill -KILL -- -"$PID" 2>/dev/null
    exit 137
  fi
  sleep 2
done
wait "$PID"
