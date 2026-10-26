#!/usr/bin/env bash
# Portable restart-loop wrapper around `python main.py demo`, for anyone who
# doesn't want to set up systemd (e.g. macOS, or a quick nohup-style run on
# a machine you're not going to configure further). See deploy/README.md.
#
# Restarts on crash with exponential backoff. Gives up -- rather than
# restart-looping forever -- if it's crashing repeatedly in a short window,
# since that means a real bug, not a transient blip (a dropped connection,
# yfinance hiccup, etc).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BROKER="${1:-simulated}"
MAX_FAILURES_IN_WINDOW=5
FAILURE_WINDOW_SECONDS=600
INITIAL_BACKOFF_SECONDS=5
MAX_BACKOFF_SECONDS=300

backoff_seconds=$INITIAL_BACKOFF_SECONDS
failure_times=()

log() { echo "[supervisor] $(date -u '+%Y-%m-%dT%H:%M:%SZ') $*"; }

log "Starting supervised demo loop (broker=$BROKER). Ctrl-C to stop for good."

while true; do
    start_time=$(date +%s)
    log "Launching: python3 main.py demo --broker $BROKER"
    python3 main.py demo --broker "$BROKER"
    exit_code=$?
    now=$(date +%s)
    run_duration=$((now - start_time))

    if [[ $exit_code -eq 0 ]]; then
        log "Demo loop exited cleanly (exit 0, e.g. Ctrl-C). Not restarting."
        exit 0
    fi

    log "Demo loop exited with code $exit_code after running for ${run_duration}s."

    # Ran long enough to count as recovered: don't let a stale backoff/failure
    # count from a past bad patch linger forever and punish an unrelated,
    # much-later crash with a huge initial delay.
    if [[ $run_duration -ge $FAILURE_WINDOW_SECONDS ]]; then
        backoff_seconds=$INITIAL_BACKOFF_SECONDS
        failure_times=()
    fi

    failure_times+=("$now")
    cutoff=$((now - FAILURE_WINDOW_SECONDS))
    kept=()
    for t in "${failure_times[@]}"; do
        [[ $t -ge $cutoff ]] && kept+=("$t")
    done
    failure_times=("${kept[@]}")

    if [[ ${#failure_times[@]} -ge $MAX_FAILURES_IN_WINDOW ]]; then
        log "Failed ${#failure_times[@]} times in the last ${FAILURE_WINDOW_SECONDS}s. Giving up --" \
            "this looks like a real bug, not a transient blip. Check logs/gold_trader.log, fix it," \
            "then re-run this script."
        exit 1
    fi

    log "Restarting in ${backoff_seconds}s..."
    sleep "$backoff_seconds"
    backoff_seconds=$(( backoff_seconds * 2 < MAX_BACKOFF_SECONDS ? backoff_seconds * 2 : MAX_BACKOFF_SECONDS ))
done
