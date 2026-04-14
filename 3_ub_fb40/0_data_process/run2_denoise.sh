#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting denoise process pipeline..."

# Running LSA, MaeClose, and MaeOpen in parallel as they are independent
echo "[1/2] Running parallel denoise scripts..."
scripts=(
    "6.0_LsaDenoise.pl"
    "7.0_MaeDenoiseClose.pl"
    "8.0_MaeDenoiseOpen.pl"
)

pids=()
for script in "${scripts[@]}"; do
    log_file="$LOG_DIR/${script%.pl}.log"
    echo "  Launching $script -> $log_file"
    perl $script > $log_file 2>&1 &
    pids+=($!)
done

# Wait for all parallel scripts to finish
for pid in "${pids[@]}"; do
    wait $pid
done

# After denoise completes, run the dnnfa for each
echo "[2/2] Running parallel dnnfa for denoise outputs..."
dnnfa_scripts=(
    "6.1_dnnfa.pl"
    "7.1_dnnfa.pl"
    "8.1_dnnfa.pl"
)

pids=()
for script in "${dnnfa_scripts[@]}"; do
    log_file="$LOG_DIR/${script%.pl}.log"
    echo "  Launching $script -> $log_file"
    perl $script > $log_file 2>&1 &
    pids+=($!)
done

# Wait for all parallel scripts to finish
for pid in "${pids[@]}"; do
    wait $pid
done

echo "Denoise pipeline finished. Check logs in $LOG_DIR/"
echo "To view YARN logs, find the application ID in the logs and run:"
echo "  yarn logs -applicationId <application_id>"
