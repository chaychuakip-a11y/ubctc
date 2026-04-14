#!/bin/bash

# Define log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting data process pipeline..."

# 1. Serial execution of 1_dnnfa.pl and 2.0_GenSeedMlf.pl
echo "[1/3] Running serial scripts: 1_dnnfa.pl and 2.0_GenSeedMlf.pl"
perl 1_dnnfa.pl > $LOG_DIR/1_dnnfa.log 2>&1
if [ $? -ne 0 ]; then echo "Error in 1_dnnfa.pl. Check $LOG_DIR/1_dnnfa.log"; exit 1; fi

perl 2.0_GenSeedMlf.pl > $LOG_DIR/2.0_GenSeedMlf.log 2>&1
if [ $? -ne 0 ]; then echo "Error in 2.0_GenSeedMlf.pl. Check $LOG_DIR/2.0_GenSeedMlf.log"; exit 1; fi

# 2. Parallel execution of AddNoise, Speedup, and Amp scripts
echo "[2/3] Running parallel augmentation scripts..."
scripts=(
    "2.1_AddNoise_car_byd.5db.20.pl"
    "2.1_AddNoise_car_dz.5db.20.pl"
    "2.1_AddNoise_duodian.5db.10.pl"
    "2.1_AddNoise_gs.5db.20.pl"
    "2.1_AddNoise_jiaju.5db.10.pl"
    "2.1_AddNoise_music_a.5db.2.pl"
    "2.1_AddNoise_music_b.5db.2.pl"
    "2.1_AddNoise_music_c.5db.2.pl"
    "2.1_AddNoise_music_onenoise.5db.2.pl"
    "2.1_AddNoise_music_tv.5db.2.pl"
    "2.1_AddNoise_pingwen.5db.10.pl"
    "3.0_speedup1.2.pl"
    "4.0_amp.pl"
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

# 3. Running 3.1_dnnfa.pl (requires speedup1.2 output)
echo "[3/3] Running 3.1_dnnfa.pl..."
perl 3.1_dnnfa.pl > $LOG_DIR/3.1_dnnfa.log 2>&1

echo "Pipeline base scripts finished. Check logs in $LOG_DIR/"
echo "To view YARN logs, find the application ID in the logs and run:"
echo "  yarn logs -applicationId <application_id>"
