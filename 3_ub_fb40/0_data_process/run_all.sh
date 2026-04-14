#!/bin/bash

# ==============================================================================
# 统一运行脚本：run_all.sh
# 功能：串行执行基础任务，并行执行增强、降噪及特征提取，最后合并。
# ==============================================================================

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "===== [开始数据处理流程] ====="

# --- 定义函数：并行执行脚本并等待完成 ---
run_parallel() {
    local scripts=("$@")
    local pids=()
    for script in "${scripts[@]}"; do
        log_file="$LOG_DIR/${script%.pl}.log"
        echo "  [后台运行] $script -> $log_file"
        perl "$script" > "$log_file" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

# --- 阶段 1: 基础串行任务 ---
echo "[1/5] 运行基础串行任务 (1_dnnfa, 2.0_GenSeedMlf)..."
perl 1_dnnfa.pl > $LOG_DIR/1_dnnfa.log 2>&1 || { echo "错误: 1_dnnfa.pl 失败"; exit 1; }
perl 2.0_GenSeedMlf.pl > $LOG_DIR/2.0_GenSeedMlf.log 2>&1 || { echo "错误: 2.0_GenSeedMlf.pl 失败"; exit 1; }

# --- 阶段 2: 增强任务并行 (加噪, 变速, 调幅) ---
echo "[2/5] 运行数据增强任务并行 (AddNoise, Speedup, Amp)..."
aug_scripts=(
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
run_parallel "${aug_scripts[@]}"

# --- 阶段 3: 降噪任务并行 & 变速后的 dnnfa ---
# 注意：3.1 依赖 3.0；6.0/7.0/8.0 依赖 2.1 的输出
echo "[3/5] 运行降噪任务 & 变速 dnnfa 并行..."
denoise_init_scripts=(
    "3.1_dnnfa.pl"
    "6.0_LsaDenoise.pl"
    "7.0_MaeDenoiseClose.pl"
    "8.0_MaeDenoiseOpen.pl"
)
run_parallel "${denoise_init_scripts[@]}"

# --- 阶段 4: 降噪后的 dnnfa 任务并行 ---
echo "[4/5] 运行降噪后的 dnnfa 任务并行..."
denoise_dnnfa_scripts=(
    "6.1_dnnfa.pl"
    "7.1_dnnfa.pl"
    "8.1_dnnfa.pl"
)
run_parallel "${denoise_dnnfa_scripts[@]}"

# --- 阶段 5: 最终特征合并 ---
echo "[5/5] 运行最终特征合并 (9_fea_merge)..."
perl 9_fea_merge.pl > $LOG_DIR/9_fea_merge.log 2>&1 || { echo "错误: 9_fea_merge.pl 失败"; exit 1; }

echo "===== [所有流程执行完毕] ====="
echo "详细日志请查看 $LOG_DIR/ 目录。"
echo "YARN 日志查看命令: yarn logs -applicationId <application_id>"
