#!/bin/bash
# UBCTC 解码示例脚本
# 对应训练: 3_ub_fb40/2_train
# 特征:     fb40 (16kHz, 25ms/10ms Hamming, offCMN + global norm)
# 输出层级: state IDs -> triphone labels -> phones -> words (逐步可选)

# ─────────────────────────────────────────────
# 0. 环境配置 (对齐 train.sh)
# ─────────────────────────────────────────────

gpu_type=$(nvidia-smi -L 2>/dev/null | head -n 1 | cut -d'(' -f1 | cut -d':' -f2 | sed -e 's/^\s\+//' -e 's/\s\+$//')
if [ "$gpu_type" = 'NVIDIA GeForce RTX 3090' ] || [ "$gpu_type" = 'NVIDIA A40' ]; then
    ln -sf c-a.so asr/c.so
    source /home3/asrdictt/taoyu/conda_zhyou2_pth39cu111tch191.bashrc
else
    ln -sf c-v.so asr/c.so
    source /home3/asrdictt/taoyu/conda_zhyou2_pth39cu102tch191.bashrc
fi

export CUDA_VISIBLE_DEVICES=0

echo "[decode] GPU       : ${gpu_type:-CPU}"
echo "[decode] python    : $(which python)"

# ─────────────────────────────────────────────
# 1. 路径配置
# ─────────────────────────────────────────────

DIR_TRAIN=$(cd "$(dirname "$0")" && pwd)   # 本脚本所在目录 (2_train/)
DIR_AM=$(dirname "$DIR_TRAIN")             # 3_ub_fb40/

# 模型 checkpoint
MODEL=$DIR_AM/2_train/exp/ubctc.final.pt

# 特征归一化 (内网文件, 无法获取时注释掉 --norm 行)
FEA_NORM=$DIR_AM/1_down_pfile/lib_fb40/fea.norm

# State-label 字典: 每行 "<triphone_label> <state_id>"
# 由 hmmlist.final 整理而来, 共 9004 行 (含 blank=9003)
STATE_DICT=$DIR_AM/res/state_label.txt

# 发音词典: 每行 "word<TAB>ph1 ph2 ..."
LEXICON=$DIR_AM/res/lexicon.txt

# ─────────────────────────────────────────────
# 输入模式 (二选一)
# ─────────────────────────────────────────────
#   单条: bash decode.sh --wav /data/test/utt001.wav
#   批量: bash decode.sh --wav_dir /data/test/wavs  [--mlf /data/test/ref.mlf]

MODE=greedy             # greedy 或 beam
BEAM=10                 # beam search 宽度 (仅 MODE=beam 时生效)
BLANK=9003              # CTC blank id = num_class - 1
SR=16000                # 采样率

OUTPUT=$DIR_AM/decode/result.txt   # 解码结果输出文件 (逐行: utt_id<TAB>phones)

# ─────────────────────────────────────────────
# 2. 解析命令行参数
# ─────────────────────────────────────────────

WAV=""
WAV_DIR=""
MLF=""
MLF_LEVEL="phone"       # phone 或 word

while [[ $# -gt 0 ]]; do
    case $1 in
        --wav)     WAV="$2";      shift 2 ;;
        --wav_dir) WAV_DIR="$2";  shift 2 ;;
        --mlf)     MLF="$2";      shift 2 ;;
        --mlf_level) MLF_LEVEL="$2"; shift 2 ;;
        *) echo "[warn] 未知参数: $1"; shift ;;
    esac
done

# 检查输入模式
if [ -z "$WAV" ] && [ -z "$WAV_DIR" ]; then
    echo "Usage:"
    echo "  单条: bash decode.sh --wav /path/to/utt.wav"
    echo "  批量: bash decode.sh --wav_dir /path/to/wavs/ [--mlf ref.mlf]"
    exit 1
fi

if [ -n "$WAV" ] && [ ! -f "$WAV" ]; then
    echo "[error] wav 文件不存在: $WAV"; exit 1
fi

if [ -n "$WAV_DIR" ] && [ ! -d "$WAV_DIR" ]; then
    echo "[error] 目录不存在: $WAV_DIR"; exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "[error] 模型文件不存在: $MODEL"
    echo "        请修改脚本中的 MODEL 路径"; exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT")"

# ─────────────────────────────────────────────
# 3. 组装解码命令
# ─────────────────────────────────────────────

CMD="python $DIR_TRAIN/decode_wav.py"
CMD="$CMD --model  $MODEL"
CMD="$CMD --mode   $MODE"
CMD="$CMD --beam   $BEAM"
CMD="$CMD --blank  $BLANK"
CMD="$CMD --sr     $SR"
CMD="$CMD --gpu    0"
CMD="$CMD --output $OUTPUT"

# 单条 / 批量
if [ -n "$WAV" ]; then
    CMD="$CMD --wav $WAV"
else
    CMD="$CMD --wav_dir $WAV_DIR"
fi

# 全局特征归一化 (内网文件, 可选)
if [ -f "$FEA_NORM" ]; then
    CMD="$CMD --norm $FEA_NORM"
else
    echo "[warn] fea.norm 不存在, 跳过全局归一化: $FEA_NORM"
fi

# Triphone -> phone 转换
if [ -f "$STATE_DICT" ]; then
    CMD="$CMD --dict $STATE_DICT --phone"
else
    echo "[warn] state_label 字典不存在, 仅输出 state ID: $STATE_DICT"
fi

# Phone -> word 转换
if [ -f "$LEXICON" ]; then
    CMD="$CMD --lex $LEXICON"
else
    echo "[warn] 发音词典不存在, 仅输出 phone 序列: $LEXICON"
fi

# MLF 参考对比
if [ -n "$MLF" ]; then
    if [ -f "$MLF" ]; then
        CMD="$CMD --mlf $MLF --mlf_level $MLF_LEVEL"
    else
        echo "[warn] MLF 文件不存在, 跳过对比: $MLF"
    fi
fi

# ─────────────────────────────────────────────
# 4. 执行解码
# ─────────────────────────────────────────────

echo ""
echo "[decode] cmd: $CMD"
echo ""
eval "$CMD"
