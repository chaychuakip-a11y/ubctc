#!/bin/bash
# get_mlf.sh — 从 HDFS 直接下载 MLF 标签 (phone + word)
#
# 原理：对齐 rnnt/1.get_mlf_sp_hu/run.sh 的模式
#   hdfs dfs -cat <hdfs_path>/*$i | fea_lab_lat_unpack_1 - ./$i <label_type>
#
# 输出：
#   0/out.mlf_fa_ph   ...  9/out.mlf_fa_ph   ← phone 级 MLF (10份)
#   0/out.mlf_sy      ...  9/out.mlf_sy       ← word  级 MLF (10份)
#   out.mlf_fa_ph                              ← 合并后 phone MLF
#   out.mlf_sy                                 ← 合并后 word  MLF
#
# 用法：
#   bash get_mlf.sh            # 下载 phone + word 两种标签
#   bash get_mlf.sh phone      # 只下载 phone 标签 (mlf_fa_ph)
#   bash get_mlf.sh word       # 只下载 word  标签 (mlf_sy)

set -e

# ─────────────────────────────────────────────
# 1. 配置
# ─────────────────────────────────────────────

# HDFS 数据路径 (对应 1_get_pfile_from_hdfs.pl 第10行)
HDIR="/workdir/asrdictt/tasrdictt/taoyu/mlg/korean/17kh_wav_aug1.8.wav_fb40_dnnfa"

# 工具路径 (对应 rnnt/1.get_mlf_sp_hu/run.sh 第6行)
BIN_UNPACK="/work1/asrdictt/hjwang11/bin/fea_lab_lat_unpack_1"

# 并行分片数 (对应 1_get_pfile_from_hdfs.pl 第11行 nPart=100, nSplit=10)
N_PARTS=10

# 解码哪种标签 (phone / word / both)
MODE=${1:-both}

# ─────────────────────────────────────────────
# 2. 检查工具
# ─────────────────────────────────────────────

if [ ! -f "$BIN_UNPACK" ]; then
    echo "[error] 解包工具不存在: $BIN_UNPACK"
    exit 1
fi

# ─────────────────────────────────────────────
# 3. 按分片并行下载
# ─────────────────────────────────────────────

echo "[get_mlf] HDFS   : $HDIR"
echo "[get_mlf] mode   : $MODE"
echo "[get_mlf] parts  : $N_PARTS"
echo ""

download_part() {
    local i=$1
    local label=$2        # mlf_fa_ph 或 mlf_sy
    local outfile=$3      # 输出文件名 (如 out.mlf_fa_ph)

    mkdir -p "./$i"
    local out="./$i/$outfile"

    if [ -f "${out}.done" ]; then
        echo "[part $i / $label] 已存在，跳过"
        return
    fi

    echo "[part $i / $label] 开始下载..."
    hdfs dfs -cat "${HDIR}/*${i}" \
        | "$BIN_UNPACK" - "./$i" "$label" \
        > "./$i/unpack_${label}.log" 2>&1

    touch "${out}.done"
    echo "[part $i / $label] 完成  ($(wc -l < "$out") 行)"
}

# 并行启动所有分片
pids=()

for ((i=0; i<N_PARTS; i++)); do
    if [ "$MODE" = "phone" ] || [ "$MODE" = "both" ]; then
        download_part "$i" "mlf_fa_ph" "out.mlf_fa_ph" &
        pids+=($!)
    fi
    if [ "$MODE" = "word" ] || [ "$MODE" = "both" ]; then
        download_part "$i" "mlf_sy" "out.mlf_sy" &
        pids+=($!)
    fi
done

# 等待全部完成，检查失败
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "[error] part pid=$pid 失败"; fail=1; }
done
[ "$fail" -eq 1 ] && { echo "[error] 有分片失败，退出"; exit 1; }

echo ""
echo "[get_mlf] 所有分片下载完成，开始合并..."

# ─────────────────────────────────────────────
# 4. 合并为单个 MLF 文件
# ─────────────────────────────────────────────

merge_mlf() {
    local label_file=$1    # 如 out.mlf_fa_ph
    local out_mlf=$2       # 合并后输出路径

    echo "[merge] $label_file -> $out_mlf"

    echo "#!MLF!#" > "$out_mlf"

    for ((i=0; i<N_PARTS; i++)); do
        local part_file="./$i/$label_file"
        if [ ! -f "$part_file" ]; then
            echo "[warn] 分片文件不存在: $part_file，跳过"
            continue
        fi
        # 跳过各分片的 #!MLF!# 头行，追加正文
        grep -v "^#!MLF!#" "$part_file" >> "$out_mlf"
    done

    local n_utt
    n_utt=$(grep -c '^\.' "$out_mlf" || true)
    echo "[merge] 完成: $out_mlf  ($n_utt 条语音)"
}

if [ "$MODE" = "phone" ] || [ "$MODE" = "both" ]; then
    merge_mlf "out.mlf_fa_ph" "./out.mlf_fa_ph"
fi
if [ "$MODE" = "word" ] || [ "$MODE" = "both" ]; then
    merge_mlf "out.mlf_sy"    "./out.mlf_sy"
fi

# ─────────────────────────────────────────────
# 5. 汇总
# ─────────────────────────────────────────────

echo ""
echo "============================================"
echo "[get_mlf] 完成"
[ -f "./out.mlf_fa_ph" ] && echo "  phone MLF : ./out.mlf_fa_ph  ($(grep -c '^\.' out.mlf_fa_ph) utts)"
[ -f "./out.mlf_sy"    ] && echo "  word  MLF : ./out.mlf_sy     ($(grep -c '^\.' out.mlf_sy) utts)"
echo ""
echo "  -> 供 decode_wav.py 使用:"
echo "     --mlf $(pwd)/out.mlf_fa_ph --mlf_level phone"
echo "============================================"
