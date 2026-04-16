#!/bin/bash
# get_mlf.sh — 从 HDFS 直接下载 MLF 标签 (phone + word)
#
# 原理：对齐 rnnt/1.get_mlf_sp_hu/run.sh 的模式
#   hdfs dfs -cat <hdfs_path>/<glob> | fea_lab_lat_unpack_1 - ./$i <label_type>
#
# 冗余设计：
#   - 自动探测 HDFS 文件数量，单文件/多文件均可处理
#   - 下载后验证输出非空且包含有效 MLF 头
#   - 标签不存在时报 warn 而不是静默生成空文件
#   - pipefail 确保管道中间失败被捕获
#
# 输出：
#   0/out.mlf_fa_ph  ...  N/out.mlf_fa_ph   ← phone 级 MLF (按分片)
#   0/out.mlf_sy     ...  N/out.mlf_sy       ← word  级 MLF (按分片)
#   out.mlf_fa_ph                             ← 合并后 phone MLF
#   out.mlf_sy                                ← 合并后 word  MLF
#
# 用法：
#   bash get_mlf.sh            # 下载 phone + word 两种标签
#   bash get_mlf.sh phone      # 只下载 phone 标签 (mlf_fa_ph)
#   bash get_mlf.sh word       # 只下载 word  标签 (mlf_sy)

set -eo pipefail

# ─────────────────────────────────────────────
# 1. 配置
# ─────────────────────────────────────────────

HDIR="/workdir/asrdictt/tasrdictt/taoyu/mlg/korean/17kh_wav_aug1.8.wav_fb40_dnnfa"
BIN_UNPACK="/work1/asrdictt/hjwang11/bin/fea_lab_lat_unpack_1"
MODE=${1:-both}     # phone / word / both

# ─────────────────────────────────────────────
# 2. 工具检查
# ─────────────────────────────────────────────

if [ ! -f "$BIN_UNPACK" ]; then
    echo "[error] 解包工具不存在: $BIN_UNPACK"
    exit 1
fi

# ─────────────────────────────────────────────
# 3. 自动探测 HDFS 文件结构，生成分片 glob 列表
# ─────────────────────────────────────────────

echo "[get_mlf] 探测 HDFS 路径: $HDIR"

# 列出所有数据文件：
#   - awk '$1 !~ /^d/'  : 只取普通文件，排除目录（_DONE 等）
#   - grep -v '/_'      : 排除 _SUCCESS / _logs 等 _ 前缀控制文件
# 兼容任意文件名（part-NNN、*.pak.N、自定义名称均可）
HDFS_FILES=$(hdfs dfs -ls "$HDIR" 2>/dev/null \
    | awk '$1 !~ /^d/ {print $NF}' \
    | grep -v '/[_\.]' || true)

N_FILES=$(echo "$HDFS_FILES" | grep -c '.' || true)

if [ "$N_FILES" -eq 0 ]; then
    echo "[error] HDFS 路径下没有找到数据文件: $HDIR"
    exit 1
fi

echo "[get_mlf] 发现 $N_FILES 个数据文件"

# 根据文件数量决定分片策略
if [ "$N_FILES" -le 10 ]; then
    # 文件少（含单文件）: 每个文件独立作为一个分片
    N_PARTS=$N_FILES
    USE_GLOB=0    # 直接用完整文件路径，不用 suffix glob
else
    # 文件多: 按末尾数字 0-9 分10组（与 1_get_pfile_from_hdfs.pl 一致）
    N_PARTS=10
    USE_GLOB=1
fi

echo "[get_mlf] 分片策略: N_PARTS=$N_PARTS  USE_GLOB=$USE_GLOB"
echo "[get_mlf] mode    : $MODE"
echo ""

# ─────────────────────────────────────────────
# 4. 下载单个分片
# ─────────────────────────────────────────────

# 验证 MLF 文件有效：存在 + 非空 + 含 #!MLF!# 头 + 至少一条语音（"."行）
validate_mlf() {
    local f=$1
    local label=$2
    if [ ! -s "$f" ]; then
        echo "[warn] 输出文件为空: $f"
        echo "       可能原因: 数据中不含 '$label' 字段，或解包工具静默失败"
        return 1
    fi
    if ! grep -q "^#!MLF!#" "$f" 2>/dev/null; then
        echo "[warn] 输出文件缺少 #!MLF!# 头: $f"
        return 1
    fi
    local n_utt
    n_utt=$(grep -c '^\.' "$f" || true)
    if [ "$n_utt" -eq 0 ]; then
        echo "[warn] 输出文件无有效语音条目 (无 '.' 结束符): $f"
        return 1
    fi
    return 0
}

download_part() {
    local i=$1          # 分片编号 (0-based)
    local label=$2      # mlf_fa_ph 或 mlf_sy
    local outfile=$3    # 输出文件名

    mkdir -p "./$i"
    local out="./$i/$outfile"
    local log="./$i/unpack_${label}.log"
    local done_flag="${out}.done"

    if [ -f "$done_flag" ]; then
        echo "[part $i/$label] 已完成，跳过"
        return 0
    fi

    # 确定本分片的 hdfs cat glob
    local hdfs_glob
    if [ "$USE_GLOB" -eq 0 ]; then
        # 单/少文件模式：取第 (i+1) 个文件路径
        hdfs_glob=$(echo "$HDFS_FILES" | sed -n "$((i+1))p")
    else
        # 多文件模式：按 suffix 筛选
        hdfs_glob="${HDIR}/*${i}"
    fi

    echo "[part $i/$label] 下载: $hdfs_glob"

    # pipefail 已开启，管道任意步骤失败都会被捕获
    hdfs dfs -cat "$hdfs_glob" \
        | "$BIN_UNPACK" - "./$i" "$label" \
        > "$log" 2>&1

    # 验证输出
    if validate_mlf "$out" "$label"; then
        local n_utt
        n_utt=$(grep -c '^\.' "$out" || true)
        echo "[part $i/$label] 完成: $n_utt 条语音"
        touch "$done_flag"
    else
        echo "[warn] 分片 $i 标签 '$label' 验证失败，详见: $log"
        # 不创建 .done，下次可重跑；不 exit，继续其他分片
    fi
}

# ─────────────────────────────────────────────
# 5. 并行下载所有分片
# ─────────────────────────────────────────────

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

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "[error] pid=$pid 异常退出"; fail=1; }
done
[ "$fail" -eq 1 ] && { echo "[error] 有分片异常退出"; exit 1; }

echo ""
echo "[get_mlf] 所有分片完成，开始合并..."

# ─────────────────────────────────────────────
# 6. 合并各分片为单个 MLF 文件
# ─────────────────────────────────────────────

merge_mlf() {
    local label_file=$1
    local out_mlf=$2

    # 收集实际存在的分片（validate 通过的）
    local found=0
    echo "#!MLF!#" > "$out_mlf"
    for ((i=0; i<N_PARTS; i++)); do
        local part="./$i/$label_file"
        if [ ! -f "${part}.done" ]; then
            echo "[merge] 跳过未完成的分片 $i ($label_file)"
            continue
        fi
        grep -v "^#!MLF!#" "$part" >> "$out_mlf"
        found=$((found+1))
    done

    if [ "$found" -eq 0 ]; then
        echo "[warn] $label_file: 所有分片均无有效数据，跳过合并"
        rm -f "$out_mlf"
        return
    fi

    local n_utt
    n_utt=$(grep -c '^\.' "$out_mlf" || true)
    echo "[merge] $out_mlf  ← $found 个分片  $n_utt 条语音"
}

if [ "$MODE" = "phone" ] || [ "$MODE" = "both" ]; then
    merge_mlf "out.mlf_fa_ph" "./out.mlf_fa_ph"
fi
if [ "$MODE" = "word" ] || [ "$MODE" = "both" ]; then
    merge_mlf "out.mlf_sy"    "./out.mlf_sy"
fi

# ─────────────────────────────────────────────
# 7. 汇总
# ─────────────────────────────────────────────

echo ""
echo "============================================"
echo "[get_mlf] 完成"
if [ -f "./out.mlf_fa_ph" ]; then
    echo "  phone MLF : ./out.mlf_fa_ph  ($(grep -c '^\.' out.mlf_fa_ph) utts)"
else
    echo "  phone MLF : 未生成 (数据中可能无 mlf_fa_ph 字段)"
fi
if [ -f "./out.mlf_sy" ]; then
    echo "  word  MLF : ./out.mlf_sy     ($(grep -c '^\.' out.mlf_sy) utts)"
else
    echo "  word  MLF : 未生成 (数据中可能无 mlf_sy 字段)"
fi
echo ""
echo "  -> 供 decode_wav.py 使用:"
echo "     --mlf $(pwd)/out.mlf_fa_ph --mlf_level phone"
echo "============================================"
