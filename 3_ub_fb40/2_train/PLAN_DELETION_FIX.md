# 韩语号码删除错误修复方案（叠字漏字 + 变长场景）

最近更新：2026-05-08
负责文件：`net_ubctc.py`、`config_car_tts.ini`、`../0_data_process/gen_tts_kr_number.py`、WFST 团队对接

## 0. 问题定位

基于 baseline (`kr_car_phone_gbz_0507`) 错误分析（400 条真车测试集）：

| 错误类型 | 数量 | 占比 |
|---|---|---|
| **删除（D）** | **51** | **55%** |
| 替换（S） | 30 | 33% |
| 插入（I） | 12 | 13% |
| **合计错误** | **93** | 100% |

删除错误细分：

| 删除模式 | 估算占比 | 典型 case |
|---|---|---|
| **连续相同数字漏字（叠字漏字）** | **~30%** | 이이→이、오오→오、칠칠→칠、팔팔→팔、구구→구、사사→사 |
| 非重复中间漏字 | ~15% | 오 이 이 → 오 이（叠字延伸）|
| 句尾命令词漏字 | ~10% | 줘 缺失 |
| 句首漏字 | ~5% | 일 漏 |

**叠字漏字是最大头**，本质是 CTC 架构限制：连续相同 token 必须 blank 隔开，否则被折叠。

## 1. 关键约束 — 变长场景

韩国电话号码**长度不固定**：

| 类型 | 总位数 | 前缀 |
|---|---|---|
| 手机（新）| 11 | 010 |
| 手机（旧）| 10 | 011/016/017/018/019 |
| 固话首尔 | 8-9 | 02 |
| 固话其他 | 9-10 | 031-064 |
| 网络电话 | 11 | 070 |
| 免费电话 | 9-10 | 080 |
| 服务号 | 8 | 1588/1599/1644 |

**严格 11 位约束不可行**。需要变长友好的修复方案。

## 2. WFST 侧方案（不依赖固定长度）

### W1. 基于前缀的条件长度预期

```
检测前缀 → 推断期望总长度
  010      → 11 位
  011/016~019 → 10 位
  02       → 9-10 位
  1588/1599/1644 → 8 位
  031-064  → 10 位
  070      → 11 位
  080      → 9-10 位
  其他/未知 → 不约束
```

不是强制，是**软偏置**：检测到前缀后给该长度的 path 加 bonus。覆盖韩国 80%+ 真实号码。

### W2. Consecutive-same blank-aware decoding（针对叠字最关键）

CTC 经典病在解码侧的修复。WFST 解码时扫描候选 path：

```
检测 path 中位置 i 出现 "Token T, blank, Token T" 的 pattern
  → 给该位置的 blank 单独加 bonus（cost - 0.5）
  → 让保留两个 token 的 path 比折叠成一个 的 path 更便宜
```

**直接打叠字漏字 30%**，不依赖任何长度约束。**最高 ROI WFST 方案**。

### W3. 软长度分布偏置

基于韩语电话号码长度分布给小 bonus：

```
11 位 path: bonus -0.3 (比基线便宜 0.3)
10 位 path: bonus -0.2
 9 位 path: bonus -0.15
 8 位 path: bonus -0.1
 其他: 0
```

不强制错误长度，鼓励模型不丢字。

### W4. 域 N-gram LM（韩语电话号码专用）

用真实电话号码语料训小 N-gram LM，N-best 重打分：
- 学到 "010 后面常跟数字"
- 学到 "1588 后面 4 位"
- LM 自然抑制不合理长度

### W5. Contextual bias FST（已在 PLAN B 节）

检测到"打电话/번호" 触发词后，对数字 token 加 bias，叠加上面的方案。

## 3. 训练侧方案（直接打叠字漏字）

### T1. ❌ TTS 加长叠字 pause（已弃用，存在严重训练-测试分布偏差）

**初始想法**：合成时对连续相同数字插入 silence。

**致命问题**：
- 训练 audio 有 silence → 模型学"看见 silence 输出 sil"
- 测试 audio（真实连读）**没有 silence** → 模型仍然连续输出 이 senone → CTC 仍然折叠
- **没解决根本问题**，反而引入分布偏差
- 多叠字（이이이）情况更糟糕，silence 越多与真实差距越大

**结论**：T1 是错误方向，已从推荐列表删除。叠字漏字的根因不是"训练没见过 silence"，
而是 **模型在连续同 token 边界帧没学到主动输出 blank**。要从这个角度下手。

### T1'. TTS voice 多样性扩量（正确的 TTS 改进方向）

不加人为 silence，而是用**多 voice + 多语速**让模型见到自然变化：

```
当前: 1-2 个 voice, 单一语速
改进: 20+ voice × 3 语速 (0.85x/1.0x/1.15x) = 60+ 组合

每个数字 token，包括叠字 pattern，都会有 60+ 不同的合成样本
不同 voice 的发音节奏 + 不同语速 → 自然产生叠字时长多样性
模型在多种变化下学会处理叠字（不依赖人为 silence）
```

属于 PLAN D1.3，工时 1-2 周（生成 + pfile 制作）。

### T2. 真车数据 hard negative mining

用现有 baseline 扫真车 10.5h，找出**含叠字 pattern 且现在错**的样本：

```python
# 抽样脚本伪代码
for utt in real_car_data:
    ref = read_ref(utt)
    hyp = decode(baseline_model, utt)
    
    # 含叠字 pattern
    has_consec_same = any(ref[i] == ref[i+1] for i in range(len(ref)-1))
    # 现在被折叠了
    is_collapsed = len(hyp) < len(ref) and not strictly_correct(ref, hyp)
    
    if has_consec_same and is_collapsed:
        save_to_hard_pfile(utt)
```

成 section 加入 TrainDatas，Rate=1.5。

### T3. CTC blank-between-same regularizer（治本但工程量大）

修改 loss 计算，**显式强制模型在连续相同 token 之间输出 blank**：

```python
# 伪代码
for utt in batch:
    target_seq = ref_sequence(utt)
    for i in range(len(target_seq) - 1):
        if target_seq[i] == target_seq[i+1]:
            # 找该叠字对在 FA 对齐中的帧边界
            boundary_frame = fa_alignment.boundary(utt, i)
            # 在该帧上加 blank emission bonus
            blank_loss = -log_probs[boundary_frame, blank_idx]
            regularizer += blank_loss

total_loss = ctc_loss + λ_reg * regularizer
```

依赖 FA 帧级对齐，需要改 dataloader 输出 alignment + 改 loss。工时 ~1 周。

## 4. 推荐组合策略

按性价比 + 实现难度排序：

### 第一波（这周做）— 快路

| 措施 | 责任方 | 预期增益 | 状态 |
|---|---|---|---|
| **aux CE 训练**（已实现）| 你 | +1-3（替换 + coda）| 待启动 |
| **W2: WFST consecutive-same blank-aware decoding 提需求** | WFST 团队 | +3-5（最强叠字治理）| 待提需求 |
| 域 LM rescore | WFST 团队 | +0.5-1 | 提需求 |
| T2: 真车 hard mining 重训 | 你 | +1-2 | 待实施 |

### 第二波（1-2 周）— 中路

| 措施 | 责任方 | 预期增益 |
|---|---|---|
| W2 实施落地 | WFST 团队 | +3-5 |
| W1: 基于前缀的软长度偏置 | WFST 团队 | +1-2 |
| W3: 软长度分布偏置 | WFST 团队 | +0.5-1 |
| **T1': TTS voice 多样性扩量**（不加 silence）| 你 + TTS 团队 | +1-2 |

### 第三波（1 个月+）— 慢路 / 治本

| 措施 | 预期增益 | 备注 |
|---|---|---|
| **T3: CTC blank-between-same regularizer** | **+2-3** | **声学侧治本**，需 FA 接入 |
| RNN-T 架构升级 | +5+ | 终极方案，工程量极大 |
| 真车数据扩量到 50h+ | +5-10 | 长期最重要 |

## 5. 当前最高 ROI 动作 — 立刻就做

**两个并行**：

### A. 启动 aux CE 训练（声学侧已 ready）

代码已实现并 commit (`9b3335d`)，直接启动：

```bash
unset CODA_BOOST
export CODA_MASK_PATH=/yrfs4/.../coda_senone_mask.pt
export AUX_CE_WEIGHT=0.3
export CODA_CE_BOOST=3.0
sed -i 's/OutDir = .*/OutDir = kr_car_phone_gbz_0507_auxCE/' config_car_tts.ini
bash train.sh
```

**预期效果**：1↔2 替换错误降 30%+、6→유기 降 50%+，整体句准 +1-3。
**不直接解叠字漏字**（需要 W2 配合）。

### B. 给 WFST 团队的需求文档（解码侧最重要）

```
[需求] WFST 解码侧针对韩语号码叠字漏字优化

背景：
  - 当前 baseline 错误中删除占 55%（D=51/93）
  - 其中约 30% 是连续相同数字漏字（叠字）
  - 这是 CTC 架构限制（blank-fold 机制）

约束：
  - 韩国电话号码长度不固定（8/9/10/11 位混合）
  - 不能用严格长度约束

请实现的优先级：

P0 (必做): Consecutive-same blank-aware decoding
  扫描候选 path，对 "Token T, blank, Token T" pattern 中的 blank
  单独加 bonus (cost - 0.5)，让保留双 token 的 path 更便宜

P1: 基于前缀的软长度偏置
  检测号码前缀 (010/011/02/1588/070/080/031-064 等)
  推断期望长度，给该长度 path 加 bonus -0.3

P2: 软长度分布偏置（韩国号码长度分布）
  按 11/10/9/8 位分别加 bonus

P3: Contextual bias FST
  电话场景触发词后给数字 token 加权（已在 PLAN B 节）

P4: 域 N-gram LM rescore
  电话号码专用 LM 重打分

预期效果：
  - 叠字漏字救回 70%+ → 整体句准 +3-5
```

## 6. 评估指标（实施后必看）

每次改动后跑 `scripts/analyze_stat.py` 对比 baseline，重点看：

| 指标 | aux CE 期望 | TTS pause 期望 | WFST W2 期望 |
|---|---|---|---|
| 1↔2 替换 | -30~-50% | 微小 | 间接 |
| 6→유기 | -50%+ | 微小 | 间接 |
| 叠字漏字 | -10~-20%（间接）| -30~-50% | **-70%+**（直接）|
| 句首/句尾漏字 | -10~-20% | 间接 | 间接 |
| 总句准 | +1~3 | +2~4 | +3-5 |

## 7. 决策记录

- **2026-05-08**：变长约束确认 — 韩语号码长度不固定，放弃严格 11 位约束方案
- **2026-05-08**：错误诊断结果显示删除 55%、叠字漏字 30%，转重点攻破方向
- **2026-05-08**：D2.1 aux CE 实现完成（commit `9b3335d`），主攻替换错误 + 部分 coda 漏字
- **2026-05-08**：**T1 (TTS 加叠字 pause) 弃用**。原因：训练加 silence 教会模型"看见 silence 输出 sil"，
  但真实测试集叠字间没 silence，模型仍连续输出 이 senone 被 CTC 折叠，**没解决根因**还引入分布偏差。
  叠字漏字根因是模型在边界帧没主动输出 blank，需要 W2 (WFST) 或 T3 (训练侧 regularizer) 直接打。
- **TODO**：WFST consecutive-same blank-aware decoding 需求文档递交（最高 ROI）；
  aux CE 训练启动（替换错误改进）；
  T1' TTS voice 多样化扩量（D1.3，1-2 周）

## 8. 速查 — 改动文件清单

| 文件 | 修改内容 | 状态 |
|---|---|---|
| `net_ubctc.py` | aux CE loss 集成 | 已完成 |
| `asr/layers/loss.py` | CodaWeightedCeLoss | 已完成 |
| `scripts/analyze_stat.py` | 错误分析工具 | 已完成 |
| `scripts/find_coda_senones.py` | 生成 coda mask | 已完成 |
| `scripts/strip_coda_buffer.py` | checkpoint 清理工具 | 已完成 |
| ~~`../0_data_process/gen_tts_kr_number.py` (加 silence)~~ | ~~T1 已弃用~~ | ❌ 不做 |
| `../0_data_process/gen_tts_kr_number.py` (多 voice 多语速) | T1' 自然多样性 | 待做 |
| WFST 团队 - HCLG | **W2: consecutive-same blank-aware** | **待提需求（最高优先级）** |
| WFST 团队 - HCLG | W1: 前缀长度偏置 | 待提需求 |
| WFST 团队 - HCLG | W3: 软长度分布偏置 | 待提需求 |
| 训练侧 (后续) | T3: CTC blank-between regularizer | 中长期 |

## 9. 当前状态总结

- **正在训 / 待启动**: aux CE 训练（基于 +1.1 baseline）— 主攻替换错误
- **必须启动**: WFST team 的 W2 (consecutive-same blank-aware decoding) — 主攻叠字漏字
- **已弃用**: T1 (TTS pause padding) — 分布偏差严重，不解决根因

**预期时间线**：
- 1-2 周：aux CE 训完，替换错误降 → 句准 +1-3
- 2-4 周：WFST W2 上线，叠字漏字降 → 句准 +3-5
- 4-6 周：W1+W3 上线，软长度偏置 → 句准 +1-2
- 1-2 个月：T3 训练侧治本（如果前面几个加起来还不够）

**累计目标**：从当前 86.5% (+1.1 baseline) → 目标 92-95% 实车句准
