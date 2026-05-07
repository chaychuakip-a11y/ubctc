# 韩语 1(일) / 2(이) 漏识别 + 混淆 优化计划

最近更新：2026-05-07
负责文件：`config_car_tts.ini`、`decode_ubctc.py`、`../0_data_process/`

## 0. 问题与根因

- 现象：韩语车载"打电话给 XXX" / "打电话 1234567" 丢字 + 1/2 互混
- 根因（按贡献度排序）
  1. **声学**：일(/il/) 与 이(/i/) 唯一区别是末尾 ㄹ(/l/)，车舱混响 + 弱尾音先丢
  2. **数据分布偏移**：TTS 合成 일/이 边界比真人清晰，模型在 train 上够，test 上不够
  3. **真人车载数据严重不足**：上一轮只有 ~80 条
  4. **fine-tune LR 偏高**：上一轮 0.0003，把通用 500h 的稳健能力洗掉了

## 1. 数据现状（2026-05-07）

| 数据集 | 量 | 状态 |
|---|---|---|
| 主集（通用 400h + 车载 400h，离线仿真） | ~800h | 已制 pfile（`/yrfs4/.../lib_fb40/`） |
| 真人车载 legacy（0416 遗留） | ~80 条 | 单独成 section，Rate=0.05，pfile 路径待填 |
| TTS 合并（Contact 39h + Num 78h，增强后） | ~117h | pfile 路径待填（合一个 section） |
| **真人车载 clean** | **10.5h × 2.8 = 29.4h（增强后）** | **正在制 pfile** |
| **真人车载 RIR** | **4h** | **正在制 pfile** |
| 真人 1/2 hard-case 子集 | 待抽 | 抽取脚本待写 |

## 2. 已完成（commit 历史）

- `c497027` TTS pipeline 文档
- `6088aec` post_process + Google Drive 上传脚本
- `d777cbd` TTS 生成脚本加 retry/resume
- `f26fa9f` TTS 多 voice 数据生成（号码场景）
- `41e1057` `tts_num_10k/wav/` 进 .gitignore，保留两份 log
- **本次提交**（见下面 commit message）：
  - `decode_ubctc.py` 加 `--bias-ids/--bias-tokens/--bias-value`，CTC prefix beam 支持数字 boosting
  - `config_car_tts.ini` 升级到 0507 版本：LR 0.0003→0.00015、Iter 20→12、Half 6→4
  - `config_car_tts.ini` 拆出三个真人数据 section：Clean / RIR / 12Hard，路径留 PLACEHOLDER
  - 新 `Rates`（DataSetting 必须首位、Rate=1.0 作锚点；其余为相对权重）：
    - DataSetting（主集 800h 通用+车载） **1.0**
    - DataSettingTTS（合并）**0.4**
    - DataSettingCarRealLegacy（0416 遗留）**0.2**
    - DataSettingCarRealClean（29.4h）**2.4**
    - DataSettingCarRealRIR（4h）**0.4**
    - ~~DataSettingCarReal12Hard~~ 暂去掉（pfile 待制作），就绪后加回 Rate=1.2

## 3. TODO（按依赖顺序，跨机器衔接看这里）

**主路线**：**A**（填 pfile 路径 + 抽 hard-case）→ **C**（启动 0507 训练 + 评估）。如不达标，进 **D**（声学侧 fallback 三阶段）。**B**（WFST 团队对接）并行推进，不阻塞主路线。

### A. 等 pfile 完成后再做

- [ ] **A1** 拿到真人 10.5h clean pfile 路径 → 填 `PLACEHOLDER_CAR_REAL_CLEAN_PFILE_DIR`
- [ ] **A2** 拿到真人 10.5h RIR pfile 路径 → 填 `PLACEHOLDER_CAR_REAL_RIR_PFILE_DIR`
- [ ] **A3**（暂跳过）1/2 hard-case 子集抽取 + pfile — 当前 0507 训练**不含 12Hard**，待此项就绪后再加回（取消注释 [DataSettingCarReal12Hard]，TrainDatas+Rates 加回 1.2）
  - 输入：真人 10.5h 的 text + utt 列表
  - 命令草稿：`grep -E '(^|[^ㄱ-ㅎ가-힣])(일|이)([^ㄱ-ㅎ가-힣]|$)' text.list | awk '{print $1}' > car_real_12hard.uttlist`
  - 用同一套 pipeline 单独制 pfile → 填 `PLACEHOLDER_CAR_REAL_12HARD_PFILE_DIR`
- [ ] **A4** 检查并填其它 2 个 PLACEHOLDER（TTS 合并 / CarRealLegacy），上一轮 0416 应该有路径，去 `OutDir = kr_car_tts_0416` 或 git log 找
- [ ] **A5** smoke test：跑 1 个 iter，确认 7 个 dataset 的 batch 都进得来，特别注意 CarReal12Hard 的 `CVSentNum=0` 不会触发边界问题

### B. WFST 团队对接需求（解码侧线上是 WFST，原 Python boosting 搁置）

**背景**：线上解码用 WFST，`decode_ubctc.py --bias-tokens` 仅是离线 A/B 工具不会上线。原 B 节（Python beam + bias 跑 baseline 验证可 boost 多少回报）当前跑不起来且暂无时间修复，搁置。以下诉求需 WFST 团队落地：

- [ ] **B1** **Contextual bias FST**：检测到"打电话/번호/전화" 等触发词后，对号码 token (일이삼사오육칠팔구공) 路径加权 +0.4~+0.7
- [ ] **B2** **格式约束 + N-best 后处理**：号码场景数字 token 必须连续 N 位（韩国手机 11 位），不合法回退第二候选
- [ ] **B3** **Class-based LM**：电话号码作为 class，电话场景下提升 class 先验
- [ ] **B4** **Confusion pair 显式建模**：LM 里给 (일, 이) 加双向 fallback path
- [ ] **B5** **场景化 HCLG**：车载电话场景专用 graph，与通用域分离

**优先级**：B1 > B2 > B3 > B4 > B5（B1+B2 即可覆盖大部分场景，B3-B5 是后置增量）

**当 Python boosting 修好后**：仍可作为离线工具用来定上限 — 跑 +0.0/0.4/0.7/1.0 四档看 1/2 召回 vs 误激活 trade-off，把最优 bias_value 当作 B1 的参数提给 WFST 团队

### C. 训练 + 评估

- [ ] **C1** A 全部完成后，启动新训练（OutDir = `kr_car_tts_0507`）
- [ ] **C2** 监控：通用 500h CV loss 不应该比 0416 起点更差（防遗忘）
- [ ] **C3** 评估指标
  - 真车回放测试集字准
  - 1/2 混淆矩阵（重点看 일↔이 错误率）
  - "打电话给 XXX" 漏字率
  - 通用域 CV（防 regression）
- [ ] **C4** 如效果好，把 boosting (B2) 的最优 bias_value 写入默认 decode.sh

### D. 0507 不达标后的 fallback 路线图（按声学侧 ROI 排序）

**前提**：解码不在我们控制范围（WFST 团队管，见 B 节），所有改动靠 acoustic + 数据 + loss。原"可选优化"升级为明确路线图：训完 0507 评估若不达标，按阶段顺序推进。

#### D 阶段 1 — 数据/采样调整（不改模型结构，复用 0507 训练框架）
- [ ] **D1.1** **Hard negative mining**：用 0507 模型扫真人 10.5h，把 1/2 错分 utt 单独抽成 section，Rate=0.5
- [ ] **D1.2** **类别加权 CTC loss**：含 일/이 的帧 loss × 2~3，需改 `train_fun.py` loss 计算
- [ ] **D1.3** **TTS voice 大幅扩量**：voice 多样性 > 合成时长，目标 20+ 不同发音风格 voice（针对"边界过清晰"问题）
- [ ] **D1.4** 离线多档 SNR：扩到 0/5/10/15/20dB（原 D1，当前只有 5dB）
- [ ] **D1.5** 离线多档变速：扩到 0.9/1.0/1.1/1.2x（原 D2，当前只有 1.2x）

#### D 阶段 2 — 针对 1/2 根因的模型/loss 改动
- [ ] **D2.1** **Phoneme/Jamo 辅助 CTC head**（原 D4）：韩语 Jamo 拆分（초성/중성/종성），给 ㄹ 单独监督信号 — **针对 1/2 根因最直接**
- [ ] **D2.2** **主动学习**：用 0507 模型扫未标注车载，挑 confidence 低样本送标注扩量

#### D 阶段 3 — 重型武器（成本高，最后考虑）
- [ ] **D3.1** fb40 → fb80 + pitch（原 D5）：尾音能量低，更高分辨率特征理论上有帮助
- [ ] **D3.2** MWER / sequence-level fine-tune（原 D6）：CTC 完全收敛后再做
- [ ] **D3.3** 字典级 일 → "이+ㄹ" 拆分（原 D3）：根因解法但改动巨大，重制字典 + lab pfile
- [ ] **D3.4** 教师蒸馏：通用 500h + 大数据 teacher 蒸到 student

**升级触发条件**：
- 阶段 1 → 2：D1 全做完且 1/2 错误率仍 > 5%
- 阶段 2 → 3：D2 全做完且 1/2 错误率仍 > 3%

明确**不做**：
- ~~AMR-NB / 8k 上采样模拟通话失真~~ — 车机近场采集，与训练数据格式一致，引入退化反而拉大 train/test gap
- ~~Python decode_ubctc.py boosting 上线~~ — 线上是 WFST，需求改走 B 节

## 4. 关键路径速查

- 训练入口：`train.sh` → `train.py`
- 模型：`net_ubctc.py`（Ubctc）
- 配置：`config_car_tts.ini`
- Decode：`decode_ubctc.py`（已加 boosting）/ `decode_pfile.py` / `decode_wav.py`
- 离线增强 perl 脚本：`../0_data_process/2.1_AddNoise_*.pl`、`3.0_speedup1.2.pl`、`9_fea_merge.pl`
- TTS 生成：`../0_data_process/gen_tts_kr_number.py` + `post_process_tts.py`
- RIR 文件：`/work1/asrdictt/ssyan2/.../reverb_pak_real_135m.npy`
- 上一轮模型：`kr_car_tts_0416/`（init from `/yrfs4/.../train_ctc_init_ce/model.iter13.part4`）

## 5. 决策记录

- **真人 10.5h 拆 Clean+RIR 两个 section**（而不是合并）：方便后续单独调比例，看哪一边对 1/2 召回贡献大
- **TTS 号码 Rate 从 0.08 降到 0.05**：合成边界过清晰，过度训反伤真人数据上的鲁棒性
- **LR 从 0.0003 降到 0.00015**：上一轮经验是 0.0001 太低、0.0003 偏高，取中
- **Iter 12 而非 20**：fine-tune 收益主要在前 8-10 轮，多训易过拟合到车载小数据
- **CarReal12Hard `CVSentNum=0`**：子集太小，UnionDataLoader 在 batch_num<=0 时会报边界问题（参考 `train_fun.py:308`）
