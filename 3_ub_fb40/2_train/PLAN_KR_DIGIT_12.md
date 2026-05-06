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
| 通用韩语 500h | 500h | 已制 pfile（`/yrfs4/.../lib_fb40/`） |
| 车载安静 | 8h | pfile 路径待填（`PLACEHOLDER_CAR_PFILE_DIR`） |
| TTS 联系人（增强后） | 39h | pfile 路径待填 |
| TTS 号码（增强后） | 78h | pfile 路径待填 |
| **真人车载 clean** | **10.5h** | **正在制 pfile** |
| **真人车载 RIR** | **10.5h** | **正在制 pfile** |
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
  - 新 `Rates`：通用 0.25 / 车载 0.5 / TTS 联系人 0.05 / TTS 号码 0.05 / 真人 Clean 0.6 / 真人 RIR 0.6 / 真人 12Hard 0.3

## 3. TODO（按依赖顺序，跨机器衔接看这里）

### A. 等 pfile 完成后再做

- [ ] **A1** 拿到真人 10.5h clean pfile 路径 → 填 `PLACEHOLDER_CAR_REAL_CLEAN_PFILE_DIR`
- [ ] **A2** 拿到真人 10.5h RIR pfile 路径 → 填 `PLACEHOLDER_CAR_REAL_RIR_PFILE_DIR`
- [ ] **A3** 1/2 hard-case 子集抽取
  - 输入：真人 10.5h 的 text + utt 列表
  - 命令草稿：`grep -E '(^|[^ㄱ-ㅎ가-힣])(일|이)([^ㄱ-ㅎ가-힣]|$)' text.list | awk '{print $1}' > car_real_12hard.uttlist`
  - 用同一套 pipeline 单独制 pfile → 填 `PLACEHOLDER_CAR_REAL_12HARD_PFILE_DIR`
- [ ] **A4** 检查并填其它 3 个 PLACEHOLDER（CarPfile / TTSContact / TTSNum），上一轮 0416 应该有路径，去 `OutDir = kr_car_tts_0416` 或 git log 找
- [ ] **A5** smoke test：跑 1 个 iter，确认 7 个 dataset 的 batch 都进得来，特别注意 CarReal12Hard 的 `CVSentNum=0` 不会触发边界问题

### B. 不依赖训练，今天就能 A/B（最高 ROI）

- [ ] **B1** 找出韩语字典里 일/이/삼/사/오/육/칠/팔/구/공 这 10 个 token 的 ID
  - 字典文件位置：上一轮 decode 用过，去 `1.run.sh` / `decode.sh` / `train.sh` 看 `--dict` 参数
  - 或：`grep -nE '^(일|이|삼|사|오|육|칠|팔|구|공)\s' <dict>`
- [ ] **B2** 用旧模型（`kr_car_tts_0416/model.iterX.partY`）+ beam decode + bias 跑测试集
  - 命令：`python decode_ubctc.py --model <旧模型> --feat <utt.npy> --dict <字典> --mode beam --beam 10 --bias-tokens "일,이,삼,사,오,육,칠,팔,구,공" --bias-value 0.4`
  - 试 0.0 / 0.4 / 0.7 / 1.0 四档，看 1/2 召回 vs 误激活的 trade-off
- [ ] **B3** 如果 B2 单独就把 1/2 拉回来，重训目标降级；否则按 A 计划继续

### C. 训练 + 评估

- [ ] **C1** A 全部完成后，启动新训练（OutDir = `kr_car_tts_0507`）
- [ ] **C2** 监控：通用 500h CV loss 不应该比 0416 起点更差（防遗忘）
- [ ] **C3** 评估指标
  - 真车回放测试集字准
  - 1/2 混淆矩阵（重点看 일↔이 错误率）
  - "打电话给 XXX" 漏字率
  - 通用域 CV（防 regression）
- [ ] **C4** 如效果好，把 boosting (B2) 的最优 bias_value 写入默认 decode.sh

### D. 后续可选优化（按 ROI 排）

- [ ] **D1** 离线加噪 SNR 多档：0/5/10/15/20dB（当前只有 5dB）
- [ ] **D2** 离线变速 0.9/1.0/1.1/1.2x（当前只有 1.2x）
- [ ] **D3** 把 일 label 显式拆成 "이+ㄹ" 强迫模型建模尾音（要改字典+重制 lab pfile）
- [ ] **D4** Phoneme-level 辅助 CTC head
- [ ] **D5** fb40 → fb80 + pitch（成本高，先观察 D1-D4）
- [ ] **D6** MWER / sequence-level fine-tune（CTC 完全收敛后）

明确**不做**：
- ~~AMR-NB / 8k 上采样模拟通话失真~~ — 车机近场采集，与训练数据格式一致，引入退化反而拉大 train/test gap

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
