# 韩语电话号码 TTS 数据生成流程

## 背景

针对韩语车载场景"일→이（ㄹ韵尾丢失）"、"육으→구（连读合并）"等误识问题，
通过多说话人、多语速、多连读模板的 TTS 合成，扩充电话号码训练数据。

**数据规模**：10000 条号码 × 8 变体（5 TTS + 3 语速）= 80000 条 wav，约 17-18 GB

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `phone_numbers_10k.txt` | 10000 条韩语电话号码输入（utt_id + 号码） |
| `gen_tts_kr_number.py` | TTS 生成脚本（Edge TTS，多声音/语速/连读） |
| `post_process_tts.py` | 后处理脚本（生成 HTK MLF 标注 + 压缩） |
| `upload_gdrive.py` | Google Drive 上传脚本（纯 Python stdlib） |
| `tts_num_10k/` | 输出目录（wav/ + manifest.txt + mlf + tar.gz） |

---

## 前置准备（一次性）

### 1. Google Drive OAuth 凭据

> 仅首次上传需要，token 缓存后无需重复操作。

1. 打开 https://console.cloud.google.com/
2. 新建项目（如 `tts-upload`）
3. **APIs & Services → Library** → 搜索 `Google Drive API` → Enable
4. **APIs & Services → Credentials → + Create Credentials → OAuth 2.0 Client ID**
5. Application type 选 **Desktop app** → Create
6. 点击 **Download JSON**，保存为：

```
3_ub_fb40/0_data_process/client_secret.json
```

### 2. Google Drive 存储空间

文件约 17 GB，需升级至 100GB 套餐（$1.99/月）：
https://one.google.com/

---

## 执行步骤

### Step 1：拉取最新代码

```bash
cd /home/lty/am/ubctc
git pull
```

### Step 2：继续生成 wav（断点续跑）

当前已生成约 23847 / 80000 条，已有的自动跳过，直接续跑：

```bash
nohup python3 3_ub_fb40/0_data_process/gen_tts_kr_number.py \
    3_ub_fb40/0_data_process/phone_numbers_10k.txt \
    3_ub_fb40/0_data_process/tts_num_10k/ \
    --variants 5 --stretch --jobs 4 \
    > 3_ub_fb40/0_data_process/tts_num_10k/gen.log 2>&1 &

echo "PID: $!"
```

**查看进度**（目标 80000）：

```bash
watch -n 30 "ls 3_ub_fb40/0_data_process/tts_num_10k/wav/ | wc -l"
```

**查看日志**：

```bash
tail -f 3_ub_fb40/0_data_process/tts_num_10k/gen.log
```

**完成标志**：日志出现 `Done. 80000 wavs`，进程自动退出。

预计耗时：约 4-5 小时（剩余 56153 条 × 5 TTS 请求）

---

### Step 3：生成 HTK MLF 标注 + 压缩

TTS 生成完成后执行（会自动等待，也可手动在完成后运行）：

```bash
python3 3_ub_fb40/0_data_process/post_process_tts.py \
    3_ub_fb40/0_data_process/tts_num_10k/
```

输出：
- `tts_num_10k/tts_kr_num.mlf` — HTK 格式标注，每行一个韩语音节
- `tts_num_10k/tts_kr_num.tar.gz` — wav + mlf + manifest 打包，约 17 GB

MLF 格式示例：
```
#!MLF!#
"*/num000001_v000.lab"
공
일
공
일
이
삼
사
오
육
칠
팔
.
```

---

### Step 4：上传到 Google Drive

```bash
python3 3_ub_fb40/0_data_process/upload_gdrive.py \
    3_ub_fb40/0_data_process/tts_num_10k/tts_kr_num.tar.gz \
    --client-secret-file 3_ub_fb40/0_data_process/client_secret.json \
    --folder-name tts_kr_num
```

首次运行会显示：
```
Open this URL in your browser:
  https://www.google.com/device
Enter code: XXXX-XXXX
Waiting for authorization ..........
```

浏览器打开链接 → 输入验证码 → 登录 Google 账号授权，脚本自动开始上传。

- 支持断点续传（32 MB/chunk）
- Token 缓存到 `~/.gdrive_token.json`，下次无需重复授权
- 完成后输出 Google Drive 文件链接

---

## 数据配比建议（更新后）

新 TTS 数据（222h）加入训练后，`config_car_tts.ini` 建议配置：

```ini
TrainDatas = DataSettingCar, DataSettingTTSContact, DataSettingTTSNum, DataSettingCarReal, DataSetting
Rates      = 1.0,            0.12,                  0.05,              0.05,               0.25
```

| 数据集 | 时长 | Rate | 有效时长 | 占比 |
|---|---|---|---|---|
| 通用韩语 | 500h | 0.25 | 125h | 84% |
| 车载安静 | 8h | 1.0 | 8h | 5.4% |
| TTS 号码（新）| 222h | 0.05 | 11.1h | 7.5% |
| TTS 联系人 | 39h | 0.12 | 4.7h | 3.2% |
| 实车录音 | ~0.2h | 0.05 | ~0h | 信号注入 |

---

## 常见问题

**Q: TTS 生成中断怎么办？**
重新执行 Step 2 命令即可，脚本自动跳过已生成的 wav。

**Q: Edge TTS 报 503 错误？**
脚本已内置指数退避重试（最多 5 次），短暂限流会自动恢复。
若频繁失败，可降低并发：`--jobs 2`

**Q: 上传中断？**
重新执行 Step 4，`upload_gdrive.py` 使用 Google Drive 可恢复上传协议，
会自动从中断处继续（注意：需重新执行命令，不是自动续传）。

**Q: Google Drive 空间不足？**
升级至 100GB 套餐：https://one.google.com/
```
