# GRPO 探针实验 · 快速启动 TODO 清单

> **定位**：AutoDL 单卡（RTX 4090/3090 24GB）上把 GRPO 探针（merge → 训练 → rollout 分流）从零跑通并回填实测数字的**一步步操作清单**。前置：SFT 探针已训完（有 adapter，见 `TODO_SFT探针快速启动.md`）。
> **口径权威源**：`notes/舆情负面精判_GRPO训练实战手册.md`（训练/Reward）、`notes/舆情负面精判_GRPO数据准备实战手册.md` §五（四维 Reward 契约）、`tests/README.md` §3（脚本用法）。冲突以上游为准。
> **纪律**：跑出来的数才回填；没跑的一律标 `[待实测]`，不编数。本清单勾完即把 GRPO 手册的「训练与 rollout 从未实跑」红线解除。
>
> **环境**：与 SFT 共用同一个虚拟环境（Step 0 已装齐 `trl>=0.19` / `peft>=0.12` / `transformers>=4.55`），无需重装。wandb 开关同 SFT（`SYNC_WANDB=1` 才上传，项目名 `sentiment-grpo-probe`）。
>
> **路径约定（AutoDL，与 SFT 清单对齐）**：模型下到 `/root/autodl-tmp/models`、数据下到 `/root/autodl-tmp/data`。
> - 探针数据上传：本机执行 `scp tests/data/grpo_prompts_probe.jsonl root@<autodl>:/root/autodl-tmp/data/grpo/`；
> - 下方命令一律用 AutoDL 绝对路径；本机跑则换成仓库相对路径（`models/`、`data/`）；
> - `train_grpo.py` / `run_grpo_probe.py` 的 `--data` / `--model_name_or_path` / `--out` 全是显式参数，不依赖仓库布局。

---

## Step 1 · SFT adapter 合并（GRPO 的起点）

> GRPO 只认「一个完整模型目录」，不认 adapter——先合并（原理见 GRPO 手册 §1.1：TRL 算 KL 走 `disable_adapter()`，双 adapter 叠加会让 reference 退化成裸基座）。

- [ ] **1.1** 确认 SFT adapter 在位：
  ```bash
  ls output/probe_lora_*/adapter_config.json   # 取最新时间戳那个
  ls models/Qwen3-4B-Instruct-2507/config.json # 基座在位
  ```
- [ ] **1.2** 合并体检（不加载模型，任何环境可跑）：
  ```bash
  python3 scripts/merge_sft_adapter.py --dry_run \
      --base /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --adapter output/probe_lora_<时间戳> \
      --out /root/autodl-tmp/models/Qwen3-4B-Instruct-SFT
  ```
  预期：路径体检通过；若打 `[WARN] adapter 训练时记录的基座是…` 说明 `--base` 与训练时不一致，**先核对再合并**。
- [ ] **1.3** 真合并（GPU 上跑，含 logits 一致性校验）：
  ```bash
  python3 scripts/merge_sft_adapter.py \
      --base /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --adapter output/probe_lora_<时间戳> \
      --out /root/autodl-tmp/models/Qwen3-4B-Instruct-SFT
  ```
  预期最后一屏：`合并完成` + `argmax 相同=True` + `merge_manifest.json` 落盘。
  **若 parity FAIL：拒绝落盘是设计行为**——先查 adapter 是不是这个基座训出来的，不要 `--no_verify` 绕过。
- [ ] **1.4** 验证据链：`cat /root/autodl-tmp/models/Qwen3-4B-Instruct-SFT/merge_manifest.json`
  → 记录 `parity.max_abs_logit_diff` 与 `argmax_equal`（回填 Step 5 表 A0）。

> **首跑留意**：merge 路径（`merge_and_unload` + parity）从未在真 GPU 上跑过，本步是它的首验。失败先看 peft 版本（≥0.12）。

---

## Step 2 · 训练前体检（不碰 GPU 也要全过）

- [ ] **2.1** 数据在位确认：
  ```bash
  wc -l /root/autodl-tmp/data/grpo_prompts_probe.jsonl    # 51 条
  ```
- [ ] **2.2** 干跑体检（数据契约 + 预算表 + ZAM 自检，零 GPU）：
  ```bash
  python3 scripts/train_grpo.py \
      --data /root/autodl-tmp/data/grpo_prompts_probe.jsonl \
      --eval_file /root/autodl-tmp/data/grpo/grpo_eval_probe.jsonl --dry_run
  ```
  逐项核对：
  - `训练规模预算` 表 11 行每个数可复算（8 题/步 × G=4 = 32 轨迹/步、51 题 → 7 步）；
  - `input_text 反解失败: 0 条 ✅`（不为 0 = prompt 格式坏了，打分会全池短路，先修数据）；
  - `ZAM 自检 PASS`；
  - `eval 基线集体检: … (8 条)`（`--eval_file` 给了才打印这一行）。
- [ ] **2.3** 造对齐税迷你基线集（8 条，从 51 条里按极性×难例分散抽；正式 3,000 条阶段换 k=4 封存集）：
  ```bash
  python3 - <<'PY'
  import json
  rows=[json.loads(l) for l in open("/root/autodl-tmp/data/grpo_prompts_probe.jsonl",encoding="utf-8") if l.strip()]
  picked, seen = [], set()
  for r in rows:
      key=(r["gold"]["sentiment"], r["gold"]["hard_case_type"])
      if key not in seen: seen.add(key); picked.append(r)
      if len(picked)>=8: break
  with open("/root/autodl-tmp/data/grpo/grpo_eval_probe.jsonl","w",encoding="utf-8") as f:
      f.writelines(json.dumps(r,ensure_ascii=False)+"\n" for r in picked)
  print("eval 基线集:", len(picked), "条")
  PY
  ```

---

## Step 3 · GRPO 探针训练（先冒烟后正式）

- [ ] **3.1** 冒烟（前台，~3 步，验证「能采样、能打分、能更新」三条腿）：
  ```bash
  python3 scripts/train_grpo.py \
      --data /root/autodl-tmp/data/grpo_prompts_probe.jsonl \
      --model_name_or_path /root/autodl-tmp/models/Qwen3-4B-Instruct-SFT \
      --output_dir /root/autodl-tmp/output/grpo_smoke \
      --eval_file /root/autodl-tmp/data/grpo/grpo_eval_probe.jsonl \
      --limit 8 --max_steps 3 --save_steps 2 \
      --log_file logs/grpo_smoke.log
  ```
  冒烟验收 4 条：
  - 启动横幅：`起点模型 … (已合并 SFT adapter · r=16 α=32 …)`；
  - `[EVAL] baseline reward 均值 = …` 出现（eval 链路通）；
  - 训练步里 reward 数值在 `[−2.5, +2.0]` 内、且**不全等于 −2.5**（全 −2.5 = 全池短路，回到 Step 2 查数据）；
  - `/root/autodl-tmp/output/grpo_smoke/` 有 checkpoint（save_steps=2 → step2 一份）。
- [ ] **3.2** 正式探针（后台，51 题 × 1 epoch = 7 步）：
  ```bash
  DATA=/root/autodl-tmp/data/grpo_prompts_probe.jsonl \
  MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-SFT \
  OUTPUT_DIR=/root/autodl-tmp/output/grpo_lora \
  EVAL_FILE=/root/autodl-tmp/data/grpo/grpo_eval_probe.jsonl \
  ./scripts/run_grpo_train.sh
  # 看进度: tail -f logs/grpo_train_*.log    停: kill $(cat train_grpo_pid.txt)
  ```
  产物：`/root/autodl-tmp/output/grpo_lora/`（adapter + `grpo_metrics.json`）+ `logs/grpo_train_*.log`。
- [ ] **3.3** 收尾核对 `/root/autodl-tmp/output/grpo_lora/grpo_metrics.json`：超参快照（G/T/lr/β/batch）与 Step 2.2 预算表逐项一致 → 回填 Step 5 表 A。

---

## Step 4 · rollout 与 k 分流（筛出真正的训练池）

- [ ] **4.1** 真实 rollout（SFT 合并模型作策略起点，同题 G=4）：
  ```bash
  python3 scripts/run_grpo_probe.py \
      --data /root/autodl-tmp/data/grpo_prompts_probe.jsonl \
      --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-SFT \
      --g 4 --temperature 0.7 --max_new_tokens 512 \
      --out /root/autodl-tmp/output/grpo/grpo_probe_kstats.json
  ```
  > 严格说策略起点应带 GRPO adapter（`--adapter /root/autodl-tmp/output/grpo_lora`）——本步先用合并模型跑 k 分布看 SFT 端的「摇摆度」，GRPO 训完再跑一次对比 k 分布迁移（两份 kstats 都留档）。
- [ ] **4.2** 读 k 直方图（脚本自动打印）：
  - **健康信号**：k=1~3 占比显著非零（有梯度可学）；k=0 与 k=4 两头都有（说明筛选有意义）。
  - **坏信号与处置**：全 k=4 → SFT 已收敛到位，GRPO 无题可训（探针 51 条上属正常，正式池要靠更难的题）；全 k=0 → 打分或数据坏了，回 Step 2 查。
- [ ] **4.3** 产物留档（把两个文件下载回本机 `tests/output/grpo/` 归档）：`grpo_probe_kstats.json`（k 直方图 + 训练池规模）+ `grpo_probe_kstats.rows.jsonl`（逐题明细）→ 回填 Step 5 表 B。

---

## Step 5 · 指标回收与回填清单

### A. 训练过程指标（→ `grpo_metrics.json` / `logs/grpo_train_*.log`）

| 编号 | 指标 | 从哪读 |
|---|---|---|
| A0 | merge parity（argmax 一致 / max\|Δlogit\|） | `merge_manifest.json` |
| A1 | reward 均值走势（训练侧） | wandb 或日志逐 step |
| A2 | 门禁短路占比（−2.5 收场的轨迹比例） | 日志 reward 分布 |
| A3 | eval 基线集 reward 均值（baseline → 各次 eval） | `[EVAL]` / `[FUSE]` 行 |
| A4 | 组内 std 是否非零（有无整组清零） | 日志 |
| A5 | 实际步数 / 耗时 / 显存峰值 | `grpo_metrics.json` + `nvidia-smi` |

### B. 探针指标（→ `grpo_probe_kstats.json`）

| 编号 | 指标 | 用途 |
|---|---|---|
| B1 | k 直方图（k=0~4 各多少条） | 验证分流设计；训练池规模 |
| B2 | 训练池难点/极性分布 | 难例覆盖是否偏科 |
| B3 | CoT 段 token 分位（p50/p95/max） | **顺带回填 TODO 5.20**——拿真分位重定 R_length 三档 |
| B4 | completion 整段长度分布 | 验证 max_completion=512 够不够 |

### C. 待回填文档映射

| 实测项 | 回填到 |
|---|---|
| A0~A5 | `notes/舆情负面精判_GRPO训练实战手册.md` §1.5 汇报口径（解除「从未实跑」红线）、模块 3 闸② 状态 `[待实现]`→`[已落盘·已实测]` |
| A3 | 同上 §1.2 超参表 `num_train_epochs` 行（补 eval 观察口径） |
| B1/B2 | `notes/舆情负面精判_GRPO数据准备实战手册.md` §四 分流记录 |
| B3 | TODO 5.20 → `reward_fns.py` 常量 + `test_reward_fns.py` + 数据准备手册 §五 + GRPO 手册 §2.5 |
| B4 | GRPO 手册 §1.2 `max_completion_length` 行 |

### D. 防串号纪律（回填时自查）

- **1066 的数字一个不进**（37%→95%、G=7、8192）；奖励区间写死 `[−2.5, +2.0]` 不许出现中间态比例；
- reward 均值必须带「基线集 8 条」分母口径；正式换 k=4 封存集后**两批数字分开记，不合并**；
- k 分布是「合并 SFT 模型」口径，与训完 GRPO 后的第二次 rollout 分开记。

---

## Step 6 · 收口条件与遗留

**本清单勾完即收口**：

- [ ] GRPO 手册文首「训练与 rollout 从未实跑」红线解除；
- [ ] 评估手册模块一④ GRPO 两项指标（奖励分布 / CoT 长度分布）从 `[已落盘]` 升级为 `[已实测]`；
- [ ] TODO 5.20 拿到 B3 分位后实施（改 `reward_fns` 常量 + 单测 + 两本手册）。

**留给正式 3,000 条阶段（不在本清单）**：

- eval 基线集换成真 k=4 封存集；`--fuse_hard` 才启用硬终止；
- `eval_four_tier.py` T4 档跑四档对照归因；
- 双卡拓扑 + vLLM 采样（`run_grpo_train.sh` 已支持，GPU 数 ≥2 自动开）。
