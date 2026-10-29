# GRPO 探针实验 · 快速启动 TODO 清单

> **定位**：AutoDL 单卡（RTX 4090/3090 24GB）上把 GRPO 探针（merge → 训练 → rollout 分流）从零跑通并回填实测数字的**一步步操作清单**。前置：SFT 探针已训完（有 adapter，见 `TODO_SFT探针快速启动.md`）。
> **口径权威源**：`notes/舆情负面精判_GRPO训练实战手册.md`（训练/Reward）、`notes/舆情负面精判_GRPO数据准备实战手册.md` §五（四维 Reward 契约）、`README.md` §3（脚本用法）。冲突以上游为准。
> **纪律**：跑出来的数才回填；没跑的一律标 `[待实测]`，不编数。本清单勾完即把 GRPO 手册的「训练与 rollout 从未实跑」红线解除。
>
> **环境**：与 SFT 共用同一个虚拟环境，`pip install -r requirements.txt` 一步到位（trl 0.14 / torch 2.5.1 组合，2026-10-27 真机验证）。wandb 开关同 SFT（`SYNC_WANDB=1` 才上传，项目名 `sentiment-grpo-probe`）。
>
> **路径约定（AutoDL，与 SFT 清单对齐）**：模型下到 `/root/autodl-tmp/models`；数据与训练/探针产物都在仓库内 `/root/autodl-tmp/senti/data/`（探针数据为仓库内置文件 `data/probe_test_grpo_train_synth.jsonl`，50 条，无需上传）。
> - 下方命令一律用 AutoDL 绝对路径；本机跑则换成仓库相对路径（`models/`、`data/`）；
> - `train_grpo.py` 用 `--model_name_or_path`；`run_grpo_probe.py` 用 `--base_model`（分体挂载另加 `--adapter`）。`--data` / `--out` 两脚本一致，全是显式参数，不依赖仓库布局。

---

## Step 1 · SFT adapter 合并（GRPO 的起点）

> GRPO 只认「一个完整模型目录」，不认 adapter——先合并（原理见 GRPO 手册 §1.1：TRL 算 KL 走 `disable_adapter()`，双 adapter 叠加会让 reference 退化成裸基座）。

- [ ] **1.1** 确认 SFT adapter 在位：
  ```bash
  ls /root/autodl-tmp/models/sft_senti/run_*/sft_lora/adapter_config.json   # 取最新时间戳那个
  ls /root/autodl-tmp/models/Qwen3-4B-Instruct-2507/config.json # 基座在位
  ```
  > ⚠️ 历史产物适配：改名前训的 run 目录里目录名是 `probe_lora`（如 `run_20261024_180116_POC3/probe_lora`），路径按实际在位的为准；改名后新训的才是 `sft_lora`。
- [ ] **1.2** 合并体检（不加载模型，任何环境可跑）：
  ```bash
  python3 scripts/merge_sft_adapter.py --dry_run \
      --base /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --adapter /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
      --out /root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test
  ```
  预期：路径体检通过；若打 `[WARN] adapter 训练时记录的基座是…` 说明 `--base` 与训练时不一致，**先核对再合并**。
- [ ] **1.3** 真合并（GPU 上跑，含 logits 一致性校验）。**必须 `--dtype fp32`**：实测（2026-10-26）bf16 合并会把「先融合权重」的舍入误差做进权重（max|Δlogit|=1.375、top1Δp≈0.032），fp32 下同一 adapter 误差 4 个数量级更小（max|Δlogit|=1.2e-4、top1Δp≈4e-6）：
  ```bash
  python3 scripts/merge_sft_adapter.py --dtype fp32 \
      --base /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --adapter /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
      --out /root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test
  ```
  预期最后一屏：`合并完成` + `argmax 相同=True` + `top1Δp` 极小 + `merge_manifest.json` 落盘。
  **若 parity FAIL（bf16 时代旧判据）**：先换 `--dtype fp32` 重跑再下结论——FAIL 未必是基座错配，更常见是 bf16 融合精度损失；fp32 仍 FAIL 才去查 adapter 基座，不要 `--no_verify` 绕过。
- [ ] **1.4** 验证据链：`cat /root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test/merge_manifest.json`
  → 记录 `parity.max_abs_logit_diff` 与 `argmax_equal`（回填 Step 5 表 A0）。

> **首跑留意**：merge 路径已在真 GPU 上首验通过（2026-10-26，fp32 口径）。失败先看 peft 版本（≥0.12）。

---

## Step 2 · 训练前体检（不碰 GPU 也要全过）

- [ ] **2.1** 数据在位确认：
  ```bash
  wc -l /root/autodl-tmp/senti/data/probe_test_grpo_train_synth.jsonl    # 当前文件 50 条
  ```
- [ ] **2.2** 干跑体检（数据契约 + 预算 + 优势口径自检，零 GPU）：
  ```bash
  python3 scripts/train_grpo.py \
      --data /root/autodl-tmp/senti/data/probe_test_grpo_train_synth.jsonl --dry_run --no_wandb
  ```
  逐项核对：
  - 数据行数为 **50 条**；每步 8 题 × G=4 = 32 轨迹/步；50 题按默认 1 epoch 预算为 **7 步**；
  - `input_text 为空: 0 条`；
  - `优势口径自检通过`；
  - `[DRY RUN] 未加载模型/TRL`。
- [x] **2.3** eval 基线集（只用于熔断功能链路冒烟；与训练集同源，**不代表独立泛化集或真实“对齐税”评估**）。
  **无需任何手工步骤**：`train_grpo.py` 不传 `--eval_file` 时自动按 `--eval_n`（默认 8）从训练池按 (极性, 难点) 去重抽取，不足不补造（逻辑在 `lib/grpo_dataset.build_eval_baseline`）。


---

## Step 3 · GRPO 探针训练（先冒烟，再长跑验证熔断）

- [x] **3.1** 冒烟（前台，3 步，只验证「能采样、能打分、能更新」）——✅ 2026-10-28 通过（run_20261028_115727_smoke，3 步 262s，reward -0.35→+0.17→-0.17，LoRA 已落盘）：
  ```bash
  # 推荐走 wrapper（拓扑/显存配置自动）；或手动命令加 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --gradient_checkpointing
  MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test \
  GRPO_ARGS="--limit 8 --max_steps 3" ./scripts/run_grpo_train.sh --tag=smoke
  ```
  验收：训练能启动、reward 在合法值域、模型/checkpoint 可保存。**此步不验证 eval 熔断**，因为只有 3 步且基线与训练池同源。
- [ ] **3.2** 长跑探针（专门验证周期 eval 与熔断链路；50 题 × 4 epochs = 28 步）：
  ```bash
  # 走 wrapper（拓扑/显存配置自动）；双卡自动开 vLLM，单卡自动降级 HF generate
  MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test \
  GRPO_ARGS="--eval_steps 2 --fuse_threshold 0.02 --num_train_epochs 4 --save_steps 2" \
  ./scripts/run_grpo_train.sh --tag=fuse_smoke
  ```
  ⚠️ 不要用裸 `python3 scripts/train_grpo.py` 直启：双卡机不加 `CUDA_VISIBLE_DEVICES=0` 会触发 DataParallel OOM，且缺 `--gradient_checkpointing`（已固化默认值，走 wrapper 自动带上）。手动调试实在要前台跑，须带：`CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ... --gradient_checkpointing`。
  验收：出现训练前 `[EVAL] pre-train baseline reward 均值`；之后每 2 步出现 `[EVAL]` 或 `[FUSE]`（由 callback 的 `on_step_end` 周期触发）；正常下降只告警，只有加 `--fuse_hard` 才设置训练停止。若想验证硬刹车，可另跑一次 `GRPO_ARGS` 里加 `--fuse_hard --fuse_threshold 0`（零阈值仅用于功能测试，非推荐训练阈值）。
  > 这 8 条从同一个 50 条探针池抽出，会和训练集重叠。它只用于验证“评估→比较→告警/停止”功能，不是泛化评估；报告中要注明此限制。更多 epochs 是为了让功能有机会被触发，不代表正式训练轮数建议。

  > **熔断阈值的两个已知坑（探针阶段把 `[FUSE]` 当功能信号，别当真实退化）**：
  > 1. **`--fuse_threshold 0.02` 是相对跌幅**，仅当 baseline 为正时生效；**baseline ≤ 0 时 `should_fuse` 退化为绝对差判定**（跌超 0.02 分即触发；恰等于门槛因 1e-9 容差不触发）。在 T=0.7 单次采样下，eval reward 均值的噪声量级（粗估 >0.1）远大于 0.02，这个绝对差分支会频繁误报——探针阶段看到 `[FUSE]` 先核实是「真跌」还是「采样噪声跨过 0.02 门槛」。
  > 2. **采样噪声的根因**：`grpo_eval_fuse._eval_reward_mean` 用 `do_sample=True + temperature=0.7` 逐条采样打分，同一样本每次得分都不同。要压低噪声，正式阶段应改 greedy（T=0）+ 更大基线集，否则 2% 相对阈值也会被噪声淹没。
- [ ] **3.3** 正式探针训练（按当前研究设计的 1 epoch，不带熔断）：
  ```bash
  DATA=/root/autodl-tmp/senti/data/probe_test_grpo_train_synth.jsonl \
  MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test \
  ./scripts/run_grpo_train.sh --tag=fuse_smoke
  # (OUTPUT_DIR 默认就是 /root/autodl-tmp/models/grpo_senti，想换地方才传)
  ```
  产物布局（对齐 SFT e2e）：一个 run 目录收全部，拷走即归档——
  `grpo_senti/run_<时间戳>[_<tag>]/`，里面 `grpo_lora/`（adapter + 指标）+ `logs/`（训练日志）。
  - `--tag=fuse_smoke` 位置任意，不传就不加后缀
  - GPU 卡数：默认 `nvidia-smi` 自动识别
  - ~~单卡 HF generate~~ **双卡 vLLM 已跑通（2026-10-28，run_20261028_115727_smoke）**：
    `NPROC=2` 自动走「训练 cuda:0 + vLLM 独占 cuda:1」，8 题冒烟 3 步约 4 分钟。
    显存配置已固化默认值：batch 2×4、`--gradient_checkpointing`、vLLM utilization 0.9（KV 12.3G / 并发 ~35x）、
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（脚本自动 export）。
    **推荐命令**（显存三件套已入默认值，可不传）：
    ```bash
    MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test \
    GRPO_ARGS="--limit 8 --max_steps 3" ./scripts/run_grpo_train.sh --tag=smoke
    ```
    单卡环境仍自动降级 `CUDA_VISIBLE_DEVICES=0` + HF generate（DataParallel 防呆保留）
- [ ] **3.4** 熔断默认开：训练入口直跑不传 `--eval_file` 即自动从训练池抽 `--eval_n`（默认 8）条基线集；要显式指定就用 `--eval_file`，`--eval_n 0` 关闭。启动脚本（`run_grpo_train.sh`）则用环境变量：设置 `EVAL_FILE`（或不设走同样自动档）、`FUSE_HARD=1` 才执行硬停止。

---

## Step 4 · rollout 与 k 分流（筛出真正的训练池）

- [ ] **4.1** 真实 rollout（SFT 合并模型作起点，同题 G=4）：
  ```bash
  BASE_MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test \
  ./scripts/run_grpo_probe.sh --tag=sft_kdist
  # (OUTPUT_DIR 默认 /root/autodl-tmp/models/grpo_senti，想换地方才传)
  ```
  产物自动收进 `run_<时间戳>[_<tag>]/`：k 直方图 + 逐题明细 + 日志。
  - 单卡即可（50 题几分钟）；无 GPU 先 `DRY=1` 验链路
  - 逐题明细 `.rows.jsonl` 带每个标签的对错 + reward 分量，调 `reward_fns.py` 直接看它
  - GRPO 训完再跑一次（换个 `--tag`），对比 k 分布迁移
- [ ] **4.2** 读 k 直方图（脚本自动打印）：
  - **健康信号**：k=1~3 占比显著非零（有梯度可学）；k=0 与 k=4 两头都有（说明筛选有意义）。
  - **坏信号与处置**：全 k=4 → SFT 已收敛到位，GRPO 无题可训（当前探针 50 条上属可能结果，正式池要靠更难的题）；全 k=0 → 打分或数据坏了，回 Step 2 查。
- [ ] **4.3** 产物留档（把两个文件下载回本机 `tests/output/grpo/` 归档）：`grpo_probe_kstats.json`（k 直方图 + 训练池规模）+ `grpo_probe_kstats.rows.jsonl`（逐题明细）→ 回填 Step 5 表 B。

---

## Step 5 · 指标回收与回填清单

### A. 训练过程指标（→ `grpo_metrics.json` / `logs/grpo_train_*.log`）

| 编号 | 指标 | 从哪读 |
|---|---|---|
| A0 | merge parity（argmax 一致 / max\|Δlogit\|） | `merge_manifest.json` |
| A1 | reward 均值走势（训练侧） | wandb 或日志逐 step |
| A2 | 门禁短路占比（−2.5 收场的轨迹比例） | 日志 reward 分布 |
| A3 | eval 基线集 reward 均值（训练前 baseline → 各次 eval；8 条同源探针集仅链路冒烟） | `[EVAL] pre-train baseline` / `[EVAL]` / `[FUSE]` 行；注明与训练集重叠 |


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
- **eval 冒烟指标注明基线只有 8 条且与训练池同源**；该结果只证明熔断流程能跑，不能当作独立泛化/对齐税结论；
- k 分布是「合并 SFT 模型」口径，与训完 GRPO 后的第二次 rollout 分开记。


---

## Step 6 · 收口条件与遗留

**本清单勾完即收口**：

- [ ] GRPO 手册文首「训练与 rollout 从未实跑」红线解除；
- [ ] 评估手册模块一④ GRPO 两项指标（奖励分布 / CoT 长度分布）从 `[已落盘]` 升级为 `[已实测]`；
- [ ] TODO 5.20 拿到 B3 分位后实施（改 `reward_fns` 常量 + 单测 + 两本手册）。

**留给正式 3,000 条阶段（不在本清单）**：

- eval 基线集换成独立留出集（若目标是测通用能力退化，应选通用指令集）；阈值需结合基线集规模与波动重新校准；**同时复审 `should_fuse` 的绝对差分支**（baseline≤0 时 0.02 门槛在采样噪声下会误报，见 Step 3.2 的坑位说明）与 `do_sample=True` 的解码噪声，必要时改 greedy 解码；
- `eval_four_tier.py` T4 档跑四档对照归因；
- 双卡拓扑 + vLLM 采样 **已跑通（2026-10-28）**：`NPROC=2` 自动启用；显存三件套（batch 2×4 / gradient_checkpointing / vLLM 0.9）已固化为默认值，见 `notes/GRPO环境踩坑实录_20261027.md` 与 `train_grpo.py` 内置补丁注释。
