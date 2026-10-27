# 实验待办（舆情负面精判 · SFT 探针）

> **定位**：需要**跑实验才能决定**的事。确定要落地的改造见 `../OPTIMIZATION_BACKLOG.md`；
> 已评估但暂不实施的方案见 `2026-10-24_POC1_...md` 附录 B。
> **纪律**：每条写清「要验证什么」与「什么条件下才做」，不做无触发条件的排期。
> **最近更新**：2026-10-24

---

## 基线

当前基准：`run_20261024_104842`（POC1 + token 解冻 + 不预置），格式门三项 100%。

三根独立的轴，做消融时**只动一根**：

| 轴 | 开关 | 当前值 |
|---|---|---|
| loss 结构 | `SFT_POC_STAGE` | `POC1`（仅前缀掩码） |
| 协议 token | `UNFREEZE_PROTOCOL_TOKENS` | `1`（解冻开；`0` 关闭） |
| 前缀预置 | `THINK_PREFIX` | `0`（关） |

---

## E1 · POC2 消融（骨架掩码）

- **要验证**：JSON 结构骨架（键名 / 引号 / 括号 / 逗号）置 0 权重，是否提升判别字段的学习效率。
- **对比对象**：POC1（当前基准）。
- **状态**：早先跑过（`run_20261024_004149`），**结果作废**——那轮 token 未解冻，格式全崩、F1 全 0。
- **命令**：
  ```bash
  NPROC=1 SFT_POC_STAGE=POC2 EPOCHS=6 THINK_PREFIX=0 \
  ./scripts/run_sft_e2e.sh traintest
  ```

## E2 · POC3 消融（字段级 FIELD_W）

- **要验证**：字段级加权（`hard_case 4.0 / sentiment 3.0 / relevance 3.0 / risk 2.0 / evidence 0.4`）
  是否提升 hard_case Macro-F1，同时 evidence 子串命中率不掉。
- **对比对象**：POC2。
- **状态**：早先跑过（`run_20261024_005424`），**结果作废**（同上）。
- **命令**：
  ```bash
  NPROC=1 SFT_POC_STAGE=POC3 EPOCHS=6 THINK_PREFIX=0 \
  ./scripts/run_sft_e2e.sh traintest
  ```

## E3 · 裸基座对照

- **要验证**：SFT 净增益（当前只有绝对指标，没有「不训练能拿多少」的参照）。
- **做法**：`sft_infer.py --adapter_path ''` 跑裸基座，同一份验证集、同一解码配置。
- **状态**：未跑。

## E4 · POC4（边界 token 提权）

- **要验证**：把 `</think>` / `<|im_end|>` 的 loss 权重从 `default=1.0` 提到 **6.0**，
  是否让闭合更稳。
- **实现**：`FIELD_WEIGHTS["boundary"] = 6.0`，新增 `SFT_POC_STAGE=POC4` 档。
- **触发条件（任一）**：
  1. 更大数据上 `p(</think>)` 掉回 < 0.5；
  2. greedy 输出出现「有 JSON 但无闭合」。
- **现状**：`p(</think>) = 0.829`、`greedy_ok = 3/3`，**条件不满足，暂不做**。
- **背景**：`</think>` 仅 8 个 token，CoT 有 1622 个，梯度被稀释到 7%。
  但当前瓶颈是「输出行冻结」（已由 token 解冻解决），不是权重不够大。

## E5 · 协议标签改名（`<|begin_of_thought|>` / `<|end_of_thought|>`）

- **要评估的**：是否换成自带 special token 的协议，换取跨基座可移植性。
- **触发条件（任一）**：
  1. 换基座模型（尤其目标模型不带 `<think>`）；
  2. 需要与 Qwen 原生 thinking 模式共存；
  3. `p(</think>)` 在更大数据上仍卡在 ~0.8 上不去。
- **已测代价**：
  - adapter **132 MB → 1607 MB**（`resize_token_embeddings` 后 peft 把整张 embedding 表塞进 adapter）；
  - 新 token 行初始范数 0.371，仍是同一个共线簇里的随机行，不天然更可分；
  - tokenizer 必须随 adapter 分发，否则下游 id 全错位；
  - 代码改动面 **19 个文件 / 82 处** `think>` 字面量。
- **若要做**：不要直接 `resize_token_embeddings` + `save_pretrained`，
  改用 `save_embedding_layers=False` 后手动只存新增两行。
- **现状**：**明确暂不实施**，需重新评估后才进入落地清单。

## E6 · canary 频次与成本的权衡

- **要验证**：canary 每 20 / 30 / 60 步，对训练耗时与「发现问题的及时性」的影响。
- **现状**：默认已从 20 调到 **30**（60 步下测 2 次）。
- **触发条件**：需要比较不同 lr 的收敛速度时（要拿到更密的曲线）。
