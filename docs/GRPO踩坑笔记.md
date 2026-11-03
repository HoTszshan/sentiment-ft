# GRPO 踩坑笔记（G 系列）

> 定位：G 系列（GRPO 对照实验）实跑中踩到的坑，每条写「现象 → 根因 → 修法 → 预防」。
> 详版登记在 `draft/GRPO对照实验_实验结果记录_20261031.md`；本笔记是压缩版速查。
> 纪律：每条坑必须有 run 目录 / 日志行号级证据，不收推测。

---

## 坑 1 · 奖励坍塌（reward collapse）：β_KL=0 时模型「静默死亡」

**配置**：G2 消融，β_KL 0.04 → 0.0，其余同 G1''（1877 池 / lr 2e-5 / LoRA r16 / 1ep）。
**run**：`run_20261103_103528_G2p2_ep1_kl0`（2ep 首版 `run_20261103_074735` 35 步同型崩溃）。

### 现象（三层假象叠加，全部「看起来正常」）

| 表面信号 | 实测 | 真相 |
|---|---|---|
| train reward | 全程 0.01~0.65 正常波动 | 与模型好坏已无关，波动纯来自 batch 难度 |
| train loss | 「收敛」到 1.6e-10 | NaN 更新对 loss 无贡献，收敛是假象 |
| grad_norm | 全程 0.1~0.2（被 max_grad_norm=1.0 截住） | 截的是单步步幅，不是累积位移 |

### 时间线（日志逐步实测）

```
step 1~15   健康     grad_norm 0.1~0.2, KL ~0.0005（与 β=0.04 枪同形态）
step ~16    漂移启动  KL 0.002 → 0.3 → 1.45（步间跳变！唯一有效前兆信号）
step 22~88  指数发散  KL 33 → 3.6e3 → 2.4e8 → 2.8e34（grad_norm 始终 ≤1.3）
step 89     NaN 坍塌  grad_norm: nan, KL: inf（权重越过 fp32 上限 3.4e38 溢出）
step 90~234 死寂     全部 nan/0.0——145 步优化器空转，白烧 ~2h
```

### 根因（机制链）

**β=0 → 没有 KL 往回拉 → 88 步「每步合规」的小梯度方向一致复利 → 权重指数膨胀
（KL 0.0005 → 2.8×10³⁴）→ step 89 越过 fp32 上限溢出为 NaN → NaN 传染全计算图 →
除法出现 0 分母 → KL=inf（step 89）→ 全 NaN（step 90 起）。**

- **关键区分：梯度截断管单步，KL 阻尼管累积**——max_grad_norm 拦不住无阻尼的复利；
- inf 与 nan 的先后：KL 先算，NaN 污染的当前策略部分 token 概率为 0 → log(0/π_ref) = inf；
  下一步 NaN 扩散到 KL 全式才变 nan。同一死因的两个时刻。

### 这算奖励坍塌吗？——算，且和 reward hacking 是两种病

| | Reward Hacking（学坏了） | **奖励坍塌（本次）** |
|---|---|---|
| 模型状态 | 活着，主动钻 reward 空子 | **分布本身失去存在意义**（NaN 权重不再是「一个策略」） |
| reward 表现 | 虚高 + eval 崩（分数与模型还有互动——它在骗分） | **完全正常波动**（分数与模型好坏**解耦**） |
| 比方 | 学生发现监考松，抄小骗分——人还在考试 | 学生猝死，监考照常发卷打分——**分数还在出，考试已结束** |
| 报警信号 | reward 与 eval 背离 | reward/loss 全瞎，**唯一信号是 KL 步间跳变** |

**操作性定义**：无阻尼 KL 发散 → 权重/概率比溢出 NaN → reward/loss 与策略质量解耦。
前兆 = KL 步间跳变（0.002→1.45），不是 reward 曲线。

### 修法与预防

1. **监控 KL 跳变**：步间跳变（如 0.002→1.45）或绝对值连续 N 步 >10 → 立即停车。
   本次判据可在 step ~22 停车 vs 实际白烧 145 步；
2. **NaN 早停**：grad_norm 连续 ≥3 步 nan/inf 即停（step 89 可停）；
3. **β_KL=0.04 保留**（G2 关账结论：撤 KL 的代价是 88 步内从健康走到数学死亡）。

---

## 坑 2 · eval 熔断盲区：PEFT+vLLM 权重同步只送了裸基座

**run**：G1（0.5563 恒定）/ G1''（0.3200/0.3469 恒定）/ G2（0.3469 恒定）——
**所有 run 内 eval 分数从第 0 步到最后一步一字不变**（run 间不同只因 eval 集不同）。

### 现象

熔断器（相对跌幅 >2%）全程 0 触发——不是该停没停，是它**看到的世界里一切正常**。
此前记录的「greedy eval 零噪声」实为「恒定裸基座」。

### 根因（实验证实，非推测）

- PEFT 模型的 `state_dict()`：base_layer = 冻结基座**原值**（`torch.equal` 验证 True），
  LoRA 增量在独立 `lora_A/B` 键（903 键中 504 个）；
- vLLM `load_weights()` **按名字逐键**加载：base 名收到冻结原值，`lora_A/B` 在 vLLM
  模型里无对应参数 → **被静默忽略**；
- 即「state_dict → load_weights」链路（trl 0.14 原版 + eval callback 照抄）在 PEFT 上
  **同步的永远是裸基座，LoRA 更新从未到达 vLLM**。

### 修法（✅ 已落码 `lib/grpo_eval_fuse.py`）

`_eval_reward_mean_vllm` 改为 **merge → 同步 → unmerge** 三步：

```python
peft_model.merge_adapter()          # LoRA 增量合并进 base_layer
state_dict = unwrapped.state_dict() # 此时 base_layer 才真正带上增量
llm_model.load_weights(state_dict.items())
peft_model.unmerge_adapter()        # finally 里还原，防梯度污染
```

非 PEFT 模型走 AttributeError 兜底；验证开关：下次训练的
`eval/reward_mean_baseline` 若与历史值（0.3469/0.5563）不同 = 修复生效。
**单测 62/62 绿。**

### 遗留待验证

训练 rollout 走 trl 内部链路（`_last_loaded_step` 缓存），与 callback 非同一条代码路径——
rollout 是否同样盲区**待专项核实**（决定历史各枪 train reward 归因是否打折）。

### 预防

**「监控指标必须量到真实策略」**——任何 eval 前先证明权重同步真的发生（可打一条
logit 抽查：同步前后 vLLM 对同一 prompt 的 top-1 logit 必须变化）。

---

## 坑 3 · merge parity 拦截 NaN adapter（防线正确工作，非坑）

**现象**：G2 adapter merge 时 `max|Δlogit|=nan` → parity FAIL 拒绝落盘。
**核实**：G2 adapter `lora_B.absmax = nan`（健康 G1'' 仅 0.0009~0.0022）。
**结论**：不是 merge 工具的 bug，是**最后一道防线拦住了崩坏权重进生产**——
不要绕过校验强 merge。adapter 目录留证（坍塌标本），不删不改。

---

## 速查：GRPO 训练健康度三看

| 看 | 信号 | 动作 |
|---|---|---|
| **KL 跳变**（第一优先） | 0.002 → 1.45 这种量级跳变，或连续 >10 | 立即停车 |
| grad_norm NaN/inf | 连续 ≥3 步 | 立即停车（已死，别烧卡时） |
| reward 曲线 | 震荡是常态（batch 难度主导），**不作为有效性/健康度判据** | — |

> 记忆钩子：**「判训练死活，别看 reward，看 KL。」**

---

## 附：G2 成果总结与「必须要 β 吗」（2026-11-03 补，面试问答版）

### G2 整理出了什么（三层成果）

1. **主结论**：β_KL 必须保留——β=0 两轮同型崩溃（2ep 版 35 步、1ep 版 89 步），
   88 步内 KL 0.0005 → 2.8×10³⁴ → NaN，不是偶发是必然；
2. **机制结论**：奖励坍塌 ≠ reward hacking（对照表见坑 1）——坍塌的 reward/loss 全程
   「正常」，唯一报警信号是 KL 步间跳变；金句「梯度截断管单步，KL 阻尼管累积」；
3. **工程结论**：排查「熔断为什么不停」时发现 PEFT+vLLM 权重同步盲区（坑 2）——
   eval 从第一天起量的就是裸基座。已修复落码。**没有 G2 的崩溃，这个 bug 还会继续藏着。**

### 必须要 β 吗？——必须有「往回拉的力」，但 β 不是唯一形式

**β 的作用 = 每走一步都交「离家的税」**，走太远税越重，逼你回来。撤税（β=0）=
模型可以一路狂奔不回头，88 步跑到数学上不存在的地步。

| 防狂奔方案 | 原理 | 判定 |
|---|---|---|
| β_KL > 0（本次方案） | 显式离家税 | ✅ 已实证必要，0.04 保留 |
| clip（PPO 式） | 限制单步概率比 | trl 0.14 无此参数（无 clip 变体）；升级 trl ≥0.16 可加，代码已留口子（`epsilon=h.epsilon`） |
| 小 lr + 短训 | 物理上走不远 | 辅助手段，不可单独依赖——G2 的 lr=2e-5 已经很小，照样崩 |
| grad_norm 截断 | ❌ 已实证无效 | 管单步步幅，管不了方向一致的小步复利 |

**准确表述**：不是「必须要 β 这个参数」，而是「**必须有 KL 承担累积约束**」——在
trl 0.14 无 clip 变体里这个角色只能由 β_KL 扮演，所以 0.04 不是可选项是必需品；
若有 clip（升级 trl），clip 管单步、KL 管累积，两者可分担。

### 面试一段话版

> 「在 trl 0.14 无 clip 的 GRPO 变体里，β_KL 是唯一的策略约束，撤掉即 88 步内分布坍塌
> （KL → 2.8e34 → NaN）——这是我们的实验实证，所以 0.04 保留。如果升级到有 clip 的版本，
> clip 和 KL 可以分担约束（clip 管单步、KL 管累积），但在当前版本里没有替代品。」

---

## 附：G 系列收官决策（2026-11-03，剩余实验取舍）

> 决策：G3 砍掉、G4 不跑、G5b 可选、**G5a 唯一推荐**。理由如下，防日后翻旧账。

### 砍 G3（lr 5e-5）的理由——不是不重要，是答案被 G2 抢答了

1. G3 预期收益是「框定 lr 安全区」；但 G2 用 **lr=2e-5（与 G1'' 相同）照样 88 步坍塌**——
   已实证「坍塌开关在 KL 不在 lr」，「小 lr = 安全带」这个前提本身被看穿了；
2. lr 上界在哪对叙事贡献极小：G1'' 主数字已出，简历行不需要 lr 敏感性分析支撑；
   5e-5 大概率又是一次可预测的崩溃/短路——预测性实验 ROI 低；
3. 省 2.8h 给 G5a 更值。

### G4（G=8）/ G5b（除 std）：不跑 / 可选

- G4：触发线（k∈1..3<30%）字面触发过，但 G1'' 主数字已出，组规模消融不改变任何结论；
- G5b：有正面（G1~G1'' 全用 scale_rewards=none 成功）没反面；面试可答「留作未来工作」。

### G5a（关 R_length）：唯一推荐，2.8h 买一句面试硬话

**问题**：简历「GRPO 把 CoT 压短 19%（273→220/186 token）」——面试官必问
「真是 GRPO 的功劳吗？会不会 SFT 本来就会越来越短？」

**G5a 就是这句话的证据**：关掉 R_length 重训，CoT 仍短 → SFT 遗产（简历话要改）；
CoT 回弹变长 → R_length 功劳实锤（简历话焊死）。

**前置改动**：`lib/reward_fns.py` 加 r_length 临时置 0 开关 + `train_grpo.py` 加
`--disable_r_length` CLI 参数（+单测）。
**判据（事先写死）**：G5a 终评 CoT token p50 显著回弹（>250）且判别力不掉 → 归因成立。

### 收官后的面试弹药清单（全部有 run 目录级证据）

| 弹药 | 出处 |
|---|---|
| Base → SFT → SFT+GRPO 三行实测数字 | 数字清单 §6（全 ✅） |
| KL 必要性实证（撤 KL → 88 步坍塌） | G2 两个 run |
| 盲区补齐因果链（池分布 → k 迁移 → 考卷转正） | G1'/G1''/G6 |
| CoT 压缩 + 安全性（无空壳化） | G1'' + 11-03 核查（G5a 跑完再加归因） |
| 重跑方差 ±1.2pp（统计诚实） | 11-02 vs 11-03 |
| eval 同步盲区修复（监控必须量到真实策略） | 坑 2 + merge parity 拦截 |

> G5a 跑完 → GRPO 实验线彻底收口，无欠账。

---

## 附：G5a/G5c 落码与启动（2026-11-03，收官双枪）

### 落码（lib/reward_fns.py，环境变量开关，默认全关零影响）

- **G5a**：`GRPO_DISABLE_R_LENGTH=1` → `r_length` 恒 0（compute_reward 内短路，维度不存在）；
- **G5c**：`GRPO_RISK_PARTIAL=1` → r_accuracy risk 未中区插部分档
  `0.5 + 0.2×Jaccard(g_rk, p_rk)`（交集>0 才生效；Jaccard=0 回落 0.5）；
- 单调链实测验证：`0.5 < 部分档(实测 0.6 @J=1/3) < 0.7 ≤ 1.0` ✅；
- 全库单测 352/352 绿（含 45 条 reward 测试）；默认关闭时行为与旧版逐字节一致。

### 启动命令（两枪 1ep / 1877 池 / 其余与 G1'' 逐字一致）

```bash
# G5c（先跑：risk 部分匹配档，~2.8h）
export SYNC_WANDB=1 WANDB_PROJECT=sentiment-grpo WANDB_MODE=offline
export DATA=/root/autodl-tmp/datasets/rl_round2/grpo_train_prompts_final.jsonl
export MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-R7_ep4-bf16
export EVAL_FILE=/root/autodl-tmp/datasets/rl_round2/grpo_eval_prompts.jsonl
export EVAL_STEPS=40 NPROC=2 GRPO_RISK_PARTIAL=1
export GRPO_ARGS="--num_train_epochs 1 --save_steps 40 --gradient_checkpointing"
nohup ./scripts/run_grpo_train.sh --tag=G5c_riskjac > logs/g5c_train.log 2>&1 &

# G5a（后跑：关 R_length，~2.8h；等 G5c 完再起，防抢卡）
export GRPO_DISABLE_R_LENGTH=1   # 其余同上，去掉 GRPO_RISK_PARTIAL
nohup ./scripts/run_grpo_train.sh --tag=G5a_nolen > logs/g5a_train.log 2>&1 &
```

### 判据（事先写死，防事后挑数字）

| 枪 | 预期信号 | 判据 |
|---|---|---|
| G5c | risk SubAcc ↑（清单预期 +1~3pp） | >77.60（G1'' 11-02 版）即正收益；同时盯 negR/FAR/CoT p50 无回退 |
| G5a | CoT p50 回弹（无压短力） | p50 显著 >230（G1'' 同口径 230）→ 压短归因成立；若仍 ≤230 → CoT 压缩主要是 SFT 遗产，简历口径要改 |
| 顺带 | eval 同步修复验证 | pre-train baseline ≠ 0.3469 = 坑 2 修复生效；若仍 0.3469 说明 rollout 链路同样盲区，需回炉 |

### 顺序纪律

先 G5c 后 G5a——G5c 在默认 R_length 下拿干净 risk 读数；G5a 放后做归因，
防「长 CoT 扰动 risk」混杂变量污染 G5c 判读。
