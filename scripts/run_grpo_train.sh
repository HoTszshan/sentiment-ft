#!/usr/bin/env bash
# ==============================================================================
# 舆情负面精判 · GRPO 训练启动脚本 (nohup 后台 + 时间戳日志 + PID 文件)
#
# 用法 (在 MVP/tests 目录):
#   ./scripts/run_grpo_train.sh                        # 自动按 GPU 数选拓扑
#   ./scripts/run_grpo_train.sh --tag=fuse_smoke       # 输出目录追加 tag 后缀
#   GRPO_ARGS="--limit 8 --max_steps 3" ./scripts/run_grpo_train.sh   # 冒烟
#   DRY=1 ./scripts/run_grpo_train.sh                  # 只跑 --dry_run 体检（无 GPU 可跑）
#
# 产物布局 (对齐 run_probe_e2e.sh: 一个 run 目录 = 一次完整实验, 拷走即归档):
#   $OUTPUT_DIR/run_<TS>[_<TAG>]/
#     ├── grpo_lora/                  adapter + grpo_metrics.json (训练产物)
#     └── logs/grpo_train_<TS>.log    训练日志 (train_grpo 双写 + 脚本 tee 同一份)
#
# 环境变量:
#   DATA           默认 data/grpo_prompts_probe.jsonl
#   MODEL          默认 models/Qwen3-4B-Instruct-SFT (先跑 merge_sft_adapter.py 产出)
#                  指向原厂基座 = 未挂 SFT 起训，只用于对照实验
#   OUTPUT_DIR     默认 /root/autodl-tmp/models/grpo_senti (AutoDL, 与 SFT 的 sft_senti 同盘同层);
#                  本机自动退回 <仓库>/output/grpo (run 目录的父目录)
#   NPROC          覆盖 GPU 卡数 (默认自动 nvidia-smi 识别; NPROC=1 单卡 / NPROC=2 双卡+vLLM)
#   EVAL_FILE      可选 reward 熔断基线集 jsonl；传入才启用 eval callback
#   EVAL_STEPS     eval 间隔（默认 2 步）
#   FUSE_THRESHOLD 相对首轮 eval reward 的跌幅阈值（默认 2%）
#   FUSE_HARD      =1 时熔断停止训练；默认只告警
#   WANDB_API_KEY  设了就同步 wandb
#   GRPO_ARGS      透传给 train_grpo.py 的额外参数 (--tag 已单独解析, 不用塞这里)
#
# 拓扑 (TRL 0.14+ colocate vLLM，参考 12S2W10_rlhf/GRPO/run_train_2gpu.sh):
#   1 卡  → 不启用 vLLM，rollout 走 HF generate（探针 50 题足够）
#   2 卡  → 训练进程数 = GPU 数 - 1 = 1，vLLM 独占 cuda:1（use_vllm=auto）
#   ⚠️ 不要在脚本里重排 CUDA_VISIBLE_DEVICES（torch 初始化后改它不生效）
# ==============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"
echo "工作目录: $BASE_DIR"

DATA="${DATA:-data/grpo_prompts_probe.jsonl}"
MODEL="${MODEL:-models/Qwen3-4B-Instruct-SFT}"
# 与 run_probe_e2e.sh 同款回退: 环境变量 > AutoDL 数据盘 > 仓库本地
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [ -n "$OUTPUT_DIR" ]; then :;
elif [ -d /root/autodl-tmp ]; then OUTPUT_DIR=/root/autodl-tmp/models/grpo_senti;
else OUTPUT_DIR="$BASE_DIR/output/grpo"; fi
EVAL_FILE="${EVAL_FILE:-}"
EVAL_STEPS="${EVAL_STEPS:-5}"
FUSE_THRESHOLD="${FUSE_THRESHOLD:-0.02}"
FUSE_HARD="${FUSE_HARD:-0}"
GRPO_ARGS="${GRPO_ARGS:-}"
DRY="${DRY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------- 0. 参数解析 (--tag=XXX, 其余透传 GRPO_ARGS)
STEP_TAG=""
GRPO_ARGS_DEFAULT="${GRPO_ARGS:-}"
GRPO_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --tag=*) STEP_TAG="${arg#--tag=}" ;;
    *) GRPO_ARGS="$GRPO_ARGS $arg" ;;
  esac
done
GRPO_ARGS="${GRPO_ARGS_DEFAULT}${GRPO_ARGS}"
GRPO_ARGS="${GRPO_ARGS# }"

# run 目录 + 总日志提前建（依赖自检起就落这里；DRY 模式也会留下含体检日志的 run 目录）
# 防呆: OUTPUT_DIR 指到仓库根 / HOME / / 时, run_<TS> 会直接铺在那一层
OUTPUT_DIR_REAL="$(readlink -f "$OUTPUT_DIR" 2>/dev/null || echo "$OUTPUT_DIR")"
case "$OUTPUT_DIR_REAL" in
  "$BASE_DIR"|"/"|"${HOME:-/nonexistent}") echo "❌ OUTPUT_DIR 不能是仓库根/HOME// ($OUTPUT_DIR)，run_<TS> 会直接铺在那里"; exit 2 ;;
esac
RUN_DIR="$OUTPUT_DIR/run_${TS}${STEP_TAG:+_${STEP_TAG}}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/grpo_train_${TS}.log"
log() { echo "[GRPO] $*" | tee -a "$RUN_LOG"; }
log "产物目录: $RUN_DIR"

# ---------------------------------------------------------------- 0. 依赖自检
log "依赖自检..."
missing=""
for m in torch transformers trl peft datasets; do
  python3 -c "import $m" 2>/dev/null || missing="$missing $m"
done
if [ -n "$missing" ]; then
  log "缺少依赖:$missing"
  log "安装: pip install 'torch>=2.2' 'transformers>=4.55' 'trl>=0.19' 'peft>=0.12' datasets accelerate"
  if [ "${SKIP_DEPS:-0}" != "1" ]; then exit 1; fi
else
  log "训练依赖齐全"
fi
python3 -c "import wandb" 2>/dev/null && log "wandb: 已安装" || log "wandb: 未安装，只落本地日志"

# ---------------------------------------------------------------- 1. 干跑体检
if [ "$DRY" = "1" ]; then
  log "DRY=1 → 只做数据/契约/预算体检"
  python3 scripts/train_grpo.py --data "$DATA" --dry_run --log_file "$RUN_LOG" $GRPO_ARGS
  exit 0
fi

# ---------------------------------------------------------------- 1.5 run 目录产物路径
# 产物布局: 训练 adapter / 指标 / 日志全收在 run_<TS>[_<TAG>]/ 一个目录, 拷走即归档
# (RUN_DIR/LOG_DIR/RUN_LOG 已在开头建好; 这里只补训练产物路径)
RUN_OUT="$RUN_DIR/grpo_lora"

# ---------------------------------------------------------------- 2. 拓扑选择
# GPU 卡数: 默认自动 nvidia-smi 识别; NPROC 环境变量可覆盖 (NPROC=1 单卡 / NPROC=2 双卡)
N_GPU="${NPROC:-}"
if [ -z "$N_GPU" ]; then
  N_GPU=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    N_GPU="$(nvidia-smi -L 2>/dev/null | awk '/^GPU/{count++} END{print count+0}')"
  fi
fi
N_REAL=0
if command -v nvidia-smi >/dev/null 2>&1; then
  N_REAL="$(nvidia-smi -L 2>/dev/null | awk '/^GPU/{count++} END{print count+0}')"
fi
if [ -n "${NPROC:-}" ] && [ "$NPROC" -gt "$N_REAL" ]; then
  log "❌ NPROC=$NPROC 但机器实际只有 $N_REAL 张卡，降回 $N_REAL 卡拓扑"
  N_GPU="$N_REAL"
fi
log "GPU 卡数: $N_GPU${NPROC:+ (NPROC 显式指定; 实机 $N_REAL 张)}"

if [ "$N_GPU" -ge 2 ]; then
  # 双卡: 1 训练进程 (cuda:0) + vLLM 独占 (cuda:1)
  # 关键: 训练进程必须只看得见 1 张卡 —— 多卡可见 + 非分布式启动时 HF Trainer
  # 会包 nn.DataParallel, TRL 1.16 GRPOTrainer 直接拒绝 (n_gpu>1 raise)。
  # 实测报错: run_20261027_110514_fuse_smoke。CUDA_VISIBLE_DEVICES=0 把训练进程
  # 限到 cuda:0, 剩下的卡留给 vLLM (colocate 模式自己挑可见卡之外的设备)。
  # trl 0.14 + vllm 在 3090 24G 上 5 次实测全部失败 (run_*vllm_smoke1~5, 2026-10-27):
  #   1) 限单卡可见 → vllm auto 推导 cuda:1 越界; 2/3) tokenizer API 被 transformers 5.x 删;
  #   4) 2 卡可见 → DataParallel 铺满两卡, vllm 无卡可起; 5) 单卡 colocate →
  #   数学不可行: 训练 11.6G + vllm 权重 7.5G + KV 2G + logits 峰值 4G+ > 24G。
  # 结论: 本机 GRPO 用单卡 HF generate (NPROC=1), vllm 需要 ≥40G 卡或 trl≥1.x 升级线。
  LAUNCH=(env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          accelerate launch --num_processes 1 --gpu_ids 0)
  VLLM_FLAG=""
  log "拓扑: 单卡 HF generate (trl 0.14 + vllm 在 24G 卡上实测不可行, 见脚本头 vllm 段)"
elif [ "$N_GPU" = "1" ]; then
  # "单卡拓扑"在多卡机器上是「只用 1 张卡」—— 也必须限可见卡, 否则 HF Trainer
  # 包 nn.DataParallel (NPROC=1 只是 1 个进程, 该进程仍看得见全部卡)。实测:
  # run_20261027_121xxx postfix, GRPOTrainer raise n_gpu>1。
  LAUNCH=(env CUDA_VISIBLE_DEVICES=0 python3)
  VLLM_FLAG=""
  log "拓扑: 单卡 (CUDA_VISIBLE_DEVICES=0)，HF generate 采样 (1k 探针/正式训练建议 NPROC=2 双卡开 vLLM)"
else
  log "未检测到 GPU，仍继续 (train_grpo 会自己报错)"
  LAUNCH=(python3)
  VLLM_FLAG=""
fi

# wandb 参数
WANDB_ARGS=""
if [ -z "${WANDB_API_KEY:-}" ] || [ "${WANDB_OFF:-0}" = "1" ]; then
  WANDB_ARGS="--no_wandb"
  log "wandb: 未设 key 或 WANDB_OFF=1，只落本地日志"
else
  log "wandb: 已检测到 WANDB_API_KEY，指标将同步在线"
fi

# 起点体检：模型目录必须有 config.json；有 merge_manifest.json 说明挂了 SFT
if [ ! -f "$MODEL/config.json" ]; then
  log "起点缺失: $MODEL 不是完整模型目录"
  log "先合并: python3 scripts/merge_sft_adapter.py \\"
  log "        --base models/Qwen3-4B-Instruct --adapter output/run_<时间戳>/sft_lora \\"
  log "        --out models/Qwen3-4B-Instruct-SFT"
  exit 1
fi
if [ -f "$MODEL/merge_manifest.json" ]; then
  log "起点: 已合并 SFT 的模型 $MODEL"
else
  log "起点: $MODEL（无 merge_manifest.json → 原厂基座，未挂 SFT；确认这是有意的对照实验）"
fi

# ---------------------------------------------------------------- 3. 后台启动
log "日志: $RUN_LOG"
CMD=("${LAUNCH[@]}" scripts/train_grpo.py \
  --data "$DATA" \
  --model_name_or_path "$MODEL" \
  --output_dir "$RUN_OUT" \
  --log_file "$RUN_LOG" \
  $VLLM_FLAG $WANDB_ARGS $GRPO_ARGS)
if [ -n "$EVAL_FILE" ]; then
  if [ ! -f "$EVAL_FILE" ]; then
    log "eval 基线集不存在: $EVAL_FILE"
    exit 1
  fi
  CMD+=(--eval_file "$EVAL_FILE" --eval_steps "$EVAL_STEPS"
        --fuse_threshold "$FUSE_THRESHOLD")
  if [ "$FUSE_HARD" = "1" ]; then CMD+=(--fuse_hard); fi
  log "eval 熔断: $EVAL_FILE（每 $EVAL_STEPS 步；阈值 ${FUSE_THRESHOLD}；"
  if [ "$FUSE_HARD" = "1" ]; then
    log "达到阈值将停止训练）"
  else
    log "只告警，不停止训练）"
  fi
fi

log "启动: ${CMD[*]}"
# 训练日志由脚本内 --log_file 双写（不会重复）。但 attach_log 装上之前的
# 崩溃（import 失败/参数错误）走不到 tee —— 落到 .boot 文件留证据，别扔黑洞。
BOOT_LOG="$RUN_LOG.boot"
nohup "${CMD[@]}" > "$BOOT_LOG" 2>&1 &
PID=$!
echo "$BOOT_LOG" > "$RUN_LOG.boot.path"
echo "$PID" > "$RUN_LOG.pid"
echo "$PID" > train_grpo_pid.txt

log "已后台启动 PID=$PID (train_grpo_pid.txt)"
log "查进度: tail -f $RUN_LOG"
log "停训练: kill $PID"
log "GPU:    nvidia-smi"
log "----------------------------------------------"
tail -n 20 "$RUN_LOG" || true
