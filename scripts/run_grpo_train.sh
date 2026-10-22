#!/usr/bin/env bash
# ==============================================================================
# 舆情负面精判 · GRPO 训练启动脚本 (nohup 后台 + 时间戳日志 + PID 文件)
#
# 用法 (在 MVP/tests 目录):
#   ./scripts/run_grpo_train.sh                        # 自动按 GPU 数选拓扑
#   GRPO_ARGS="--limit 8 --max_steps 3" ./scripts/run_grpo_train.sh   # 冒烟
#   DRY=1 ./scripts/run_grpo_train.sh                  # 只跑 --dry_run 体检（无 GPU 可跑）
#
# 环境变量:
#   DATA           默认 data/grpo/grpo_prompts_probe.jsonl
#   MODEL          默认 models/Qwen3-4B-Instruct
#   RESUME_LORA    默认 output/probe_lora (SFT 探针产出的 adapter 起点；留空=裸基座)
#   OUTPUT_DIR     默认 output/grpo_lora
#   WANDB_API_KEY  设了就同步 wandb
#   GRPO_ARGS      透传给 train_grpo.py 的额外参数
#
# 拓扑 (TRL 0.14+ colocate vLLM，参考 12S2W10_rlhf/GRPO/run_train_2gpu.sh):
#   1 卡  → 不启用 vLLM，rollout 走 HF generate（探针 51 题足够）
#   2 卡  → 训练进程数 = GPU 数 - 1 = 1，vLLM 独占 cuda:1（use_vllm=auto）
#   ⚠️ 不要在脚本里重排 CUDA_VISIBLE_DEVICES（torch 初始化后改它不生效）
# ==============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"
echo "工作目录: $BASE_DIR"

DATA="${DATA:-data/grpo/grpo_prompts_probe.jsonl}"
MODEL="${MODEL:-models/Qwen3-4B-Instruct}"
RESUME_LORA="${RESUME_LORA:-output/probe_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-output/grpo_lora}"
GRPO_ARGS="${GRPO_ARGS:-}"
DRY="${DRY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" output

RUN_LOG="$LOG_DIR/grpo_train_${TS}.log"
log() { echo "[GRPO] $*" | tee -a "$RUN_LOG"; }

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

# ---------------------------------------------------------------- 2. 拓扑选择
N_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
  N_GPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)
fi
log "检测到 GPU 数: $N_GPU"

if [ "$N_GPU" -ge 2 ]; then
  # 双卡: 1 训练进程 (cuda:0) + vLLM 独占 (cuda:1)
  LAUNCH=(accelerate launch --num_processes 1)
  VLLM_FLAG="--use_vllm"
  log "拓扑: accelerate 1 进程训练 (cuda:0) + vLLM 采样 (cuda:1)"
elif [ "$N_GPU" = "1" ]; then
  LAUNCH=(python3)
  VLLM_FLAG=""
  log "拓扑: 单卡，HF generate 采样 (51 题探针够用；正式 3000 题建议双卡开 vLLM)"
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

# SFT adapter 起点（探针默认从它起，探针没跑就裸基座）
LORA_ARGS=""
if [ -n "$RESUME_LORA" ] && [ -f "$RESUME_LORA/adapter_config.json" ]; then
  LORA_ARGS="--resume_lora $RESUME_LORA"
  log "起点: SFT adapter $RESUME_LORA"
else
  log "起点: 裸基座 $MODEL（没找到 SFT adapter；探针建议先跑 run_probe_e2e.sh train）"
fi

# ---------------------------------------------------------------- 3. 后台启动
log "日志: $RUN_LOG"
CMD=("${LAUNCH[@]}" scripts/train_grpo.py \
  --data "$DATA" \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --log_file "$RUN_LOG" \
  $LORA_ARGS $VLLM_FLAG $WANDB_ARGS $GRPO_ARGS)

log "启动: ${CMD[*]}"
nohup "${CMD[@]}" >> "$RUN_LOG" 2>&1 &
PID=$!
echo "$PID" > "$RUN_LOG.pid"
echo "$PID" > train_grpo_pid.txt

log "已后台启动 PID=$PID (train_grpo_pid.txt)"
log "查进度: tail -f $RUN_LOG"
log "停训练: kill $PID"
log "GPU:    nvidia-smi"
log "----------------------------------------------"
tail -n 20 "$RUN_LOG" || true
