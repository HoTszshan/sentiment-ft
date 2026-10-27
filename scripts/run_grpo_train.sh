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
#   DATA           默认 data/probe_test_grpo_train_synth.jsonl
#   MODEL          默认 models/Qwen3-4B-Instruct-SFT (先跑 merge_sft_adapter.py 产出)
#                  指向原厂基座 = 未挂 SFT 起训，只用于对照实验
#   OUTPUT_DIR     默认 output/grpo_lora
#   EVAL_FILE      可选 reward 熔断基线集 jsonl；传入才启用 eval callback
#   EVAL_STEPS     eval 间隔（默认 2 步）
#   FUSE_THRESHOLD 相对首轮 eval reward 的跌幅阈值（默认 2%）
#   FUSE_HARD      =1 时熔断停止训练；默认只告警
#   WANDB_API_KEY  设了就同步 wandb
#   GRPO_ARGS      透传给 train_grpo.py 的额外参数
#
# 拓扑 (TRL 0.14+ colocate vLLM，参考 12S2W10_rlhf/GRPO/run_train_2gpu.sh):
#   1 卡  → 不启用 vLLM，rollout 走 HF generate（探针 50 题足够）
#   2 卡  → 训练进程数 = GPU 数 - 1 = 1，vLLM 独占 cuda:1（use_vllm=auto）
#   ⚠️ 不要在脚本里重排 CUDA_VISIBLE_DEVICES（torch 初始化后改它不生效）
#
# accelerate 配置: config/accelerate_grpo.yaml（仅 GRPO 用；SFT 走 NPROC，不读本文件），
#   其中 num_processes: 1 是占位值，真实值由下方 --num_processes 覆盖：
#   --num_processes N = 启动 N 个训练进程（=N 张卡各占一个 DDP rank），
#   双卡拓扑下 N = GPU-1，剩余那张卡留给 colocate vLLM 独占采样。
# ==============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"
echo "工作目录: $BASE_DIR"

DATA="${DATA:-data/probe_test_grpo_train_synth.jsonl}"
MODEL="${MODEL:-models/Qwen3-4B-Instruct-SFT}"
OUTPUT_DIR="${OUTPUT_DIR:-output/grpo_lora}"
EVAL_FILE="${EVAL_FILE:-}"
EVAL_STEPS="${EVAL_STEPS:-5}"
FUSE_THRESHOLD="${FUSE_THRESHOLD:-0.02}"
FUSE_HARD="${FUSE_HARD:-0}"
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
  N_GPU="$(nvidia-smi -L 2>/dev/null | awk '/^GPU/{count++} END{print count+0}')"
fi
log "检测到 GPU 数: $N_GPU"

if [ "$N_GPU" -ge 2 ]; then
  # 双卡: 1 训练进程 (cuda:0) + vLLM 独占 (cuda:1)
  # 走 config/accelerate_grpo.yaml（与参考 run_train_2gpu.sh 同构），--num_processes 覆盖占位值
  LAUNCH=(accelerate launch --config_file config/accelerate_grpo.yaml --num_processes 1)
  VLLM_FLAG="--use_vllm"
  log "拓扑: accelerate 1 进程训练 (cuda:0) + vLLM 采样 (cuda:1)"
elif [ "$N_GPU" = "1" ]; then
  LAUNCH=(python3)
  VLLM_FLAG=""
  log "拓扑: 单卡，HF generate 采样 (探针 50 题够用；正式 3000 题建议双卡开 vLLM)"
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
  log "        --base models/Qwen3-4B-Instruct --adapter output/probe_lora_<时间戳> \\"
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
  --output_dir "$OUTPUT_DIR" \
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
