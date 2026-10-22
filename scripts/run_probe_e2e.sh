#!/usr/bin/env bash
# ==============================================================================
# 舆情负面精判 · 小样本探针实验 一键串跑 (MVP/tests)
#
# 用法 (在 MVP/tests 目录下，或任何位置直接执行本脚本):
#   ./scripts/run_probe_e2e.sh              # 全流程: 下载→混合→训练→推理→评测
#   ./scripts/run_probe_e2e.sh download     # 只跑第一步
#   ./scripts/run_probe_e2e.sh mix          # 只跑混合
#   ./scripts/run_probe_e2e.sh train        # 只跑训练
#   ./scripts/run_probe_e2e.sh test         # 只跑推理+评测
#
# 开关 (环境变量):
#   WANDB_API_KEY=xxx  设了就同步 wandb；不设则只落本地日志
#   MODEL_NAME=...     覆盖基座模型 (默认 Qwen/Qwen3-4B-Instruct-2507)
#   MODEL_DIR=...      模型存放目录 (默认 $BASE_DIR/models；AutoDL 可设 /root/autodl-tmp/models)
#   DATA_DIR=...       公开数据集与混合产物目录 (默认 $BASE_DIR/data；AutoDL 可设 /root/autodl-tmp/datasets)
#   DOMAIN_FILE=...    领域训练数据 (默认 $BASE_DIR/data/train_synth_81.jsonl，仓库内置不外移)
#   EPOCHS=3           覆盖训练轮数
#   FORCE=1            忽略已有产物强制重跑
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"
echo "工作目录: $BASE_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_DIR="${MODEL_DIR:-$BASE_DIR/models}"
DATA_DIR="${DATA_DIR:-$BASE_DIR/data}"
DOMAIN_FILE="${DOMAIN_FILE:-$BASE_DIR/data/train_synth_81.jsonl}"
ANCHOR_FILE="$DATA_DIR/coig_cqia_anchor.jsonl"
MIXED_DIR="$DATA_DIR/mixed"
EPOCHS="${EPOCHS:-3}"
FORCE="${FORCE:-0}"
# Loss 消融阶段 (POC1=仅前缀掩码 / POC2=+骨架掩码 / POC3=+字段加权 FIELD_W)
# 透传给 sft_weight_planner，一次只动一个变量
export SFT_POC_STAGE="${SFT_POC_STAGE:-POC1}"

TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR" output

RUN_LOG="$LOG_DIR/probe_e2e_${TS}.log"
log() { echo "[PROBE] $*" | tee -a "$RUN_LOG"; }

# ---------------------------------------------------------------- 0. 环境自检
check_deps() {
  log "环境自检..."
  missing=""
  for m in torch transformers peft; do
    python3 -c "import $m" 2>/dev/null || missing="$missing $m"
  done
  if [ -n "$missing" ]; then
    log "缺少依赖:$missing"
    log "安装: pip install torch transformers peft accelerate wandb"
    if [ "${SKIP_DEPS:-0}" != "1" ]; then exit 1; fi
  else
    log "训练依赖齐全"
  fi
  if [ -n "${WANDB_API_KEY:-}" ]; then
    log "wandb: 已检测到 WANDB_API_KEY，训练指标将同步在线"
    WANDB_FLAG=""
  else
    log "wandb: 未设 WANDB_API_KEY，只落本地日志 (需要同步请先 wandb login)"
    WANDB_FLAG="--no_wandb"
  fi
  python3 -c "import wandb" 2>/dev/null || { log "提示: 未安装 wandb (pip install wandb)"; WANDB_FLAG="--no_wandb"; }
}

# ---------------------------------------------------------------- 1. 下载
step_download() {
  log "步骤 1/5 · 下载公开 Anchor 数据集 + 基座模型 (data→$DATA_DIR, model→$MODEL_DIR)"
  mkdir -p "$DATA_DIR" "$MODEL_DIR"
  if [ "$FORCE" != "1" ] && [ -f "$ANCHOR_FILE" ]; then
    log "Anchor 已存在，跳过 (FORCE=1 可强制重下)"
  else
    python3 scripts/download_assets.py --download_anchor --anchor_samples 100 --data_dir "$DATA_DIR" | tee -a "$RUN_LOG"
  fi

  if [ "$FORCE" != "1" ] && { [ -d "$MODEL_DIR/$(basename "$MODEL_NAME")" ]; }; then
    log "基座模型已存在: $MODEL_DIR/$(basename "$MODEL_NAME")，跳过"
    LOCAL_MODEL="$MODEL_DIR/$(basename "$MODEL_NAME")"
  else
    log "下载基座: $MODEL_NAME (AutoDL 建议 ModelScope 国内直连)"
    python3 scripts/download_assets.py --model "$MODEL_NAME" --model_dir "$MODEL_DIR" --source modelscope | tee -a "$RUN_LOG"
    LOCAL_MODEL="$MODEL_DIR/$(basename "$MODEL_NAME")"
  fi
  log "基座本地路径: $LOCAL_MODEL"
}

# ---------------------------------------------------------------- 2. 混合
step_mix() {
  log "步骤 2/5 · 领域数据 : 公开 Anchor = 9 : 1 混合 + 分层切分 val"
  python3 scripts/mix_sft_dataset.py \
    --domain_file "$DOMAIN_FILE" \
    --anchor_file "$ANCHOR_FILE" \
    --out_dir "$MIXED_DIR" \
    --anchor_ratio 0.1 --val_ratio 0.1 --seed 42 2>&1 | tee -a "$RUN_LOG"

  # 混合日志与四查质检一并落盘，供回看与回填
  MIX_LOG="$LOG_DIR/mix_check_${TS}.log"
  python3 scripts/evaluate_sft.py --test_file "$MIXED_DIR/sft_train.jsonl" \
    --out "output/report_train_check_${TS}.json" > "$MIX_LOG" 2>&1 \
    && log "四查(Train) PASS -> $MIX_LOG" \
    || log "⚠️ 四查(Train) FAIL -> $MIX_LOG (详见 report_train_check_${TS}.json)"
}

# ---------------------------------------------------------------- 3. 训练
step_train() {
  log "步骤 3/5 · LoRA SFT 探针训练 (前缀掩码 + 骨架掩码 + FIELD_W 字段级加权)"
  TRAIN_LOG="$LOG_DIR/train_probe_${TS}.log"
  RUN_NAME="probe_${TS}"
  python3 scripts/train_sft_lora.py \
    --model_name_or_path "${LOCAL_MODEL:-$MODEL_NAME}" \
    --train_file "$MIXED_DIR/sft_train.jsonl" \
    --val_file "$MIXED_DIR/sft_val.jsonl" \
    --output_dir "output/probe_lora_${TS}" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --eval_steps 5 \
    --bf16 \
    --save_strategy no \
    --remove_unused_columns false \
    --log_file "$TRAIN_LOG" \
    --wandb_project sentiment-sft-probe \
    --wandb_run_name "$RUN_NAME" \
    $WANDB_FLAG 2>&1 | tee -a "$RUN_LOG"

  ADAPTER_DIR="output/probe_lora_${TS}"
  echo "$ADAPTER_DIR" > output/latest_adapter_path.txt
  log "Adapter 产出: $ADAPTER_DIR"
  log "训练日志   : $TRAIN_LOG"
}

# ---------------------------------------------------------------- 4. 推理
step_infer() {
  ADAPTER_DIR="$(cat output/latest_adapter_path.txt 2>/dev/null || echo "")"
  log "步骤 4/5 · 加载 Adapter 在内部 val 上批量推理"
  python3 scripts/probe_infer.py \
    --base_model "${LOCAL_MODEL:-$MODEL_NAME}" \
    --adapter_path "$ADAPTER_DIR" \
    --data_file "$MIXED_DIR/sft_val.jsonl" \
    --out "output/pred_val_${TS}.jsonl" 2>&1 | tee -a "$RUN_LOG"
  echo "output/pred_val_${TS}.jsonl" > output/latest_pred_path.txt
}

# ---------------------------------------------------------------- 5. 评测
step_eval() {
  log "步骤 5/5 · 评测验收 (模式 B: 金标 vs 预测)"
  PRED="$(cat output/latest_pred_path.txt 2>/dev/null || echo "")"
  if [ -z "$PRED" ] || [ ! -f "$PRED" ]; then
    log "未找到预测文件，跳过 (先跑 test)"
    return 0
  fi
  # 数据侧四查验收 (模式 A) —— 确认喂进去的语料本身是干净的
  python3 scripts/evaluate_sft.py --test_file "$MIXED_DIR/sft_train.jsonl" 2>&1 | tee -a "$RUN_LOG" || true
  # 模型对照评测 (模式 B)
  python3 scripts/evaluate_sft.py --gold_file "$MIXED_DIR/sft_val.jsonl" --pred_file "$PRED" 2>&1 | tee -a "$RUN_LOG"
  log "评测日志: $RUN_LOG"
}

# ---------------------------------------------------------------- 入口
STEP="${1:-all}"
check_deps
log "执行模式: $STEP"

case "$STEP" in
  download) step_download ;;
  mix)      step_mix ;;
  train)    step_train ;;
  test)     step_infer; step_eval ;;
  all)      step_download; step_mix; step_train; step_infer; step_eval ;;
  *) echo "未知步骤: $STEP (可选 download|mix|train|test|all)"; exit 2 ;;
esac

log "本轮探针结束 · 总日志: $RUN_LOG"
