#!/usr/bin/env bash
# ==============================================================================
# 舆情负面精判 · 小样本探针实验 一键串跑 (仓库根 /root/autodl-tmp/senti)
#
# 用法 (在仓库根目录下，或任何位置直接执行本脚本):
#   ./scripts/run_sft_e2e.sh              # 全流程: 下载→混合→训练→推理→评测
#   ./scripts/run_sft_e2e.sh download     # 只跑第一步
#   ./scripts/run_sft_e2e.sh mix          # 只跑混合
#   ./scripts/run_sft_e2e.sh train        # 只跑训练
#   ./scripts/run_sft_e2e.sh test         # 只跑推理+评测
#   ./scripts/run_sft_e2e.sh traintest    # 只跑训练+推理+评测 (复用已有混合数据与基座模型)
#   ./scripts/run_sft_e2e.sh --tag=POC1_epoch6 traintest
#                              # 同 traintest, 但输出目录名追加 tag 后缀
#                              # (run_20261024_180116 → run_20261024_180116_POC1_epoch6)
#                              # --tag 位置任意, 不传则保持原有输出路径不变
#
# traintest 说明: 数据(公开 Anchor + 领域数据混合产物)与基座模型都不重复下载/重复混合,
#                 直接读 $DATA_DIR/mixed/sft_{train,val}.jsonl 与 $MODEL_DIR 下的本地基座。
#                 缺产物时给出明确报错(用 mix / download 补齐), 不自动触发下载。
#
# 开关 (环境变量):
#   SYNC_WANDB=1       探针默认不上传 wandb (只落本地日志)；正式跑设此开关才同步
#                      (需已 wandb login 或设 WANDB_API_KEY)
#   MODEL_NAME=...     覆盖基座模型 (默认 Qwen/Qwen3-4B-Instruct-2507)
#   MODEL_DIR=...      模型存放目录 (缺省: /root/autodl-tmp/models，本机自动退回 $BASE_DIR/models)
#   DATA_DIR=...       公开数据集与混合产物目录 (缺省: /root/autodl-tmp/datasets，本机自动退回 $BASE_DIR/data)
#   DOMAIN_FILE=...    领域训练数据 (默认 $BASE_DIR/data/probe_test_sft_train_synth.jsonl（探针 81 条；正式批次用 DOMAIN_FILE=data/round1_train_aligned.jsonl 显式切换），仓库内置不外移)
#   EPOCHS=3           覆盖训练轮数
#   LR=1e-4            覆盖学习率 (对照实验用, 如 LR=5e-5 / LR=2e-4)
#   FORCE=1            忽略已有产物强制重跑
#   ANCHOR_MAX_CHARS=1200  锚点单条最大字符数 (混合前过滤, 防 OOM 长尾; 1200≈1680 token)
#   PER_DEV_BATCH=2       单卡 batch (默认 2; 有效 batch 恒 8, accum 自动折算; OOM 时设 1)
#   INFER_BACKEND=hf      推理后端: hf (单卡 PyTorch 原生，默认推荐) | vllm (高吞吐批量推理)
#   INFER_GPU=0           方案 1 (hf) 时绑定的单卡编号 (默认 0)
#   TRAINER_BACKEND=...  训练器后端 (默认 auto: 有 trl 用 trl.SFTTrainer 否则 HF Trainer;
#                        显式 trl 缺库即报错; 显式 hf 强制回退 HF Trainer)
#   BG=1                  后台执行: nohup + PID 文件 + 日志重定向, 断 SSH 不死
#                         (正式训练 3~6 小时必用; 探针 2~3 分钟可前台直跑)
#                         三件套: tail -f <nohup日志> / ps -p $(cat <pid文件>) / kill $(cat <pid文件>)
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"
echo "工作目录: $BASE_DIR"

# 与 download_assets.py 的默认逻辑对齐：环境变量 > AutoDL 数据盘 > 仓库本地
if [ -n "$MODEL_DIR" ]; then :;
elif [ -d /root/autodl-tmp ]; then MODEL_DIR=/root/autodl-tmp/models;
else MODEL_DIR="$BASE_DIR/models"; fi
if [ -n "$DATA_DIR" ]; then :;
elif [ -d /root/autodl-tmp ]; then DATA_DIR=/root/autodl-tmp/datasets;
else DATA_DIR="$BASE_DIR/data"; fi
# 微调产物（LoRA adapter / 预测 / 评测报告 / 日志）与基座同盘：
#   AutoDL 下 /root/autodl-tmp/models/sft_senti/，本机退回 $BASE_DIR/output
if [ -n "$OUTPUT_DIR" ]; then :;
elif [ -d /root/autodl-tmp ]; then OUTPUT_DIR=/root/autodl-tmp/models/sft_senti;
else OUTPUT_DIR="$BASE_DIR/output"; fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
DOMAIN_FILE="${DOMAIN_FILE:-$BASE_DIR/data/probe_test_sft_train_synth.jsonl}"
ANCHOR_FILE="$DATA_DIR/coig_cqia_anchor.jsonl"
MIXED_DIR="$DATA_DIR/mixed"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
echo "路径解析: MODEL_DIR=$MODEL_DIR  DATA_DIR=$DATA_DIR  DOMAIN_FILE=$DOMAIN_FILE"
FORCE="${FORCE:-0}"
# Loss 消融阶段 (POC1=仅前缀掩码 / POC2=+骨架掩码 / POC3=+字段加权 FIELD_W)
# 透传给 sft_weight_planner，一次只动一个变量
export SFT_POC_STAGE="${SFT_POC_STAGE:-POC1}"

TS=$(date +"%Y%m%d_%H%M%S")

# 参数解析: 第一个非 --tag 参数为步骤名, --tag=XXX 为实验标签
# (传入 tag: 输出目录名追加后缀, 如 run_20261024_180116_POC1_epoch6; 不传保持原样)
STEP="all"
TAG=""
for arg in "$@"; do
  case "$arg" in
    --tag=*) TAG="${arg#--tag=}" ;;
    *) STEP="$arg" ;;
  esac
done

# 显存碎片缓解：大 vocab(151936)×seq2560 的 logits/loss 临时块 + DDP 通信 buffer
# 容易把 reserved-but-unallocated 顶高；expandable_segments 让 caching allocator
# 按需扩展段而非按 2 的幂整块预留，显著降低 OOM 概率（对 3090 24G 尤其有效）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 单次实验独立目录: 本轮所有产物 (adapter/预测/报告) 与日志全收在 run_<TS>/ 下,
# 一个目录 = 一次完整实验, 拷走即归档; latest 指针放 OUTPUT_DIR 顶层供跨步衔接
RUN_DIR="$OUTPUT_DIR/run_${TS}${TAG:+_${TAG}}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$OUTPUT_DIR"

# LOCAL_MODEL 只在 step_download 里赋值；train/test 单步跑时为空。
# 兜底：若 $MODEL_DIR 下已有同名目录则直接用本地副本，避免误走 HF Hub 联网下载。
resolve_local_model() {
  if [ -z "${LOCAL_MODEL:-}" ] && [ -d "$MODEL_DIR/$(basename "$MODEL_NAME")" ]; then
    LOCAL_MODEL="$MODEL_DIR/$(basename "$MODEL_NAME")"
    log "基座本地路径(探测): $LOCAL_MODEL"
  fi
}

RUN_LOG="$LOG_DIR/probe_e2e.log"
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
  # wandb 默认关：探针实验不上传，正式跑才 SYNC_WANDB=1 显式开启。
  # 登录态有两条路径: WANDB_API_KEY 环境变量 或 ~/.netrc 里的 api.wandb.ai 凭据
  if [ "${SYNC_WANDB:-0}" != "1" ]; then
    log "wandb: 探针模式默认关闭，只落本地日志 (正式跑: SYNC_WANDB=1 ./scripts/run_sft_e2e.sh)"
    WANDB_FLAG="--no_wandb"
  elif ! python3 -c "import wandb" 2>/dev/null; then
    log "wandb: SYNC_WANDB=1 但未安装 wandb，降级为本地日志 (pip install wandb)"
    WANDB_FLAG="--no_wandb"
  elif [ -n "${WANDB_API_KEY:-}" ] || grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    log "wandb: SYNC_WANDB=1 且已登录，训练指标将同步在线"
    WANDB_FLAG=""
  else
    log "wandb: SYNC_WANDB=1 但未登录，降级为本地日志 (wandb login 或设 WANDB_API_KEY)"
    WANDB_FLAG="--no_wandb"
  fi
}

# ---------------------------------------------------------------- 0.5 复用产物自检
# traintest 模式专用: 确认"已混合数据 + 本地基座"就位, 缺失则直接报错退出,
# 不触发任何下载或混合 (那两步由 download / mix 单独负责)。
check_reuse_assets() {
  miss_data=""
  miss_model=""
  if [ ! -f "$MIXED_DIR/sft_train.jsonl" ] || [ ! -f "$MIXED_DIR/sft_val.jsonl" ]; then
    miss_data="1"
  fi
  if [ ! -d "$MODEL_DIR/$(basename "$MODEL_NAME")" ]; then
    miss_model="1"
  fi
  if [ -n "$miss_data" ] || [ -n "$miss_model" ]; then
    log "❌ 缺少可复用产物:"
    [ -n "$miss_data" ]  && log "   混合数据缺失 ($MIXED_DIR/sft_train.jsonl, sft_val.jsonl) → 先跑: ./scripts/run_sft_e2e.sh mix"
    [ -n "$miss_model" ] && log "   基座模型缺失 ($MODEL_DIR/$(basename "$MODEL_NAME"))            → 先跑: ./scripts/run_sft_e2e.sh download"
    log "   (也可用 DATA_DIR / MODEL_DIR / MODEL_NAME 指定现有产物位置)"
    # 自检失败即退出：清掉本次创建的空 run 目录，避免留下无用产物
    rm -rf "$RUN_DIR"
    exit 1
  fi
  log "复用资产自检通过: 混合数据=$MIXED_DIR  基座=$MODEL_DIR/$(basename "$MODEL_NAME")"
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
    --anchor_ratio 0.1 --val_ratio 0.1 --seed 42 \
    --anchor_max_chars "${ANCHOR_MAX_CHARS:-1200}" 2>&1 | tee -a "$RUN_LOG"

  # 混合日志与四查质检一并落盘，供回看与回填
  MIX_LOG="$LOG_DIR/mix_check.log"
  python3 scripts/evaluate_sft.py --test_file "$MIXED_DIR/sft_train.jsonl" \
    --out "$RUN_DIR/report_train_check.json" > "$MIX_LOG" 2>&1 \
    && log "四查(Train) PASS -> $MIX_LOG" \
    || log "⚠️ 四查(Train) FAIL -> $MIX_LOG (详见 report_train_check.json)"
}

# ---------------------------------------------------------------- 3. 训练
step_train() {
  log "步骤 3/5 · LoRA SFT 训练 (前缀掩码 + 骨架掩码 + FIELD_W 字段级加权)"
  resolve_local_model
  TRAIN_LOG="$LOG_DIR/train_probe.log"
  RUN_NAME="probe_${TS}"

  # ---- 启动方式选择: 自动探测 GPU 数, 可用 NPROC 环境变量覆盖 ----
  #   1 卡 → python3 直启
  #   2 卡 → accelerate launch --multi_gpu DDP (不用 DeepSpeed)
  # 有效 batch 恒为 8 (消融口径不变): ACCUM = 8 / (PER_DEV_BATCH × NPROC)
  #   默认 per_dev=2: 单卡 2×4=8, 双卡 2×2×2=8; OOM 时 PER_DEV_BATCH=1 退回保守档
  # 显存注记: 3090×2 DDP 曾在「未做 anchor 长度过滤(1835 token 长尾)+batch=2」下顶穿 23.5G;
  #           anchor 过滤后全库 max≈719 token, batch=2 峰值回落 ~2.5 倍, 默认 2 可用。
  if [ -z "${NPROC:-}" ]; then
    NPROC=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    [ "$NPROC" -lt 1 ] && NPROC=1
  fi
  PER_DEV_BATCH="${PER_DEV_BATCH:-2}"
  ACCUM=$(( 8 / (PER_DEV_BATCH * NPROC) ))
  if [ "$ACCUM" -lt 1 ] || [ $(( PER_DEV_BATCH * NPROC * ACCUM )) -ne 8 ]; then
    log "⚠️ PER_DEV_BATCH=$PER_DEV_BATCH × NPROC=$NPROC 不能整除有效 batch 8, 回退 per_dev=1"
    PER_DEV_BATCH=1
    ACCUM=$(( 8 / NPROC ))
  fi
  if [ "$NPROC" -gt 1 ]; then
    log "启动方式: accelerate launch --multi_gpu (NPROC=$NPROC, per_dev=$PER_DEV_BATCH, accum=$ACCUM, eff_batch=8)"
    LAUNCH_CMD=(accelerate launch --multi_gpu --num_processes "$NPROC"
                --mixed_precision bf16)
  else
    log "启动方式: python3 直启 (单卡 per_dev=$PER_DEV_BATCH, accum=$ACCUM, eff_batch=8)"
    LAUNCH_CMD=(python3)
  fi
  TRAIN_ARGS=(--gradient_accumulation_steps "$ACCUM")

  # 显存紧时的杠杆选项 (MAX_LEN=... 覆盖):
  #   2560 = SOP 定值 (P99); 数据实测 P95=650 / max=1821 → 2560 只是留安全边际,
  #          砍到 1024 只截尾部超长样本, 但 activation/logits 随 seq 平方省。
  #   探针已开 gradient_checkpointing (activation -80%), 2560 可扛;
  #   若仍 OOM 就 MAX_LEN=1536/1024 再压, 或直接显式 --max_seq_length。
  MAX_LEN="${MAX_LEN:-2560}"

  # 协议 token 解冻：默认开；设置 UNFREEZE_PROTOCOL_TOKENS=0 可关（消融对照用）
  UNFREEZE_ARG=""
  [ "${UNFREEZE_PROTOCOL_TOKENS:-1}" = "0" ] && UNFREEZE_ARG="--no_unfreeze_protocol_tokens"

  "${LAUNCH_CMD[@]}" scripts/train_sft_lora.py \
    --model_name_or_path "${LOCAL_MODEL:-$MODEL_NAME}" \
    --train_file "$MIXED_DIR/sft_train.jsonl" \
    --val_file "$MIXED_DIR/sft_val.jsonl" \
    --output_dir "$RUN_DIR/sft_lora" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PER_DEV_BATCH" \
    --max_seq_length "$MAX_LEN" \
    "${TRAIN_ARGS[@]}" \
    --learning_rate "${LR:-1e-4}" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --eval_steps 5 \
    --bf16 \
    --save_strategy no \
    --remove_unused_columns false \
    --trainer_backend "${TRAINER_BACKEND:-auto}" \
    --think_prefix "${THINK_PREFIX:-0}" \
    ${UNFREEZE_ARG} \
    --canary_every "${CANARY_EVERY:-30}" \
    --canary_n "${CANARY_N:-3}" \
    --canary_max_new "${CANARY_MAX_NEW:-400}" \
    --log_file "$TRAIN_LOG" \
    --wandb_project sentiment-sft-probe4 \
    --wandb_run_name "$RUN_NAME" \
    $WANDB_FLAG 2>&1 | tee -a "$RUN_LOG"
  train_rc=${PIPESTATUS[0]}

  ADAPTER_DIR="$RUN_DIR/sft_lora"
  if [ "$train_rc" -ne 0 ]; then
    log "⚠️ 训练失败 (exit=$train_rc)，跳过 adapter 登记 —— 详见 $TRAIN_LOG"
    return "$train_rc"
  fi
  echo "$ADAPTER_DIR" > "$OUTPUT_DIR/latest_adapter_path.txt"
  log "Adapter 产出: $ADAPTER_DIR"
  log "训练日志   : $TRAIN_LOG"
}

# ---------------------------------------------------------------- 4. 推理
step_infer() {
  resolve_local_model
  ADAPTER_DIR="$(cat "$OUTPUT_DIR/latest_adapter_path.txt" 2>/dev/null || echo "")"
  INFER_BACKEND="${INFER_BACKEND:-hf}"
  INFER_GPU="${INFER_GPU:-0}"
  log "步骤 4/5 · 加载 Adapter 在内部 val 上批量推理 (后端: $INFER_BACKEND)"
  # 渲染协议必须与训练侧一致：THINK_PREFIX=1 时训练把 <think>\n 写进了前缀，
  # 推理侧不加同一 flag 就会「训练有前缀、推理没前缀」，格式立刻崩。
  THINK_PREFIX_ARG=""
  [ "${THINK_PREFIX:-0}" = "1" ] && THINK_PREFIX_ARG="--think_prefix"

  if [ "$INFER_BACKEND" = "vllm" ]; then
    python3 scripts/sft_infer.py \
      --backend vllm \
      --base_model "${LOCAL_MODEL:-$MODEL_NAME}" \
      --adapter_path "$ADAPTER_DIR" \
      --data_file "$MIXED_DIR/sft_val.jsonl" \
      --out "$RUN_DIR/pred_val.jsonl" \
      $THINK_PREFIX_ARG 2>&1 | tee -a "$RUN_LOG"
  else
    # 方案 1: 显式单卡环境绑定，避免 PEFT trainable_tokens 多卡切分张量不一致 bug
    CUDA_VISIBLE_DEVICES="$INFER_GPU" python3 scripts/sft_infer.py \
      --backend hf \
      --base_model "${LOCAL_MODEL:-$MODEL_NAME}" \
      --adapter_path "$ADAPTER_DIR" \
      --data_file "$MIXED_DIR/sft_val.jsonl" \
      --out "$RUN_DIR/pred_val.jsonl" \
      $THINK_PREFIX_ARG 2>&1 | tee -a "$RUN_LOG"
  fi
  echo "$RUN_DIR/pred_val.jsonl" > "$OUTPUT_DIR/latest_pred_path.txt"
}

# ---------------------------------------------------------------- 5. 评测
step_eval() {
  log "步骤 5/5 · 评测验收 (模式 B: 金标 vs 预测)"
  PRED="$(cat "$OUTPUT_DIR/latest_pred_path.txt" 2>/dev/null || echo "")"
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

# ---- BG=1: 自我转后台 (nohup + PID 文件 + 日志重定向), 断 SSH 不死 ----
# 正式训练 3~6 小时必用; 探针 2~3 分钟可前台直跑。实现: 递归调用自身一次,
# 子进程经 nohup 脱离终端, 父进程打印三件套用法后立即退出。
if [ "${BG:-0}" = "1" ] && [ -z "${BG_CHILD:-}" ]; then
  BG_TS=$(date +"%Y%m%d_%H%M%S")
  BG_LOG="nohup_${STEP}_${BG_TS}.log"
  BG_PID="train_${BG_TS}.pid"
  nohup env BG_CHILD=1 "$0" "$@" > "$BG_LOG" 2>&1 &
  echo $! > "$BG_PID"
  echo "[BG] 已转后台执行: $*"
  echo "[BG] 看进度 : tail -f $BG_LOG"
  echo "[BG] 确认活着: ps -p \$(cat $BG_PID) && echo 还在跑 || echo 已结束"
  echo "[BG] 停止   : kill \$(cat $BG_PID)"
  echo "[BG] 总控日志与产物仍按内部路径落盘 (OUTPUT_DIR/run_<TS>/logs/)"
  exit 0
fi

check_deps
log "执行模式: $STEP"
[ -n "$TAG" ] && log "实验标签: $TAG (输出目录: run_${TS}_${TAG})"
[ -n "$TAG" ] && log "实验标签: $TAG (输出目录: run_${TS}_${TAG})"

case "$STEP" in
  download) step_download ;;
  mix)      step_mix ;;
  train)    step_train ;;
  test)     step_infer; step_eval ;;
  traintest) check_reuse_assets; step_train; step_infer; step_eval ;;
  all)      step_download; step_mix; step_train; step_infer; step_eval ;;
  *) echo "未知步骤: $STEP (可选 download|mix|train|test|traintest|all)"; exit 2 ;;
esac

log "本轮探针结束 · 总日志: $RUN_LOG"
