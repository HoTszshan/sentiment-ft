#!/usr/bin/env bash
# ==============================================================================
# 舆情负面精判 · 小样本探针实验 一键串跑 (仓库根 /root/autodl-tmp/senti)
#
# 用法 (在仓库根目录下，或任何位置直接执行本脚本):
#   ./scripts/run_sft_e2e.sh              # 全流程: 下载→混合→训练→合并→推理→评测
#   ./scripts/run_sft_e2e.sh download     # 只跑第一步
#   ./scripts/run_sft_e2e.sh mix          # 只跑混合
#   ./scripts/run_sft_e2e.sh train        # 只跑训练
#   ./scripts/run_sft_e2e.sh merge        # 只跑合并 (adapter → 独立模型)
#   ./scripts/run_sft_e2e.sh test         # 只跑合并+推理+评测
#   ./scripts/run_sft_e2e.sh traintest    # 只跑训练+合并+推理+评测 (复用已有混合数据与基座模型)
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
#   WANDB_PROJECT=...  覆盖 wandb project (默认 sentiment-sft-probe4)
#   WANDB_RUN_NAME=... 覆盖 wandb run 名 (默认 probe_<TS>[_<TAG>])
#   MODEL_NAME=...     覆盖基座模型 (默认 Qwen/Qwen3-4B-Instruct-2507)
#   MODEL_DIR=...      模型存放目录 (缺省: /root/autodl-tmp/models，本机自动退回 $BASE_DIR/models)
#   DATA_DIR=...       公开数据集与混合产物目录 (缺省: /root/autodl-tmp/datasets，本机自动退回 $BASE_DIR/data)
#   DOMAIN_FILE=...    领域训练数据 (默认 $BASE_DIR/data/probe_test_sft_train_synth.jsonl（探针 81 条；正式批次用 DOMAIN_FILE=data/round1_train_aligned.jsonl 显式切换），仓库内置不外移)
#   EPOCHS=3           覆盖训练轮数
#   LR=1e-4            覆盖学习率 (对照实验用, 如 LR=5e-5 / LR=2e-4)
#   FORCE=1            忽略已有产物强制重跑
#   ANCHOR_MAX_CHARS=1200  锚点单条最大字符数 (混合前过滤, 防 OOM 长尾; 1200≈1680 token)
#   PER_DEV_BATCH=2       单卡 batch (默认 2; 有效 batch 恒 8, accum 自动折算; OOM 时设 1)
#   INFER_BACKEND=auto    推理后端: auto (默认, 探测到 vLLM 用 vllm, 否则回退 hf) | hf | vllm
#   INFER_DATA=...        推理数据 (默认 $MIXED_DIR/sft_val.jsonl; 黄金集: INFER_DATA=data/gold500_test.jsonl)
#   ADAPTER_DIR=...       显式指定 adapter 目录 (默认读 latest_adapter_path.txt 指针; 多 run 对照必用)
#   CLEAN_MERGE=1         评测完成后删除合并模型 (~7.6G; adapter/报告/预测保留, 可重新 merge 复原)
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
echo "路径解析: MODEL_DIR=$MODEL_DIR  DATA_DIR=$DATA_DIR"
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
  # wandb run 名可覆盖；默认带 tag，对照组在 wandb 看板上一眼可辨
  RUN_NAME="${WANDB_RUN_NAME:-probe_${TS}${TAG:+_${TAG}}}"

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
    --per_device_eval_batch_size "${PER_DEV_EVAL_BATCH:-1}" \
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
    --wandb_project "${WANDB_PROJECT:-sentiment-sft}" \
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

# ---------------------------------------------------------------- 4. 合并 (vLLM 前置条件)
# 为什么要有这一步: vLLM 0.7.x 对 Qwen3 无原生实现 → 回退 Transformers 后端 → 该后端
# 不支持 LoRA ("TransformersModel does not support LoRA yet")。「基座+adapter」直接走
# vllm 必崩。解法: 训练后先把 adapter merge 进基座, 推理统一用合并产物 (不再挂 LoRA)。
# 合并产物缓存在 OUTPUT_DIR/merged/<adapter目录名>/, 重复跑 test 不重复 merge。
step_merge() {
  # adapter 来源优先级: ADAPTER_DIR 环境变量 (显式指定, 不依赖指针文件) > latest_adapter_path.txt
  ADAPTER_DIR="${ADAPTER_DIR:-$(cat "$OUTPUT_DIR/latest_adapter_path.txt" 2>/dev/null || echo "")}"
  if [ -z "$ADAPTER_DIR" ] || [ ! -d "$ADAPTER_DIR" ]; then
    log "⚠️ 未找到 adapter (设 ADAPTER_DIR=... 显式指定, 或先跑 train 生成 $OUTPUT_DIR/latest_adapter_path.txt)，跳过 merge"
    return 0
  fi
  log "adapter 来源: $ADAPTER_DIR ${ADAPTER_DIR:+(显式指定)}"
  resolve_local_model
  local BASE="${LOCAL_MODEL:-$MODEL_DIR/$(basename "$MODEL_NAME")}"
  MERGED_DIR="$OUTPUT_DIR/merged/$(basename "$(dirname "$ADAPTER_DIR")")_$(basename "$ADAPTER_DIR")"
  echo "$MERGED_DIR" > "$OUTPUT_DIR/latest_merged_path.txt"

  if [ -d "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ] && [ "$FORCE" != "1" ]; then
    log "合并产物已存在，跳过 merge: $MERGED_DIR (FORCE=1 强制重合并)"
    return 0
  fi
  log "步骤 4/6 · 合并 adapter → 独立模型 (fp32 合并+校验, bf16 落盘)"
  python3 scripts/merge_sft_adapter.py \
    --base "$BASE" \
    --adapter "$ADAPTER_DIR" \
    --out "$MERGED_DIR" \
    --dtype fp32 2>&1 | tee -a "$RUN_LOG"
  local merge_rc=${PIPESTATUS[0]}
  if [ "$merge_rc" -ne 0 ]; then
    log "❌ merge 失败 (exit=$merge_rc)，删除半成品并中止推理"
    rm -rf "$MERGED_DIR"
    return "$merge_rc"
  fi
  # fp32 合并仅为让 logits 校验通过 (bf16 舍入噪声会触发 top1Δp 误报);
  # 落盘转 bf16 —— vLLM 加载/显存/速度均按 bf16, 体积减半
  # ⚠️ 不能写 `python3 - ARG <<EOF | tee`: heredoc 会绑定给管道末端的 tee 而非 python,
  #    后果是 tee 把源码当文本回显、python 读到空 stdin 静默退出 (bf16 目录不产生,
  #    fp32 目录不删) —— 234853/235538 两轮跑挂即此坑。落成临时脚本文件执行。
  local CAST_SCRIPT="$LOG_DIR/cast_bf16.py"
  cat > "$CAST_SCRIPT" <<'PYEOF'
import sys, json, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
src = Path(sys.argv[1]); dst = Path(str(src) + "-bf16")
model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(src)
model.save_pretrained(dst, safe_serialization=True); tok.save_pretrained(dst)
cfg = json.loads((dst/"config.json").read_text()); cfg["torch_dtype"] = "bfloat16"
(dst/"config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
m = json.loads((src/"merge_manifest.json").read_text())
m["bf16_cast"] = {"cast_from": str(src), "cast_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "parity_vs_fp32": m["parity"],
                  "note": "fp32 合并校验通过后 cast bf16; bf16 目录为推理/评测正式产物"}
(dst/"merge_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
import shutil; shutil.rmtree(src)
print(f"[OK] bf16 产物: {dst} (fp32 中间件已清理)")
PYEOF
  python3 "$CAST_SCRIPT" "$MERGED_DIR" 2>&1 | tee -a "$RUN_LOG"
  if [ ! -f "$MERGED_DIR-bf16/config.json" ]; then
    log "❌ bf16 转存未产出，中止 (fp32 中间件保留待排查)"
    return 1
  fi
  MERGED_DIR="$OUTPUT_DIR/merged/$(basename "$(dirname "$ADAPTER_DIR")")_$(basename "$ADAPTER_DIR")-bf16"
  echo "$MERGED_DIR" > "$OUTPUT_DIR/latest_merged_path.txt"
  log "合并产物: $MERGED_DIR"
  # 评测通过后可设 CLEAN_MERGE=1 删掉 ~7.6G 合并模型腾盘;
  # 实际删除点在 step_eval 末尾 (要等 gate 结果确认后才删), 这里只登记
  CLEAN_MERGE="${CLEAN_MERGE:-0}"
}

# ---------------------------------------------------------------- 5. 推理
step_infer() {
  resolve_local_model
  # 与 step_merge 同源: 显式 ADAPTER_DIR 优先 (否则指针), 且用同一条路径规则重算合并产物,
  # 保证「显式传参单步跑」与「traintest 链路」两边找到的是同一个合并目录
  ADAPTER_DIR="${ADAPTER_DIR:-$(cat "$OUTPUT_DIR/latest_adapter_path.txt" 2>/dev/null || echo "")}"
  MERGED_DIR="$(cat "$OUTPUT_DIR/latest_merged_path.txt" 2>/dev/null || echo "")"
  # 推理数据可覆盖: 默认内部 val; INFER_DATA=data/gold500_test.jsonl 切黄金测试集
  INFER_DATA="${INFER_DATA:-$MIXED_DIR/sft_val.jsonl}"
  PRED_NAME="pred_$(basename "$INFER_DATA" .jsonl).jsonl"
  # 推理底模选择: 有合并产物用合并产物 (裸模型, 不挂 adapter);
  # 无合并产物才退回「基座+adapter」—— 此时 vllm+LoRA 对 Qwen3 会崩, 自动强制 hf。
  INF_BASE="${LOCAL_MODEL:-$MODEL_NAME}"
  INF_ADAPTER="$ADAPTER_DIR"
  if [ -n "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ]; then
    INF_BASE="$MERGED_DIR"; INF_ADAPTER=""
    log "推理底模: 合并产物 $MERGED_DIR (不挂 adapter)"
  elif [ -n "$ADAPTER_DIR" ] && [ -d "$ADAPTER_DIR" ]; then
    # 合并产物存在但按显式 ADAPTER_DIR 的路径规则重算 (跨 run / 单步跑场景)
    local MD="$OUTPUT_DIR/merged/$(basename "$(dirname "$ADAPTER_DIR")")_$(basename "$ADAPTER_DIR")-bf16"
    if [ -f "$MD/config.json" ]; then
      INF_BASE="$MD"; INF_ADAPTER=""
      MERGED_DIR="$MD"
      log "推理底模: 合并产物 $MD (不挂 adapter)"
    else
      log "推理底模: 基座+adapter ($INF_BASE + $ADAPTER_DIR)"
      INF_ADAPTER="$ADAPTER_DIR"
      if [ -z "${INFER_BACKEND:-}" ] || [ "${INFER_BACKEND:-}" = "vllm" ]; then
        # 探测收紧: vllm+adapter 组合仅当模型有 vllm 原生实现时可用;
        # Qwen3/Qwen3MoE 等在 vllm 0.7.x 无原生实现 → 回退 Transformers 后端 → LoRA 崩。
        if ! python3 -c "
import json, sys
arch = json.load(open('$INF_BASE/config.json'))['architectures'][0]
sys.exit(0 if arch in ('LlamaForCausalLM','MistralForCausalLM','Qwen2ForCausalLM') else 1)" 2>/dev/null; then
          INFER_BACKEND="hf"
          log "⚠️ 该架构挂 LoRA 走 vllm 会崩 (无原生实现, 回退 Transformers 后端不支持 LoRA) → 自动改用 hf"
        fi
      fi
    fi
  else
    log "推理底模: 基座+adapter ($INF_BASE + $INF_ADAPTER)"
    if [ -z "${INFER_BACKEND:-}" ] || [ "${INFER_BACKEND:-}" = "vllm" ]; then
      # 探测收紧: vllm+adapter 组合仅当模型有 vllm 原生实现时可用;
      # Qwen3/Qwen3MoE 等在 vllm 0.7.x 无原生实现 → 回退 Transformers 后端 → LoRA 崩。
      if ! python3 -c "
import json, sys
arch = json.load(open('$INF_BASE/config.json'))['architectures'][0]
sys.exit(0 if arch in ('LlamaForCausalLM','MistralForCausalLM','Qwen2ForCausalLM') else 1)" 2>/dev/null; then
        INFER_BACKEND="hf"
        log "⚠️ 该架构挂 LoRA 走 vllm 会崩 (无原生实现, 回退 Transformers 后端不支持 LoRA) → 自动改用 hf"
      fi
    fi
  fi
  # 后端选择: 未显式指定时自动探测 —— 有 vLLM 就走 vLLM (高吞吐批量推理),
  # 没有再回退 hf 原生; 显式 INFER_BACKEND=hf|vllm 仍可强制覆盖。
  INFER_GPU="${INFER_GPU:-0}"
  if [ -z "${INFER_BACKEND:-}" ]; then
    if python3 -c "import vllm" 2>/dev/null; then
      INFER_BACKEND="vllm"
      log "推理后端: 未指定, 探测到 vLLM → 自动使用 vllm (强制 hf 请设 INFER_BACKEND=hf)"
    else
      INFER_BACKEND="hf"
      log "推理后端: 未指定, 未安装 vLLM → 回退 hf (装 vllm 后自动切换)"
    fi
  fi
  log "步骤 5/6 · 批量推理 $INFER_DATA (后端: $INFER_BACKEND, 底模: $INF_BASE)"
  # 渲染协议必须与训练侧一致：THINK_PREFIX=1 时训练把 <think>\n 写进了前缀，
  # 推理侧不加同一 flag 就会「训练有前缀、推理没前缀」，格式立刻崩。
  THINK_PREFIX_ARG=""
  [ "${THINK_PREFIX:-0}" = "1" ] && THINK_PREFIX_ARG="--think_prefix"

  if [ "$INFER_BACKEND" = "vllm" ]; then
    python3 scripts/sft_infer.py \
      --backend vllm \
      --base_model "$INF_BASE" \
      --adapter_path "$INF_ADAPTER" \
      --data_file "$INFER_DATA" \
      --out "$RUN_DIR/$PRED_NAME" \
      $THINK_PREFIX_ARG 2>&1 | tee -a "$RUN_LOG"
  else
    # 方案 1: 显式单卡环境绑定，避免 PEFT trainable_tokens 多卡切分张量不一致 bug
    CUDA_VISIBLE_DEVICES="$INFER_GPU" python3 scripts/sft_infer.py \
      --backend hf \
      --base_model "$INF_BASE" \
      --adapter_path "$INF_ADAPTER" \
      --data_file "$INFER_DATA" \
      --out "$RUN_DIR/$PRED_NAME" \
      $THINK_PREFIX_ARG 2>&1 | tee -a "$RUN_LOG"
  fi
  echo "$RUN_DIR/$PRED_NAME" > "$OUTPUT_DIR/latest_pred_path.txt"
}

# ---------------------------------------------------------------- 6. 评测
step_eval() {
  log "步骤 6/6 · 评测验收 (模式 B: 金标 vs 预测)"
  PRED="$(cat "$OUTPUT_DIR/latest_pred_path.txt" 2>/dev/null || echo "")"
  if [ -z "$PRED" ] || [ ! -f "$PRED" ]; then
    log "未找到预测文件，跳过 (先跑 test)"
    return 0
  fi
  # 金标与预测同目录同名 (pred_<数据名>.jsonl ↔ gold500_test.jsonl / sft_val.jsonl);
  # INFER_DATA 指向哪份, 评测就对哪份 —— 指针机制与 gold500 场景对齐
  local GOLD="$(basename "$PRED" .jsonl)"
  GOLD="${GOLD#pred_}"
  # 数据名 → 金标文件: gold500_test → data/gold500_test.jsonl; sft_val → 混合目录
  if [ -f "data/$GOLD.jsonl" ]; then
    GOLD_FILE="data/$GOLD.jsonl"
  else
    GOLD_FILE="$MIXED_DIR/$GOLD.jsonl"
  fi
  log "评测对象: gold=$GOLD_FILE  pred=$PRED"
  # 数据侧四查验收 (模式 A) —— 确认喂进去的语料本身是干净的
  python3 scripts/evaluate_sft.py --test_file "$MIXED_DIR/sft_train.jsonl" 2>&1 | tee -a "$RUN_LOG" || true
  # 模型对照评测 (模式 B)
  python3 scripts/evaluate_sft.py --gold_file "$GOLD_FILE" --pred_file "$PRED" \
    --out "$RUN_DIR/report_$(basename "$GOLD_FILE" .jsonl).json" 2>&1 | tee -a "$RUN_LOG"
  log "评测日志: $RUN_LOG"
  # CLEAN_MERGE=1: 评测完成后删除本轮合并模型 (~7.6G)。adapter (142M) 与报告/预测保留,
  # 模型可随时由 adapter 重新 merge 复原; 下轮 test 会自动重 merge。
  if [ "${CLEAN_MERGE:-0}" = "1" ] && [ -n "${MERGED_DIR:-}" ] && [ -d "$MERGED_DIR" ]; then
    log "CLEAN_MERGE=1 → 删除合并模型: $MERGED_DIR (adapter 与报告保留, 可重新 merge 复原)"
    rm -rf "$MERGED_DIR"
  fi
}

# ---------------------------------------------------------------- 入口

# ---- BG=1: 自我转后台 (nohup + PID 文件 + 日志重定向), 断 SSH 不死 ----
# 正式训练 3~6 小时必用; 探针 2~3 分钟可前台直跑。实现: 递归调用自身一次,
# 子进程经 nohup 脱离终端, 父进程打印三件套用法后立即退出。
if [ "${BG:-0}" = "1" ] && [ -z "${BG_CHILD:-}" ]; then
  BG_TS=$(date +"%Y%m%d_%H%M%S")
  # nohup 日志与 PID 统一收进 nohup/ 目录，不散落仓库根
  BG_DIR="$BASE_DIR/nohup"
  mkdir -p "$BG_DIR"
  BG_LOG="$BG_DIR/nohup_${STEP}_${BG_TS}.log"
  BG_PID="$BG_DIR/train_${BG_TS}.pid"
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

# 全量超参快照（对齐 run_grpo_train.sh 风格）：一屏看清本次跑的配置，
# 回填实验登记表 / 排查「这枪和上枪差在哪」时不用翻代码猜默认值
log "---------------- 超参快照 ----------------"
log "  模型/数据 : MODEL_NAME=$MODEL_NAME"
log "              MODEL_DIR=$MODEL_DIR  DATA_DIR=$DATA_DIR"
log "              MIXED_DIR=$MIXED_DIR"
# DOMAIN_FILE / ANCHOR_FILE 仅 mix 步真正消费；traintest 复用产物时不打，防误导
if [ "$STEP" = "mix" ] || [ "$STEP" = "all" ]; then
  log "              DOMAIN_FILE=$DOMAIN_FILE"
  log "              ANCHOR_FILE=$ANCHOR_FILE"
  log "  mix参数   : --anchor_ratio 0.1 --val_ratio 0.1 --seed 42  ANCHOR_MAX_CHARS=${ANCHOR_MAX_CHARS:-1200}"
fi
log "  训练核心  : SFT_POC_STAGE=$SFT_POC_STAGE  EPOCHS=$EPOCHS  LR=$LR"
# 三级 batch 拆解：单卡 batch × 卡数 × 梯度累积 = 有效 batch（参数更新频率的口径）
#   每步训练消耗 PER_DEV_BATCH×NPROC 条样本，累计 ACCUM 个 micro-batch 才 optimizer.step() 一次
_NPROC_SNAP="${NPROC:-}"
_ACCUM_SNAP=$(( 8 / (${PER_DEV_BATCH:-2} * ${_NPROC_SNAP:-1}) ))
[ "${_ACCUM_SNAP:-0}" -lt 1 ] && _ACCUM_SNAP=$(( 8 / ${_NPROC_SNAP:-1} ))
log "              MAX_LEN=${MAX_LEN:-2560}"
log "              per_device_train_batch_size=${PER_DEV_BATCH:-2} × NPROC=${_NPROC_SNAP:-自动(单卡为1)} × gradient_accumulation_steps=$_ACCUM_SNAP → 有效batch=8 (每 $_ACCUM_SNAP 个 micro-batch、共 $(( 8 )) 条样本更新一次参数)"
log "              per_device_eval_batch_size=${PER_DEV_EVAL_BATCH:-1}  warmup_ratio=0.05  scheduler=cosine  logging_steps=1  eval_steps=5"
log "              UNFREEZE_PROTOCOL_TOKENS=${UNFREEZE_PROTOCOL_TOKENS:-1}  THINK_PREFIX=${THINK_PREFIX:-0}"
log "              TRAINER_BACKEND=${TRAINER_BACKEND:-auto}  NPROC=${NPROC:-自动}"
log "              CANARY_EVERY=${CANARY_EVERY:-30}  CANARY_N=${CANARY_N:-3}  CANARY_MAX_NEW=${CANARY_MAX_NEW:-400}"
# mix 比例只在上面 mix 分支里打印（traintest 复用产物时，比例来自当时的混合，不在本脚本手里）
log "  混合/推理 : INFER_BACKEND=${INFER_BACKEND:-auto}  INFER_GPU=${INFER_GPU:-0}  INFER_DATA=${INFER_DATA:-默认val}"
log "  wandb     : SYNC_WANDB=${SYNC_WANDB:-0}  WANDB_PROJECT=${WANDB_PROJECT:-sentiment-sft}"
log "              WANDB_RUN_NAME=${WANDB_RUN_NAME:-probe_${TS}${TAG:+_${TAG}}}"
log "  输出      : RUN_DIR=$RUN_DIR"
log "              FORCE=${FORCE:-0}  BG=${BG:-0}  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
log "------------------------------------------"

case "$STEP" in
  download) step_download ;;
  mix)      step_mix ;;
  train)    step_train ;;
  merge)    step_merge ;;
  test)     step_merge; step_infer; step_eval ;;
  traintest) check_reuse_assets; step_train; step_merge; step_infer; step_eval ;;
  all)      step_download; step_mix; step_train; step_merge; step_infer; step_eval ;;
  *) echo "未知步骤: $STEP (可选 download|mix|train|merge|test|traintest|all)"; exit 2 ;;
esac

log "本轮探针结束 · 总日志: $RUN_LOG"
