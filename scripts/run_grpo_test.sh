#!/usr/bin/env bash
# ==============================================================================
# 舆情负面精判 · GRPO 终评一键脚本 (merge → 推理 → 模式 B 打分)
#
# 与 run_sft_e2e.sh test 的区别: 起点不是「基座 + SFT adapter」, 而是
# 「SFT 合并模型(=GRPO 训练起点) + GRPO adapter」—— RL LoRA 叠 SFT 产物,
# 不需要再借 MODEL_NAME 的 basename 绕路。
#
# 产物落位: **不新建 run 目录** —— 预测/报告/日志直接挂到 adapter 所属 run 目录
# (run_<TS>_<TAG>/) 下, 与训练产物同目录归档:
#   run_<TS>_<TAG>/
#     ├── grpo_lora/                        # 训练产物 (已有)
#     ├── logs/grpo_test_<TS>.log           # 终评日志
#     ├── pred_<数据名>.jsonl               # 预测
#     └── report_<数据名>.json              # 模式 B 评测报告
#
# 用法 (在仓库根目录):
#   ./scripts/run_grpo_test.sh                                   # 默认 gold500_test
#   ./scripts/run_grpo_test.sh data/gold500_test.jsonl           # 位置参数传考卷
#   ./scripts/run_grpo_test.sh --tag=G1_final data/gold500_test.jsonl
#   ADAPTER_DIR=... MODEL=... ./scripts/run_grpo_test.sh         # 多枪对照显式指定
#
# 环境变量 (与 run_grpo_train.sh 同名同义):
#   ADAPTER_DIR    GRPO adapter 目录
#                  (默认: 最新 run_*/grpo_lora; 不存在 latest 指针文件——无脚本写入该文件)
#   MODEL          GRPO 训练起点 = SFT 合并模型目录
#                  (默认 models/Qwen3-4B-Instruct-SFT; 必须与训练该 adapter 时的起点一致)
#   OUTPUT_DIR     默认 /root/autodl-tmp/models/grpo_senti (与 run_grpo_train.sh 同源)
#   INFER_BACKEND  vllm (默认) | hf
#   INFER_GPU      hf 后端绑卡 (默认 0)
#   MAX_NEW_TOKENS 默认 1024
#   FORCE=1        强制重合并 (默认合并产物已存在则跳过)
#   CLEAN_MERGE=1  评测后删合并模型 (~7.6G; adapter/报告/预测保留, 可重 merge 复原)
#
# 合并产物: $OUTPUT_DIR/merged/<起点名>_<run名>_<adapter名>-bf16 (幂等复用)
# ==============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"
echo "工作目录: $BASE_DIR"

MODEL="${MODEL:-models/Qwen3-4B-Instruct-SFT}"
if [ -n "${OUTPUT_DIR:-}" ]; then :;
elif [ -d /root/autodl-tmp ]; then OUTPUT_DIR=/root/autodl-tmp/models/grpo_senti;
else OUTPUT_DIR="$BASE_DIR/output/grpo"; fi
INFER_BACKEND="${INFER_BACKEND:-vllm}"
INFER_GPU="${INFER_GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
FORCE="${FORCE:-0}"
CLEAN_MERGE="${CLEAN_MERGE:-0}"

TS="$(date +%Y%m%d_%H%M%S)"

# 参数解析: 位置参数 = 待评估数据文件 (可多个? 不, 只取第一个); --tag 保留兼容(仅记日志)
INFER_DATA="${INFER_DATA:-data/gold500_test.jsonl}"
for arg in "$@"; do
  case "$arg" in
    --tag=*) TAG="${arg#--tag=}" ;;
    -*) echo "未知选项: $arg"; exit 2 ;;
    *)  INFER_DATA="$arg" ;;
  esac
done
if [ ! -f "$INFER_DATA" ]; then
  echo "❌ 待评估数据文件不存在: $INFER_DATA"
  exit 1
fi

# ---------------------------------------------------------------- 0. adapter 定位
# 优先级: 显式 ADAPTER_DIR > 最新 run_*/grpo_lora
# 注: 曾设计 latest_grpo_adapter.txt 指针文件, 但全仓库无脚本写入, 恒走兜底, 已删
# 防呆: grpo_lora 空目录 = 训练还在跑或已挂 (adapter 未落盘), 选了它后面 merge 必炸
if [ -z "${ADAPTER_DIR:-}" ]; then
  ADAPTER_DIR=""
fi
adapter_usable() { [ -d "$1" ] && ls "$1"/adapter_model.safetensors >/dev/null 2>&1; }
if ! adapter_usable "$ADAPTER_DIR"; then
  for d in $(ls -dt "$OUTPUT_DIR"/run_*/grpo_lora 2>/dev/null); do
    if adapter_usable "$d"; then ADAPTER_DIR="$d"; break; fi
  done
fi
if ! adapter_usable "$ADAPTER_DIR"; then
  echo "❌ 未找到含 adapter_model.safetensors 的 GRPO adapter"
  echo "   (训练仍在跑? ps -p \$(cat $BASE_DIR/train_grpo_pid.txt); 或设 ADAPTER_DIR=... 显式指定)"
  exit 1
fi

# 产物挂 adapter 所属 run 目录 (adapter 路径形如 <OUTPUT_DIR>/run_<TS>_<TAG>/grpo_lora);
# adapter 不在 run_*/ 下时 (如裸目录测试), 退回 OUTPUT_DIR/test_<TS>_<数据名> 避免污染
ADAPTER_RUN="$(dirname "$ADAPTER_DIR")"
ADAPTER_RUN_NAME="$(basename "$ADAPTER_RUN")"
case "$ADAPTER_RUN_NAME" in
  run_*) RUN_DIR="$ADAPTER_RUN" ;;
  *)     RUN_DIR="$OUTPUT_DIR/test_${TS}_$(basename "$INFER_DATA" .jsonl)"; mkdir -p "$RUN_DIR" ;;
esac
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/grpo_test_${TS}.log"
log() { echo "[GRPO-TEST] $*" | tee -a "$RUN_LOG"; }
log "adapter: $ADAPTER_DIR"
log "产物落位: $RUN_DIR${TAG:+ (tag=$TAG, 仅记日志)}"

# 起点体检: 必须是完整模型目录, 且是 GRPO 训练时的那个起点
if [ ! -f "$MODEL/config.json" ]; then
  log "❌ $MODEL 不是完整模型目录 (缺 config.json) — 应为 SFT 合并模型"
  exit 1
fi
if [ ! -f "$MODEL/merge_manifest.json" ]; then
  log "⚠️ $MODEL 无 merge_manifest.json → 像是原厂基座而非 SFT 合并模型"
  log "   GRPO adapter 应叠在其训练起点上, 确认这是有意的对照"
fi

# ---------------------------------------------------------------- 1. 合并 (fp32 合并+校验, bf16 落盘; 幂等)
# 命名规则: <起点名>_<run名或adapter父目录名>_<adapter名>, 不同起点/不同枪不撞名
MERGED_NAME="$(basename "$MODEL")_${ADAPTER_RUN_NAME}_$(basename "$ADAPTER_DIR")"
MERGED_DIR="$OUTPUT_DIR/merged/$MERGED_NAME"

if [ -d "$MERGED_DIR-bf16" ] && [ -f "$MERGED_DIR-bf16/config.json" ] && [ "$FORCE" != "1" ]; then
  log "合并产物已存在，跳过 merge: $MERGED_DIR-bf16 (FORCE=1 强制重合并)"
else
  log "步骤 1/3 · 合并 GRPO adapter → $MODEL (fp32 合并+校验, bf16 落盘)"
  # [P1 修复] set -euo pipefail 下 merge 管道一非零就会触发 errexit 直接退出，
  # 后面的失败清理 (rm 半成品) 永远不执行 → 先关 -e 再跑管道，事后按 PIPESTATUS 判定
  set +e
  python3 scripts/merge_sft_adapter.py \
    --base "$MODEL" --adapter "$ADAPTER_DIR" --out "$MERGED_DIR" \
    --dtype fp32 2>&1 | tee -a "$RUN_LOG"
  merge_rc=${PIPESTATUS[0]}
  set -e
  if [ "$merge_rc" -ne 0 ]; then
    log "❌ merge 失败 (exit=$merge_rc)，删除半成品并中止"
    rm -rf "$MERGED_DIR"
    exit "$merge_rc"
  fi
  # fp32 合并仅为 logits 校验, 落盘转 bf16 (同 run_sft_e2e.sh 的坑注记:
  # heredoc 不能绑给 tee 管道, 落临时脚本文件执行)
  CAST_SCRIPT="$LOG_DIR/cast_bf16_${TS}.py"
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
                  "note": "GRPO adapter 叠 SFT 起点; fp32 校验通过后 cast bf16 落盘"}
(dst/"merge_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
import shutil; shutil.rmtree(src)
print(f"[OK] bf16 产物: {dst} (fp32 中间件已清理)")
PYEOF
  python3 "$CAST_SCRIPT" "$MERGED_DIR" 2>&1 | tee -a "$RUN_LOG"
  if [ ! -f "$MERGED_DIR-bf16/config.json" ]; then
    log "❌ bf16 转存未产出，中止 (fp32 中间件保留待排查)"
    exit 1
  fi
fi
MERGED_DIR="$MERGED_DIR-bf16"
log "合并产物: $MERGED_DIR"

# ---------------------------------------------------------------- 2. 推理
PRED_NAME="pred_$(basename "$INFER_DATA" .jsonl).jsonl"
log "步骤 2/3 · 推理 $INFER_DATA (后端: $INFER_BACKEND, max_new_tokens=$MAX_NEW_TOKENS)"
if [ "$INFER_BACKEND" = "vllm" ]; then
  python3 scripts/sft_infer.py \
    --backend vllm --base_model "$MERGED_DIR" --adapter_path '' \
    --data_file "$INFER_DATA" --out "$RUN_DIR/$PRED_NAME" \
    --max_new_tokens "$MAX_NEW_TOKENS" 2>&1 | tee -a "$RUN_LOG"
else
  CUDA_VISIBLE_DEVICES="$INFER_GPU" python3 scripts/sft_infer.py \
    --backend hf --base_model "$MERGED_DIR" --adapter_path '' \
    --data_file "$INFER_DATA" --out "$RUN_DIR/$PRED_NAME" \
    --max_new_tokens "$MAX_NEW_TOKENS" 2>&1 | tee -a "$RUN_LOG"
fi

# ---------------------------------------------------------------- 3. 评测 (模式 B)
# 金标定位: 显式 GOLD_FILE > 数据文件本身
# 不再按 stem 反推 data/<stem>.jsonl——传入 /datasets/rl_round1/x.jsonl
# 而 data/x.jsonl 恰好是另一份内容时, 旧逻辑会静默选错金标 (条数相同则整场评测
# 对错题静默错位)。默认金标就是数据文件本身, 需要不同金标时显式传 GOLD_FILE。
GOLD_STEM="$(basename "$PRED_NAME" .jsonl)"; GOLD_STEM="${GOLD_STEM#pred_}"
GOLD_FILE="${GOLD_FILE:-$INFER_DATA}"
log "步骤 3/3 · 评测 gold=$GOLD_FILE  pred=$RUN_DIR/$PRED_NAME"
python3 scripts/evaluate_sft.py \
  --gold_file "$GOLD_FILE" --pred_file "$RUN_DIR/$PRED_NAME" \
  --out "$RUN_DIR/report_${GOLD_STEM}.json" 2>&1 | tee -a "$RUN_LOG"

if [ "$CLEAN_MERGE" = "1" ]; then
  log "CLEAN_MERGE=1 → 删除合并模型: $MERGED_DIR (adapter 与报告保留)"
  rm -rf "$MERGED_DIR"
fi
log "终评结束 · 总日志: $RUN_LOG"
