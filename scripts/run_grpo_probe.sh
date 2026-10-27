#!/usr/bin/env bash
# ==============================================================================
# 舆情负面精判 · GRPO rollout 探针启动脚本 (k 分流筛题; 与 run_grpo_train.sh 同产物布局)
#
# 用法 (任意目录可执行, 脚本内自行 cd 仓库根):
#   ./scripts/run_grpo_probe.sh                        # 默认: 裸合并模型, G=4
#   ./scripts/run_grpo_probe.sh --tag=sft_kdist        # run 目录追加 tag 后缀
#   ./scripts/run_grpo_probe.sh --adapter models/Qwen3-4B-Instruct-SFT   # 挂 adapter 对照
#   DRY=1 ./scripts/run_grpo_probe.sh                  # --dry_run 体检 (零 GPU)
#
# 产物布局 (与 run_grpo_train.sh 对齐: 一个 run 目录 = 一次实验, 拷走即归档):
#   $OUTPUT_DIR/run_<TS>[_<TAG>]/
#     ├── grpo_probe_kstats.json        k 直方图 + 训练池统计
#     ├── grpo_probe_kstats.rows.jsonl  逐题明细
#     └── logs/grpo_probe.log           控制台全量日志 (tee)
#
# 环境变量:
#   DATA         默认 data/grpo_prompts_probe.jsonl
#   BASE_MODEL   默认 models/Qwen3-4B-Instruct-SFT (合并 SFT 后的完整模型目录)
#   OUTPUT_DIR   默认 /root/autodl-tmp/models/grpo_senti (AutoDL, 与训练脚本同目录);
#                本机自动退回 <仓库>/output/grpo
#   G            同题采样份数 (默认 4)
#   DRY          =1 只跑 --dry_run 体检 (无 GPU 可跑)
#
# 说明: 探针 50 题单卡几分钟跑完, 本脚本固定单卡 (CUDA_VISIBLE_DEVICES=0);
#       1k 探针再考虑对半分双卡, 届时直接跑两个进程各自 --limit 切分即可。
# ==============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"
echo "工作目录: $BASE_DIR"

DATA="${DATA:-data/grpo_prompts_probe.jsonl}"
BASE_MODEL="${BASE_MODEL:-models/Qwen3-4B-Instruct-SFT}"
# 与 run_grpo_train.sh 同款回退: 环境变量 > AutoDL 数据盘 > 仓库本地
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [ -n "$OUTPUT_DIR" ]; then :;
elif [ -d /root/autodl-tmp ]; then OUTPUT_DIR=/root/autodl-tmp/models/grpo_senti;
else OUTPUT_DIR="$BASE_DIR/output/grpo"; fi
G="${G:-4}"

TS="$(date +%Y%m%d_%H%M%S)"

# 参数解析: --tag=XXX 进目录名, 其余原样透传给 run_grpo_probe.py
STEP_TAG=""
PROBE_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --tag=*) STEP_TAG="${arg#--tag=}" ;;
    *) PROBE_ARGS="$PROBE_ARGS $arg" ;;
  esac
done
PROBE_ARGS="${PROBE_ARGS# }"
# 裸 --tag 会漏进透传列表被 python 报错, 这里提示正确用法
case " ${PROBE_ARGS} " in *" --tag "*) echo "❌ 裸 --tag 不支持, 请用 --tag=<名称>"; exit 2 ;; esac

# run 目录: 统计/明细/日志同收一处 (探针无训练产物, out 直接放 run 目录顶层)
# 防呆: OUTPUT_DIR 指到仓库根 / HOME / / 时, run_<TS> 会直接铺在那一层
OUTPUT_DIR_REAL="$(readlink -f "$OUTPUT_DIR" 2>/dev/null || echo "$OUTPUT_DIR")"
case "$OUTPUT_DIR_REAL" in
  "$BASE_DIR"|"/"|"${HOME:-/nonexistent}") echo "❌ OUTPUT_DIR 不能是仓库根/HOME// ($OUTPUT_DIR)，run_<TS> 会直接铺在那里"; exit 2 ;;
esac
RUN_DIR="$OUTPUT_DIR/run_${TS}${STEP_TAG:+_${STEP_TAG}}"
LOG_DIR="$RUN_DIR/logs"
OUT_BASE="$RUN_DIR/grpo_probe_kstats.json"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/grpo_probe.log"
log() { echo "[PROBE] $*" | tee -a "$RUN_LOG"; }

# 单卡固定: 探针规模小, 双卡并行不值当 (见文件头说明)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---------------------------------------------------------------- dry_run 分支
if [ "${DRY:-0}" = "1" ]; then
  log "DRY=1 → 零 GPU 体检 (gold 反构采样结果, 只验数据+打分+分桶链路)"
  python3 scripts/run_grpo_probe.py --data "$DATA" --dry_run \
    --out "$OUT_BASE" $PROBE_ARGS 2>&1 | tee -a "$RUN_LOG"
  log "体检产物: $RUN_DIR"
  exit 0
fi

# ---------------------------------------------------------------- 起点体检
if [ ! -f "$BASE_MODEL/config.json" ]; then
  log "起点缺失: $BASE_MODEL 不是完整模型目录"
  log "先合并: python3 scripts/merge_sft_adapter.py \\"
  log "        --base models/Qwen3-4B-Instruct --adapter output/run_<时间戳>/sft_lora \\"
  log "        --out models/Qwen3-4B-Instruct-SFT"
  exit 1
fi
if [ -f "$BASE_MODEL/merge_manifest.json" ]; then
  log "策略起点: 已合并 SFT 的模型 $BASE_MODEL"
else
  log "策略起点: $BASE_MODEL（无 merge_manifest.json → 原厂基座裸跑；k 分布口径注明）"
fi

# ---------------------------------------------------------------- 启动 (前台直跑, 探针几分钟)
log "run 目录: $RUN_DIR"
log "数据: $DATA  基座: $BASE_MODEL  G=$G"
# shellcheck disable=SC2086
python3 scripts/run_grpo_probe.py \
  --data "$DATA" \
  --base_model "$BASE_MODEL" \
  --g "$G" \
  --out "$OUT_BASE" \
  $PROBE_ARGS 2>&1 | tee -a "$RUN_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && { log "❌ 探针失败 (exit=$rc)，详见 $RUN_LOG"; exit "$rc"; }

log "统计: $OUT_BASE"
log "逐题: ${OUT_BASE%.json}.rows.jsonl"
log "日志: $RUN_LOG"
log "----------------------------------------------"
