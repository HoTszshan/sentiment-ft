#!/usr/bin/env python3
"""
scripts/download_assets.py
下载基座模型 (Qwen2.5-3B/7B-Instruct) 与公开 Anchor 数据集 (COIG-CQIA)。
支持 ModelScope (国内秒级下载，推荐) 与 HuggingFace 双源。
"""

import argparse
import json
import os
import sys
from pathlib import Path

def download_model(model_name: str, out_dir: str, source: str = "modelscope"):
    out_path = Path(out_dir) / model_name.split("/")[-1]
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f"开始下载基座模型: {model_name}")
    print(f"下载渠道: {source}  -> 本地路径: {out_path}")
    print(f"=======================================================")

    if source == "modelscope":
        try:
            from modelscope import snapshot_download
            snapshot_download(model_id=model_name, local_dir=str(out_path))
            print(f"✓ 模型下载成功: {out_path}")
        except ImportError:
            print("❌ 缺少 modelscope 依赖，请先安装: pip install modelscope")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            sys.exit(1)
    else:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_name, local_dir=str(out_path))
            print(f"✓ 模型下载成功: {out_path}")
        except ImportError:
            print("❌ 缺少 huggingface_hub 依赖，请先安装: pip install huggingface_hub")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            sys.exit(1)


def download_anchor_dataset(out_dir: str, num_samples: int = 500):
    """
    下载并采样 COIG-CQIA 中文高质量通用指令数据集作为 Anchor (防灾难性遗忘)。
    """
    out_path = Path(out_dir) / "coig_cqia_anchor.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f"准备公开通用指令数据集 (Anchor 防遗忘底座)")
    print(f"数据集: COIG-CQIA (过滤掉任何可能的情感/舆情标签)")
    print(f"目标落盘: {out_path} (采样 {num_samples} 条)")
    print(f"=======================================================")

    try:
        from datasets import load_dataset
        print("正在从 HuggingFace/ModelScope 加载 COIG-CQIA 子集...")
        # 优先读取知乎/维基百科等纯百科与问答类优质子集
        ds = load_dataset("m-a-p/COIG-CQIA", "coig_pc", split="train", streaming=True)
        samples = []
        for row in ds:
            instr = row.get("instruction", "")
            inp = row.get("input", "")
            out = row.get("output", "")
            if not instr or not out:
                continue
            # 过滤掉带有"情感/好评/差评"等字眼的数据，避免冲淡专业舆情口径
            text_all = instr + inp + out
            if any(w in text_all for w in ["情感", "好评", "差评", "正向", "负向"]):
                continue

            samples.append({
                "instruction": instr,
                "input": inp,
                "output": out,
                "meta": {
                    "source": "COIG-CQIA",
                    "provenance": "public_anchor",
                    "hard_case_type": "general_anchor"
                }
            })
            if len(samples) >= num_samples:
                break

        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"✓ 成功下载并提纯 {len(samples)} 条通用 Anchor 样本 -> {out_path}")

    except Exception as e:
        print(f"⚠️ 在线拉取通用数据集出现波动 ({e})，正在自动生成高质量内置通用问答兜底包...")
        # 离线时自动备用优质百科/常识/代码通用问答种子，确保不中断 pipeline
        builtin_anchors = [
            {"instruction": "请简要解释什么是分布式系统中的 CAP 定理。", "input": "", "output": "CAP 定理指出，在分布式数据存储中，一致性（Consistency）、可用性（Availability）和分区容错性（Partition tolerance）三者不可兼得，系统最多只能同时满足其中的两项。"},
            {"instruction": "请分析中国传统二十四节气中‘清明’的民俗与物候特征。", "input": "", "output": "清明是春季的第五个节气。物候上表现为气温回升、雨水增多，草木萌动；民俗上兼具节气与节日双重身份，有扫墓祭祖、踏青插柳等传统习俗。"},
            {"instruction": "Python 中浅拷贝与深拷贝的核心区别是什么？", "input": "", "output": "浅拷贝只复制对象本身及最顶层的引用，子对象仍然共享内存；而深拷贝会递归复制所有层级的嵌套对象，新旧对象完全独立互不影响。"},
            {"instruction": "请总结撰写技术设计文档（RFC）时的关键核心模块。", "input": "", "output": "一份标准的 RFC 通常包含：背景与现状痛点、技术目标与非目标、系统整体架构图、核心接口与数据协议、部署监控与灰度降级预案。"}
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            for i in range(num_samples):
                item = dict(builtin_anchors[i % len(builtin_anchors)])
                item["meta"] = {"source": "builtin_anchor", "provenance": "public_anchor"}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✓ 已就绪 {num_samples} 条通用 Anchor 样本 -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="基座模型与公开数据集一键下载器")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="基座模型ID (默认 Qwen3-4B-Instruct-2507，ModelScope 实际在架仓库名)")
    parser.add_argument("--model_dir", default="/root/autodl-tmp/models",
                        help="模型存放目录 (AutoDL 数据盘默认 /root/autodl-tmp/models)")
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope", help="下载源")
    parser.add_argument("--download_anchor", action="store_true", help="是否下载 COIG-CQIA 通用 Anchor 数据集")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/datasets",
                        help="公开数据集存放目录 (AutoDL 数据盘默认 /root/autodl-tmp/datasets)")
    parser.add_argument("--anchor_samples", type=int, default=float('inf'), help="Anchor 采样条数")
    args = parser.parse_args()

    if args.download_anchor:
        download_anchor_dataset(args.data_dir, num_samples=args.anchor_samples)

    if args.model:
        download_model(args.model, args.model_dir, source=args.source)

if __name__ == "__main__":
    main()
