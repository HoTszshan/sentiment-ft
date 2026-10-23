#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语料 Schema 与提示词加载 · 纯逻辑 (零外部依赖)

原为 scripts/generate_corpus.py 的内嵌常量与提示词加载函数，
因 scripts/build_eval_set.py 也要用同一份「行业风险二级词表 + 标注 System Prompt」
（避免两处漂移导致评测集与训练集口径不一致）而抽出到 lib/。

真理源: materials/DataPipeline_SOP_v4.md §3.2/§3.3 (IAA 校准流程, kappa >= 0.7)
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 闭集枚举的唯一权威定义在 schema_contract.py（训练/评测/质检三侧同源）。
# 本文件按原名再导出，老代码 `from corpus_schema import VALID_HC` 零改动。
from schema_contract import (  # noqa: E402,F401
    VALID_HARD_CASE,
    VALID_RELEVANCE,
    VALID_RISK,
    VALID_SENTIMENT,
)

# tests/ 根目录（本文件在 tests/lib/ 下，parents[1] == tests/）
ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"

INDUSTRY_SCHEMAS = {
    "automotive": {
        "product": "刹车失灵, 自燃, 断轴, 气囊未弹, 智驾失控, 续航虚标, 车机死机",
        "service": "售后推诿, 交车延期, 拒保",
        "marketing": "虚假宣传, 变相降价, 虚标参数",
        "compliance": "监管约谈, 缺陷召回, 违规上路",
        "governance": "高管失德言论, 组织内斗, 歧视公众",
        "finance": "股价下挫, 财报亏损, 资金链承压",
        "general_negative": "外观吐槽, 价格略贵, 常规抱怨",
    },
    "electronics_3c": {
        "product": "主板烧毁, 过热降频, 屏幕绿线, 电池鼓包, 烧WiFi, 断流",
        "service": "天价维修, 售后推诿, 换新被拒",
        "marketing": "虚假宣传, 虚假跑分, 减配变相收费",
        "compliance": "反垄断诉讼, 缺陷隐瞒, 侵权",
        "governance": "高管离职, 劳资争议",
        "finance": "代工业务巨亏, 市值腰斩",
        "general_negative": "外观难看, 重量偏沉, 常规吐槽",
    },
    "finance_insurance": {
        "product": "理财暴雷, 净值腰斩, 假信托",
        "service": "客服失联, 网点关门, 交易系统瘫痪",
        "marketing": "保本高息虚假宣传, 误导定投",
        "compliance": "非法吸收公众存款罪, 经侦立案侦查, 监管约谈通报",
        "governance": "实控人跑路失联, 高管被抓",
        "finance": "资金链彻底断裂, 无法兑付, 坏账攀升",
        "general_negative": "排队时间长, 费率微高, 正常财报波动",
    },
    "beauty_cosmetics": {
        "product": "严重烂脸, 接触性皮炎, 红肿过敏流脓, 检出违禁成分, 严重搓泥",
        "service": "开箱破损拒赔, 发临期产品, 客服态度傲慢",
        "marketing": "虚假宣传美白, 虚标SPF指数, 双11先涨后降",
        "compliance": "售假, 未备案成分, 监管通报",
        "governance": "高管言论翻车, 辱女营销, 劳资纠纷",
        "finance": "业绩下滑, 减持套现",
        "general_negative": "包装难看, 价格偏贵, 主观不适合",
    },
    "fmcg": {
        "product": "食品安全变质, 异物混入, 菌落总数超标, 异味发酸",
        "service": "退换货受阻, 霸王条款",
        "marketing": "零添加虚假宣传, 概念炒作, 诱导消费",
        "compliance": "食品安全立案处罚, 查封下架, 假冒伪劣",
        "governance": "食品伦理缺失, 员工违规操作曝光",
        "finance": "渠道压货暴雷, 营收骤降",
        "general_negative": "口感一般, 包装简陋, 性价比低",
    },
    "real_estate": {
        "product": "楼体墙面开裂, 地下车库渗水, 承重墙移位, 减配交房",
        "service": "物业暴力催费, 维修拖延, 绿化虚假承诺",
        "marketing": "买房送学位虚假宣传, 规划缩水",
        "compliance": "违规挪用预售资金, 违章建筑拆除, 涉嫌欺诈",
        "governance": "高管失联, 卷款跑路, 拖欠工程款民工讨薪",
        "finance": "百亿债务违约, 债券爆仓, 申请破产重整, 资金链断裂",
        "general_negative": "户型不够方正, 物业费略贵, 周边配套待建",
    },
}

# 行业英文名 → 中文名（【行业风险体系】标题用）
INDUSTRY_CN = {
    "automotive": "汽车",
    "electronics_3c": "电子3C",
    "finance_insurance": "金融保险",
    "beauty_cosmetics": "美妆个护",
    "fmcg": "快消",
    "real_estate": "房地产",
}

# 风险大类英文 Key → 中文语义（渲染行业风险体系列表行）
RISK_KEY_CN = {
    "product": "产品/质量",
    "service": "服务/售后",
    "marketing": "营销/宣传",
    "compliance": "合规/监管",
    "governance": "治理/高管",
    "finance": "财务/经营",
    "general_negative": "一般负面(兜底吐槽，不构成危机)",
}


SUBJECTS = {
    "automotive": ["理想 L9", "蔚来 ET5", "小鹏 X9", "极氪 001", "问界 M7",
                   "小米 SU7", "比亚迪汉", "特斯拉 Model Y", "智己 LS6", "阿维塔 12"],
    "electronics_3c": ["小米 15", "华为 Mate 70", "iPhone 16 Pro", "索尼 A7M4",
                       "联想小新 Pro", "ROG 游戏手机", "vivo X200", "OPPO Find X8",
                       "大疆 Pocket 3", "英特尔 i9-14900K"],
    "finance_insurance": ["招商银行", "中信信托", "中泰基金", "平安证券", "徽商银行",
                          "易方达基金", "广发证券", "浦发银行", "华泰柏瑞", "中国平安"],
    "beauty_cosmetics": ["珀莱雅双抗精华", "修丽可色修", "兰蔻小黑瓶", "SK-II 神仙水",
                         "赫莲娜黑绷带", "欧莱雅注白瓶", "薇诺娜特护霜", "资生堂红腰子"],
    "fmcg": ["三只松鼠每日坚果", "元气森林", "农夫山泉", "伊利金典", "蒙牛特仑苏",
             "统一老坛酸菜", "喜茶轻乳茶", "瑞幸咖啡"],
    "real_estate": ["碧桂园天玺", "融创壹号院", "保利和光尘樾", "龙湖天街",
                    "绿城桃花源", "中海天钻", "金茂府", "万科城市花园"],
}

# ---- 随机化维度表（防止模板化，这是本次的核心改进）----
PLATFORM_VOICES = [
    "微博吐槽（短促、带话题标签感、情绪外露）",
    "黑猫投诉长文（时间线清晰、诉求明确、带单据引用）",
    "小红书种草翻车（第一人称、肤感/体感描述细、配图语气）",
    "懂车帝车友圈（口语化、含具体车型配置与路况）",
    "知乎长文分析（半专业、含数据与原理推测）",
    "抖音评论区（极短、口语、可能有错别字与网络梗）",
    "贴吧老哥发帖（粗粝、同行黑话多）",
    "财经快讯（客观书面、含数字与机构名）",
    "官方通报转述（公文体、含文号与部门名）",
    "企业季报数据播报（纯数据、无情绪）",
    "测评博主实测报告（含仪器读数、对照条件）",
    "维权群公告（号召性、含时间地点）",
]
PERSONAS = ["普通消费者", "资深玩家", "车主", "投资者/股东", "记者", "业内人士",
            "围观群众", "前员工", "代购/经销商", "专业检测人士"]
REGISTERS = ["全口语", "半专业混口语", "纯客观书面", "情绪爆发式"]
LENGTH_HINTS = ["短（30~80字）", "中（80~150字）", "中长（150~250字）", "长（250~500字）"]

HARD_CASES = ["explicit", "fact_implicit", "irony", "metaphor",
              "comparative", "counterfactual", "multi_subject"]

# ---- 难点类型 × 目标极性 兼容矩阵（**唯一权威源**）----
# ⚠️ 2026-10-21 用户裁决：**全面放开**。
#   7 难点 × 3 极性 = 21 格全部允许。理由：
#   1) 真实舆情里「irony 中性」是存在的（正话正说的行业黑话式客观描述）；
#      「fact_implicit 正面」也存在（纯客观参数披露 + 客观陈述好评事实）；
#   2) 之前的限制来自 v3.0 的经验推定，不是语义的硬约束；
#   3) 既然改成「按种子铺开 + 下采样控配比」，就没必要在矩阵层提前砍格。
HC_SENT_ALLOWED: Dict[str, set] = {
    "explicit":       {"negative", "positive", "neutral"},
    "fact_implicit":  {"negative", "positive", "neutral"},
    "irony":          {"negative", "positive", "neutral"},
    "metaphor":       {"negative", "positive", "neutral"},
    "comparative":    {"negative", "positive", "neutral"},
    "counterfactual": {"negative", "positive", "neutral"},
    "multi_subject":  {"negative", "positive", "neutral"},
}

# ---- 难点类型 × 语篇地位 兼容矩阵（**唯一权威源**）----
# 层级语义：
#   本矩阵 = **法典层**（难点类型在语义上允许哪些语篇地位）。
#   sample_specs_v2.LEGAL_RELS = **实测收敛层**（由本矩阵派生，砍掉实测会被
#   打标端判成别的难点的格子，如 explicit×passing → 实测必判 fact_implicit）。
#   两者是「法典允许 ⊇ 实测可用」的包含关系，不是两份互相漂移的副本。
# ⚠️ 2026-10-21：统一为单份权威源，sample_specs_v2 / sample_specs_v3 一律 import 本处，
#    杜绝双份定义漂移（2026-10-21 教训：HC_SENT_ALLOWED 本地副本忘了放开 neutral）。
HC_REL_ALLOWED: Dict[str, set] = {
    "explicit":       {"core", "passing", "foil"},
    "fact_implicit":  {"core", "passing", "foil"},
    "irony":          {"core", "passing", "foil"},
    "metaphor":       {"core", "passing", "foil"},
    "comparative":    {"core", "passing", "foil"},
    "counterfactual": {"core", "passing", "foil"},
    "multi_subject":  {"core", "passing", "foil"},
}

def _extract_code_block(md_text: str, heading: str) -> str:
    """从 markdown 中抽取指定小节下的第一个 ```text 代码块。"""
    idx = md_text.find(heading)
    if idx == -1:
        raise ValueError(f"提示词文件中未找到小节: {heading}")
    seg = md_text[idx:]
    m = re.search(r"```text\n(.*?)\n```", seg, re.DOTALL)
    if not m:
        raise ValueError(f"小节 {heading} 下未找到 ```text 代码块")
    return m.group(1).strip()


def load_prompts() -> Tuple[str, str]:
    """返回 (生成 System Prompt, 标注 System Prompt)"""
    gen_md = (PROMPTS_DIR / "gen_prompt.md").read_text(encoding="utf-8")
    anno_md = (PROMPTS_DIR / "anno_prompt.md").read_text(encoding="utf-8")
    gen_sys = _extract_code_block(gen_md, "## 1. System Prompt")
    anno_sys = _extract_code_block(anno_md, "## 1. System Prompt")
    return gen_sys, anno_sys


def schema_to_inline(ind: str) -> str:
    return "\n".join(f"{k}: [{v}]" for k, v in INDUSTRY_SCHEMAS[ind].items())


# ---- 闭集枚举（质检用；权威定义在 schema_contract.py，此处同源再导出）----

VALID_HC = set(HARD_CASES)
VALID_SENT = set(VALID_SENTIMENT)
VALID_REL = set(VALID_RELEVANCE)
VALID_RISK = set(VALID_RISK)  # noqa: F811  (同名再导出，值同源)

# 自检: 本文件 HARD_CASES 与契约层闭集必须逐字一致 (不一致 = 两处漂移，评测必炸)
assert VALID_HC == VALID_HARD_CASE, "HARD_CASES 与 schema_contract.VALID_HARD_CASE 漂移"
