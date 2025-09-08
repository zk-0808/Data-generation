import yaml
import json
import random
from typing import List, Dict
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import certifi
import os

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# 加载模型
augmenter = pipeline(
    "text2text-generation",
    model="uer/t5-base-chinese-cluecorpussmall",
    trust_remote_code=True,
    device_map="auto"
)

# --------------------------
# 参数配置
# --------------------------
FIELD_POOL = [
    ("客户", "客户等级", "INT"),
    ("客户", "客户状态", "STRING"),
    ("订单", "订单金额", "FLOAT"),
    ("订单", "支付方式", "STRING"),
    ("商品", "库存数量", "INT")
]

RELATION_POOL = [
    ("客户", "订单"),
    ("订单", "商品"),
    ("客户", "地址")
]

# --------------------------
# 意图解析函数
# --------------------------
def parse_instruction(instr: str) -> List[Dict]:
    intents = []
    if "添加" in instr or "增加" in instr:
        for concept, field, dtype in FIELD_POOL:
            if field in instr and concept in instr:
                intents.append({
                    "intentType": "ADD_COLUMN",
                    "targetConceptName": concept,
                    "props": {"name": field, "stdSqlType": dtype}
                })
    if "删除字段" in instr or "移除字段" in instr:
        for concept, field, _ in FIELD_POOL:
            if field in instr and concept in instr:
                intents.append({
                    "intentType": "DELETE_COLUMN",
                    "targetConceptName": concept,
                    "props": {"name": field}
                })
    if "更改字段类型" in instr:
        for concept, field, _ in FIELD_POOL:
            if field in instr and concept in instr:
                intents.append({
                    "intentType": "UPDATE_COLUMN_TYPE",
                    "targetConceptName": concept,
                    "props": {"name": field, "stdSqlType": "STRING"}  # 假设更改为 STRING
                })
    if "改名为" in instr or "重命名为" in instr:
        for concept in ["客户", "订单", "商品"]:
            if concept in instr:
                intents.append({
                    "intentType": "RENAME_ENTITY",
                    "targetConceptName": concept,
                    "props": {"newName": f"{concept}_信息"}
                })
    if "建立关联" in instr or "添加关联" in instr:
        for from_entity, to_entity in RELATION_POOL:
            if from_entity in instr and to_entity in instr:
                intents.append({
                    "intentType": "ADD_RELATION",
                    "targetConceptName": from_entity,
                    "props": {"from": from_entity, "to": to_entity}
                })
    if "删除关联" in instr or "移除关联" in instr:
        for from_entity, to_entity in RELATION_POOL:
            if from_entity in instr and to_entity in instr:
                intents.append({
                    "intentType": "DELETE_RELATION",
                    "targetConceptName": from_entity,
                    "props": {"from": from_entity, "to": to_entity}
                })
    if "启用逻辑删除" in instr:
        for concept in ["客户", "订单"]:
            if concept in instr:
                intents.append({
                    "intentType": "UPDATE_ENTITY",
                    "targetConceptName": concept,
                    "props": {"useLogicalDelete": True}
                })
    return intents


# --------------------------
# 增广函数
# --------------------------
def augment_instruction(original: str, num_augments: int = 5) -> List[str]:
    templates = [
        f"请帮我{original}",
        f"系统需要{original}",
        f"请用口语方式表达：{original}",
        f"换一种说法表达：{original}",
        f"将以下语句改写为更通俗的说法：{original}",
        f"改写为行业术语表达：{original}",
        f"用命令式表达：{original}",
        f"改成两个短句表达：{original}",
        f"从系统角度陈述：{original}",
        f"给开发人员的指令是：{original}"
    ]
    results = []
    for prompt in templates[:num_augments]:
        try:
            out = augmenter(prompt, do_sample=True, temperature=0.7, max_new_tokens=60)[0]["generated_text"]
            if out.strip() and out.strip() != original:
                results.append(out.strip())
        except:
            continue
    return results

# --------------------------
# 对抗样本生成
# --------------------------
def generate_adversarial_samples(intent: List[Dict]) -> List[Dict]:
    samples = []
    for i in intent:
        if i["intentType"] == "ADD_COLUMN":
            wrong = i.copy()
            wrong["props"]["stdSqlType"] = "FLOAT" if wrong["props"]["stdSqlType"] == "INT" else "INT"
            samples.append({
                "instruction": f"为{i['targetConceptName']}表添加错误类型的字段 {i['props']['name']}",
                "intent": [wrong],
                "label": "对抗样本-类型错误"
            })
        if i["intentType"] == "DELETE_RELATION":
            wrong = i.copy()
            wrong["intentType"] = "ADD_RELATION"
            samples.append({
                "instruction": f"添加 {i['props']['from']} 与 {i['props']['to']} 的关联关系",
                "intent": [wrong],
                "label": "对抗样本-关系方向错误"
            })
    samples.extend([
        {"instruction": "添加个字段吧", "intent": [], "label": "负样本-字段名不明"},
        {"instruction": "改一改那张表", "intent": [], "label": "负样本-意图不明确"},
        {"instruction": "客护表加个东东", "intent": [], "label": "负样本-拼写错误"},
        {"instruction": "搞一个字段等级", "intent": [], "label": "负样本-语序混乱"}
    ])
    return samples

# --------------------------
# 去重函数
# --------------------------
def filter_duplicates(samples: List[Dict], threshold: float = 0.9) -> List[Dict]:
    if len(samples) <= 1:
        return samples
    instructions = [s["instruction"] for s in samples]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(instructions)
    sim = cosine_similarity(matrix)
    keep = []
    for i in range(len(samples)):
        if all(sim[i][j] < threshold for j in keep):
            keep.append(i)
    return [samples[i] for i in keep]

# --------------------------
# 保存函数
# --------------------------
def save_training_data(samples: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            json.dump({
                "instruction": s["instruction"],
                "intent": s.get("intent", []),
                "label": s.get("label", "正常样本")
            }, f, ensure_ascii=False)
            f.write("\n")

# --------------------------
# 主流程
# --------------------------
if __name__ == "__main__":
    base_instructions = [
        "为客户表添加客户等级字段，类型为整型",
        "订单表启用逻辑删除",
        "移除订单与商品之间的关联",
        "更改客户状态字段的数据类型",
        "重命名客户表为客户信息",
        "为订单表增加支付方式字段，类型为字符串",
        "为商品表添加库存数量字段，整型",
        "建立客户与地址的关联关系",
        "删除订单金额字段",
        "重命名订单表为历史订单表"
    ]

    all_samples = []
    for base in base_instructions * 20:  # 扩展样本规模（10条 × 20倍）
        intent = parse_instruction(base)
        if not intent:
            continue
        all_samples.append({"instruction": base, "intent": intent})
        # 增广样本
        aug = augment_instruction(base, num_augments=6)
        all_samples.extend([{"instruction": a, "intent": intent} for a in aug])
        # 对抗样本
        adv = generate_adversarial_samples(intent)
        all_samples.extend(adv)

    final_samples = filter_duplicates(all_samples)
    save_training_data(final_samples, "augmented_intent_dataset.jsonl")
    print(f"✅ 样本生成完成，共生成样本数：{len(final_samples)} 条")
