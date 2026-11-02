
import csv
import random
import torch

from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. 基本设置
# ==========================================

MODEL_PATH = "./models/Qwen3-0.6B"
ADAPTER_PATH = "./outputs/data_efficiency/train_6/final_adapter"
DATA_PATH = Path("data/processed/validation.csv")
OUTPUT_PATH = Path("results/data_efficiency_6_validation.csv")
LABELS = ["positive", "neutral", "negative"]

SAMPLES_PER_CLASS = 20
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}", flush=True)


# ==========================================
# 2. 加载 Qwen 模型
# ==========================================

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("\n========== TOKENIZER CHECK ==========")

for label in LABELS:
    ids = tokenizer.encode(
        label,
        add_special_tokens=False
    )

    print(
        f"{label}: Token IDs = {ids}, "
        f"Token count = {len(ids)}"
    )

print("=====================================\n")

print("Loading Qwen model...", flush=True)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
).to(DEVICE)

model.eval()
print("Model loaded successfully!\n", flush=True)


# ==========================================
# 3. 从验证集随机抽取 60 条
# ==========================================

print("Loading validation data...", flush=True)

with open(
    DATA_PATH,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    all_rows = list(csv.DictReader(f))

# 使用完整验证集，不进行抽样
rows = all_rows

print(f"Selected {len(rows)} samples.")

print("True label distribution:")

for label in LABELS:
    count = sum(
        row["label"].strip().lower() == label
        for row in rows
    )
    print(f"  {label}: {count}")

print()


# ==========================================
# 4. 新的提示词
# ==========================================

SYSTEM_PROMPT = (
    "Classify the sentiment as positive, neutral, or negative. "
    "Reply with one word."
)


# ==========================================
# 5. 标签似然评分
# ==========================================

def classify(text):

    messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": text
    }
]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False
    )["input_ids"]

    scores = {}

    # 分别计算三个候选答案的概率
    for label in LABELS:

        # 不添加额外空格
        answer_ids = tokenizer(
            label,
            add_special_tokens=False
        )["input_ids"]

        all_ids = prompt_ids + answer_ids

        input_ids = torch.tensor(
            [all_ids],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.inference_mode():

            outputs = model(
                input_ids=input_ids
            )

            logits = outputs.logits

        log_probs = torch.log_softmax(
            logits.float(),
            dim=-1
        )

        score = 0.0

        for j, token_id in enumerate(answer_ids):

            position = len(prompt_ids) + j - 1

            score += log_probs[
                0,
                position,
                token_id
            ].item()

        scores[label] = score

    prediction = max(
        scores,
        key=scores.get
    )

    return prediction, scores


# ==========================================
# 6. 开始测试
# ==========================================

true_labels = []
predicted_labels = []
saved_results = []

print("========== EVALUATION START ==========\n")

for i, row in enumerate(rows, start=1):

    text = row["text"]

    true_label = row["label"].strip().lower()

    prediction, scores = classify(text)

    true_labels.append(true_label)
    predicted_labels.append(prediction)

    saved_results.append({
        "text": text,
        "true_label": true_label,
        "predicted_label": prediction,
        "positive_score": scores["positive"],
        "neutral_score": scores["neutral"],
        "negative_score": scores["negative"]
    })

    print(
        f"{i}/{len(rows)} | "
        f"True: {true_label} | "
        f"Predicted: {prediction}",
        flush=True
    )


# ==========================================
# 7. 计算准确率和 Macro F1
# ==========================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    labels=LABELS,
    average="macro",
    zero_division=0
)

print("\n========== RESULTS ==========")

print(f"Total samples: {len(rows)}")

print(f"Accuracy: {accuracy:.2%}")

print(f"Macro F1: {macro_f1:.4f}")


# ==========================================
# 8. 预测类别分布
# ==========================================

print("\nPrediction distribution:")

prediction_counts = Counter(predicted_labels)

for label in LABELS:

    print(
        f"{label}: {prediction_counts[label]}"
    )


# ==========================================
# 9. 分类报告
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        zero_division=0
    )
)


# ==========================================
# 10. 混淆矩阵
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=LABELS
)

print("Rows = True labels")
print("Columns = Predicted labels")
print("Label order:", LABELS)

print(matrix)


# ==========================================
# 11. 保存结果到 CSV
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "text",
            "true_label",
            "predicted_label",
            "positive_score",
            "neutral_score",
            "negative_score"
        ]
    )

    writer.writeheader()
    writer.writerows(saved_results)

print(
    f"\nResults saved to: {OUTPUT_PATH}"
)

# 12. 保存 Evaluation Report
# ==========================================

REPORT_PATH = Path("results/reports/report_6.txt")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

report_text = f"""
========== RESULTS ==========
Total samples: {len(rows)}
Accuracy: {accuracy:.2%}
Macro F1: {macro_f1:.4f}

Prediction distribution:
positive: {prediction_counts["positive"]}
neutral: {prediction_counts["neutral"]}
negative: {prediction_counts["negative"]}

========== CLASSIFICATION REPORT ==========
{classification_report(
    true_labels,
    predicted_labels,
    labels=LABELS,
    zero_division=0
)}

========== CONFUSION MATRIX ==========
Rows = True labels
Columns = Predicted labels
Label order: {LABELS}
{matrix}
"""

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:
    f.write(report_text)

print(f"Report saved to: {REPORT_PATH}")

print("\nEvaluation completed!")
