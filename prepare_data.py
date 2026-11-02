from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 找到原始金融新闻数据
input_path = Path("data/Sentences_50Agree.txt")

# 2. 读取新闻和正确答案
rows = []

with open(input_path, "r", encoding="iso-8859-1") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        text, label = line.rsplit("@", 1)

        rows.append({
            "text": text.strip(),
            "label": label.strip().lower()
        })

df = pd.DataFrame(rows)

# 3. 检查数据
valid_labels = {"positive", "neutral", "negative"}

assert set(df["label"]).issubset(valid_labels), "发现未知标签"

print("Original samples:", len(df))
print("\nOriginal label distribution:")
print(df["label"].value_counts())

# 4. 删除重复新闻
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("\nSamples after removing duplicates:", len(df))

# 5. 第一次划分：70%训练，30%剩余
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

# 6. 第二次划分：15%验证，15%测试
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# 7. 创建保存文件夹
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# 8. 保存三个数据集
train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "validation.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("\nDataset split completed!")
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

print("\nTraining label distribution:")
print(train_df["label"].value_counts())
