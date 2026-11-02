
import csv
import torch

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


# ==========================================
# 1. 基本设置
# ==========================================

MODEL_PATH = "./models/Qwen3-0.6B"
TRAIN_PATH = Path("data/processed/train.csv")
OUTPUT_DIR = "./outputs/qwen3-financial-lora"

SYSTEM_PROMPT = (
    "Classify the sentiment as positive, neutral, or negative. "
    "Reply with one word."
)

MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    raise RuntimeError("需要 CUDA GPU 才能运行本训练配置。")

print("GPU:", torch.cuda.get_device_name(0), flush=True)


# ==========================================
# 2. 加载 tokenizer 和模型
# ==========================================

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

print("Model loaded.", flush=True)


# ==========================================
# 3. 添加 LoRA
# ==========================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(model, lora_config)

model.enable_input_require_grads()
model.print_trainable_parameters()


# ==========================================
# 4. 读取训练数据
# ==========================================

print("Loading training data...", flush=True)

with open(
    TRAIN_PATH,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

print(f"Training samples: {len(rows)}", flush=True)

LABELS = {"positive", "neutral", "negative"}

for row in rows:
    if row["label"].strip().lower() not in LABELS:
        raise ValueError(f"Unknown label: {row['label']}")


# ==========================================
# 5. 构造训练样本
# ==========================================

class FinancialDataset(torch.utils.data.Dataset):

    def __init__(self, rows):
        self.samples = []

        for row in rows:
            text = row["text"]
            label = row["label"].strip().lower()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            answer_ids = tokenizer(
                label,
                add_special_tokens=False,
            )["input_ids"]

            # 只让模型学习预测正确的分类标签。
            # 不让它重复学习整段提示词。
            input_ids = prompt_ids + answer_ids + [
                tokenizer.eos_token_id
            ]

            labels = (
                [-100] * len(prompt_ids)
                + answer_ids
                + [tokenizer.eos_token_id]
            )

            if len(input_ids) > MAX_LENGTH:
                # 保留分类标签和结束符
                excess = len(input_ids) - MAX_LENGTH
                input_ids = input_ids[excess:]
                labels = labels[excess:]

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


train_dataset = FinancialDataset(rows)

print("Dataset prepared.", flush=True)


# ==========================================
# 6. 数据整理器
# ==========================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    padding=True,
    label_pad_token_id=-100,
)


# ==========================================
# 7. 训练参数
# ==========================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
  warmup_steps=100,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    optim="adamw_torch",
    seed=42,
)


# ==========================================
# 8. 开始训练
# ==========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("\n========== TRAINING START ==========\n", flush=True)

trainer.train()


# ==========================================
# 9. 保存 LoRA adapter
# ==========================================

final_path = Path(OUTPUT_DIR) / "final_adapter"
final_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print("\n========== TRAINING COMPLETE ==========")
print("LoRA adapter saved to:", final_path)
