"""
Vaidya — Phase 2: QLoRA configuration constants.
Imported by train.py. All values are data-driven from Phase 1.
"""

# ── Model ────────────────────────────────────────────────────
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ── Data (from Phase 1 token_stats.json) ─────────────────────
MAX_SEQ_LENGTH = 640      # p95 = 574, rounded to nearest 128

# ── 4-bit quantization (BitsAndBytes NF4) ────────────────────
BNB = dict(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",   # compute in bf16 even when storing in 4-bit
    bnb_4bit_use_double_quant=True,      # 2nd quantization of quant constants → ~0.4 bpw saving
    bnb_4bit_quant_type="nf4",           # Normal Float 4 — best for normally distributed weights
)

# ── LoRA ─────────────────────────────────────────────────────
LORA = dict(
    r=16,
    lora_alpha=32,          # scaling = alpha/r = 2 → standard starting point
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Training (tuned for P100 16 GB, max_seq_length=640) ──────
# Effective batch = per_device_train_batch_size * gradient_accumulation_steps = 32
# If OOM on P100: reduce BATCH_SIZE to 1 and set GRAD_ACCUM = 32
TRAINING = dict(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    packing=True,           # sequence packing via TRL ConstantLengthDataset
    dataset_text_field="text",
)

# ── Artifact store ────────────────────────────────────────────
S3_BUCKET      = "s3://vaidya-artifacts"
CHECKPOINT_DIR = "./checkpoints"
