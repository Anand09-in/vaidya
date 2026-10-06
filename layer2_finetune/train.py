"""
Vaidya — Phase 2: QLoRA fine-tuning on MedMCQA.

Pinned stack: transformers==4.47.0, trl==0.15.2, peft==0.14.0,
              bitsandbytes==0.45.3, accelerate==0.35.0

Run on Kaggle T4 ×2 via kaggle_train.ipynb, or directly:
    python train.py [--run-name run1]
"""

__version__ = "2.12"

import os, sys, time, argparse, logging

# Hide GPU 1 before torch loads — avoids multi-GPU confusion on single-process runs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
import mlflow
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import qlora_config as C
from utils import setup_credentials, push_to_s3, setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run1",
                   help="Name for this training run (used in checkpoint dir and MLflow)")
    p.add_argument("--batch-size", type=int, default=C.TRAINING["per_device_train_batch_size"])
    p.add_argument("--grad-accum", type=int, default=C.TRAINING["gradient_accumulation_steps"])
    p.add_argument("--max-steps", type=int, default=-1, help="Cap training steps (-1 = full epoch)")
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────────────
def load_data():
    if Path("/kaggle/working").exists():
        data_dir = Path("/kaggle/working/data")
    else:
        data_dir = Path(__file__).resolve().parent / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.parquet"

    if not train_path.exists():
        log.info("Data not found — downloading from S3...")
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="data/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    dest = data_dir / key.split("/")[-1]
                    log.info("  downloading %s", key)
                    s3.download_file(bucket, key, str(dest))
        assert train_path.exists(), f"train.parquet missing after S3 download"

    train_ds = Dataset.from_pandas(pd.read_parquet(data_dir / "train.parquet")[["text"]], preserve_index=False)
    val_ds   = Dataset.from_pandas(pd.read_parquet(data_dir / "val.parquet")[["text"]],   preserve_index=False)
    log.info("Data loaded — train: %d  val: %d", len(train_ds), len(val_ds))
    return train_ds, val_ds


# ── Model + tokenizer ────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    cc = torch.cuda.get_device_capability()
    # T4 is CC 7.5 — is_bf16_supported() returns True but BF16 is software-emulated (slow).
    # Only use BF16 on Ampere+ (CC >= 8.0) where it has native hardware support.
    use_bf16_compute = cc[0] >= 8

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16_compute else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    attn_impl = "eager"
    if cc[0] >= 8:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass
    log.info("GPU CC: %d.%d | attn: %s | compute dtype: %s",
             cc[0], cc[1], attn_impl, "bf16" if use_bf16_compute else "fp16")

    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_ID,
        quantization_config=bnb_cfg,
        device_map={"": 0},   # single GPU — pipeline parallel across T4×2 is slower than 1 GPU
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=C.TRAINING["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("Model loaded: %s", C.MODEL_ID)
    return model, tokenizer, attn_impl


# ── LoRA ─────────────────────────────────────────────────────────────────────
def apply_lora(model):
    model = get_peft_model(model, LoraConfig(**C.LORA))
    model.print_trainable_parameters()
    return model


# ── Train ────────────────────────────────────────────────────────────────────
def train(args, model, tokenizer, attn_impl, train_ds, val_ds):
    cc_train = torch.cuda.get_device_capability()
    use_bf16 = cc_train[0] >= 8   # native BF16 only on Ampere+ (CC >= 8.0)
    log.info("Training precision: %s (CC %d.%d)", "bf16" if use_bf16 else "fp16", cc_train[0], cc_train[1])

    run_dir = Path(C.CHECKPOINT_DIR) / args.run_name

    sft_cfg = SFTConfig(
        output_dir=str(run_dir),
        # SFT-specific
        max_seq_length=C.MAX_SEQ_LENGTH,
        packing=C.TRAINING["packing"],
        dataset_text_field=C.TRAINING["dataset_text_field"],
        # Training hyperparams
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        num_train_epochs=C.TRAINING["num_train_epochs"],
        learning_rate=C.TRAINING["learning_rate"],
        lr_scheduler_type=C.TRAINING["lr_scheduler_type"],
        warmup_ratio=C.TRAINING["warmup_ratio"],
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=C.TRAINING["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=C.TRAINING["logging_steps"],
        eval_strategy=C.TRAINING["eval_strategy"],
        eval_steps=C.TRAINING["eval_steps"],
        save_strategy=C.TRAINING["save_strategy"],
        save_steps=C.TRAINING["save_steps"],
        save_total_limit=C.TRAINING["save_total_limit"],
        load_best_model_at_end=False,
        report_to="mlflow",
        run_name=f"vaidya-qlora-{args.run_name}",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    eff_batch = args.batch_size * args.grad_accum
    with mlflow.start_run(run_name=f"vaidya-qlora-{args.run_name}"):
        mlflow.log_params({
            "model_id":       C.MODEL_ID,
            "lora_r":         C.LORA["r"],
            "lora_alpha":     C.LORA["lora_alpha"],
            "lora_dropout":   C.LORA["lora_dropout"],
            "target_modules": str(C.LORA["target_modules"]),
            "max_seq_length": C.MAX_SEQ_LENGTH,
            "batch_size":     args.batch_size,
            "grad_accum":     args.grad_accum,
            "eff_batch_size": eff_batch,
            "learning_rate":  C.TRAINING["learning_rate"],
            "num_epochs":     C.TRAINING["num_train_epochs"],
            "precision":      "bf16" if use_bf16 else "fp16",
            "attn_impl":      attn_impl,
        })

        log.info("Training started — effective batch size: %d", eff_batch)
        t0 = time.time()
        result = trainer.train()
        elapsed = time.time() - t0

        mlflow.log_metrics({
            "final_train_loss":   result.training_loss,
            "train_runtime_sec":  elapsed,
            "samples_per_second": result.metrics.get("train_samples_per_second", 0),
        })

        trainer.save_model(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))
        log.info("Checkpoint saved → %s", run_dir)

    return trainer, run_dir


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    log.info("train.py v%s", __version__)
    setup_credentials()
    setup_mlflow("vaidya-qlora", C.S3_BUCKET)

    train_ds, val_ds            = load_data()
    model, tokenizer, attn_impl = load_model_and_tokenizer()
    model                       = apply_lora(model)
    trainer, run_dir            = train(args, model, tokenizer, attn_impl, train_ds, val_ds)

    try:
        push_to_s3(run_dir, args.run_name, C.S3_BUCKET)
    except Exception as e:
        log.warning("S3 upload failed — checkpoint still local at %s: %s", run_dir, e)

    log.info("✅ Phase 2 complete.")
    log.info("   Checkpoint : %s/checkpoints/%s/", C.S3_BUCKET, args.run_name)
    log.info("   MLflow db  : %s/mlflow/mlflow.db", C.S3_BUCKET)
    log.info("   Next       : Phase 3 — eval accuracy (base vs fine-tuned)")


if __name__ == "__main__":
    main()
