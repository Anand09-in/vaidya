"""
Vaidya — Phase 2: QLoRA fine-tuning on MedMCQA.

Run on Kaggle P100 via kaggle_train.ipynb, or directly:
    python train.py [--run-name run1]

Or with DeepSpeed (single GPU via Accelerate):
    accelerate launch --config_file accelerate_config.yaml train.py
"""

__version__ = "1.2"

import os, sys, time, argparse, logging
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
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

import qlora_config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run1",
                   help="Name for this training run (used in checkpoint dir and MLflow)")
    p.add_argument("--batch-size", type=int, default=C.TRAINING["per_device_train_batch_size"])
    p.add_argument("--grad-accum", type=int, default=C.TRAINING["gradient_accumulation_steps"])
    return p.parse_args()


# ── AWS / credentials ────────────────────────────────────────────────────────
def setup_credentials():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        from kaggle_secrets import UserSecretsClient
        s = UserSecretsClient()
        os.environ["AWS_ACCESS_KEY_ID"]     = s.get_secret("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = s.get_secret("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"]    = s.get_secret("AWS_DEFAULT_REGION")
        log.info("AWS credentials loaded from Kaggle secrets.")
    elif "google.colab" in sys.modules:
        from google.colab import userdata
        os.environ["AWS_ACCESS_KEY_ID"]     = userdata.get("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"]    = userdata.get("AWS_DEFAULT_REGION")
        log.info("AWS credentials loaded from Colab secrets.")
    else:
        os.environ.setdefault("AWS_PROFILE", "vaidya")
        log.info("Using AWS profile: vaidya")


# ── MLflow ───────────────────────────────────────────────────────────────────
def setup_mlflow():
    artifact_loc = f"{C.S3_BUCKET}/mlflow"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # MLflow 3.x ignores MLFLOW_ARTIFACT_ROOT — set artifact location on the experiment
    try:
        mlflow.create_experiment("vaidya-qlora", artifact_location=artifact_loc)
    except Exception:
        pass  # already exists
    mlflow.set_experiment("vaidya-qlora")
    log.info("MLflow: sqlite:///mlflow.db  |  artifacts → %s", artifact_loc)


# ── Data ─────────────────────────────────────────────────────────────────────
def load_data():
    # Explicit path: Kaggle working dir is /kaggle/working; elsewhere relative to repo root
    if Path("/kaggle/working").exists():
        data_dir = Path("/kaggle/working/data")
    else:
        data_dir = Path(__file__).resolve().parent.parent / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.parquet"

    if not train_path.exists():
        log.info("Parquet splits not found at %s — pulling from S3 via boto3...", data_dir)
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix="data/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    dest = data_dir / key.split("/")[-1]
                    log.info("  %s → %s", key, dest)
                    s3.download_file(bucket, key, str(dest))
        assert train_path.exists(), f"train.parquet still missing after S3 download at {train_path}"

    train_df = pd.read_parquet(data_dir / "train.parquet")[["text"]]
    val_df   = pd.read_parquet(data_dir / "val.parquet")[["text"]]

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df,   preserve_index=False)

    log.info("Data loaded from %s — train: %d  val: %d", data_dir, len(train_ds), len(val_ds))
    return train_ds, val_ds


# ── Model + tokenizer ────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    import torch
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Flash Attention 2 requires Ampere+ (compute capability ≥ 8.0).
    # P100 is Pascal (6.0) → fall back to standard attention.
    cc = torch.cuda.get_device_capability()
    attn_impl = "eager"
    if cc[0] >= 8:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass
    log.info("GPU compute capability: %d.%d — attention: %s", cc[0], cc[1], attn_impl)

    # T4 (CC 7.5) has no native BF16 hardware — use FP16 for dequantization too
    if not torch.cuda.is_bf16_supported():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for packing

    log.info("Model loaded: %s", C.MODEL_ID)
    return model, tokenizer, attn_impl


# ── LoRA ─────────────────────────────────────────────────────────────────────
def apply_lora(model):
    lora_cfg = LoraConfig(**C.LORA)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Train ────────────────────────────────────────────────────────────────────
def train(args, model, tokenizer, attn_impl, train_ds, val_ds):
    # Auto-detect best mixed precision for this GPU
    use_bf16 = torch.cuda.is_bf16_supported()
    log.info("Mixed precision: %s", "bf16" if use_bf16 else "fp16")

    run_dir = Path(C.CHECKPOINT_DIR) / args.run_name

    import re

    def _safe_init(cls, kwargs):
        """Instantiate cls(**kwargs), retrying after removing each rejected kwarg."""
        kw = dict(kwargs)
        for _ in range(20):  # max 20 unknown params before giving up
            try:
                return cls(**kw)
            except TypeError as e:
                m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                if not m:
                    raise
                bad = m.group(1)
                log.warning("%s: dropping unsupported param '%s'", cls.__name__, bad)
                kw.pop(bad, None)
        raise RuntimeError(f"Could not instantiate {cls.__name__} after removing params")

    desired = dict(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=C.TRAINING["num_train_epochs"],
        learning_rate=C.TRAINING["learning_rate"],
        lr_scheduler_type=C.TRAINING["lr_scheduler_type"],
        warmup_ratio=C.TRAINING["warmup_ratio"],
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=C.TRAINING["logging_steps"],
        eval_strategy=C.TRAINING["eval_strategy"],
        eval_steps=C.TRAINING["eval_steps"],
        save_strategy=C.TRAINING["save_strategy"],
        save_steps=C.TRAINING["save_steps"],
        save_total_limit=C.TRAINING["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="mlflow",
        run_name=f"vaidya-qlora-{args.run_name}",
    )
    sft_extra = dict(
        max_length=C.MAX_SEQ_LENGTH,
        packing=C.TRAINING["packing"],
        dataset_text_field=C.TRAINING["dataset_text_field"],
    )
    trainer_sft_kwargs = {}
    try:
        sft_cfg = _safe_init(SFTConfig, {**desired, **sft_extra})
    except Exception:
        log.warning("SFTConfig failed; falling back to TrainingArguments + trainer-level SFT params")
        sft_cfg = _safe_init(TrainingArguments, desired)
        trainer_sft_kwargs = sft_extra

    import trl
    trainer_kwargs = dict(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        **trainer_sft_kwargs,
    )
    # TRL 0.12+ renamed tokenizer → processing_class; handle major version bump too
    _parts = trl.__version__.split(".")
    _trl_major, _trl_minor = int(_parts[0]), int(_parts[1])
    if _trl_major > 0 or _trl_minor >= 12:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

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

        log.info("Starting training — effective batch size: %d", eff_batch)
        t0 = time.time()
        result = trainer.train()
        elapsed = time.time() - t0

        mlflow.log_metrics({
            "final_train_loss":   result.training_loss,
            "train_runtime_sec":  elapsed,
            "samples_per_second": result.metrics.get("train_samples_per_second", 0),
        })

        # Save adapter + tokenizer
        trainer.save_model(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))
        log.info("Checkpoint saved → %s", run_dir)

    return trainer, run_dir


# ── S3 sync ──────────────────────────────────────────────────────────────────
def push_to_s3(run_dir: Path, run_name: str):
    import boto3
    log.info("Syncing checkpoint to S3 via boto3...")
    s3 = boto3.client("s3")
    bucket = C.S3_BUCKET.replace("s3://", "")

    uploaded = 0
    for local_file in run_dir.rglob("*"):
        if local_file.is_file():
            key = f"checkpoints/{run_name}/{local_file.relative_to(run_dir).as_posix()}"
            s3.upload_file(str(local_file), bucket, key)
            uploaded += 1
    log.info("Uploaded %d checkpoint files to %s/checkpoints/%s/", uploaded, C.S3_BUCKET, run_name)

    mlflow_db = Path("mlflow.db")
    if mlflow_db.exists():
        s3.upload_file(str(mlflow_db), bucket, "mlflow/mlflow.db")
        log.info("MLflow db uploaded to %s/mlflow/mlflow.db", C.S3_BUCKET)


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    log.info("train.py version: %s", __version__)
    setup_credentials()
    setup_mlflow()

    train_ds, val_ds     = load_data()
    model, tokenizer, attn_impl = load_model_and_tokenizer()
    model                       = apply_lora(model)
    trainer, run_dir            = train(args, model, tokenizer, attn_impl, train_ds, val_ds)
    try:
        push_to_s3(run_dir, args.run_name)
    except Exception as e:
        log.warning("S3 upload failed (checkpoint is still local at %s): %s", run_dir, e)

    log.info("✅ Phase 2 complete.")
    log.info("   Checkpoint : %s/checkpoints/%s/", C.S3_BUCKET, args.run_name)
    log.info("   MLflow db  : %s/mlflow/mlflow.db", C.S3_BUCKET)
    log.info("   Next step  : Phase 3 — eval accuracy (base vs fine-tuned)")


if __name__ == "__main__":
    main()
