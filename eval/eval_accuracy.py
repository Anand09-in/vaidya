"""
Vaidya — Phase 3: Accuracy eval
Base Mistral-7B-Instruct-v0.3 vs SFT (LoRA) vs DPO on MedMCQA test set.

Usage:
    python eval_accuracy.py --run-name run3
    python eval_accuracy.py --run-name run3 --dpo-run-name dpo-run1
    python eval_accuracy.py --run-name run3 --dpo-run-name dpo-run1 --finetuned-only
"""

__version__ = "1.2"

import os, re, sys, json, argparse, logging
from pathlib import Path

import torch
import pandas as pd
import mlflow
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

_layer2 = str(Path(__file__).resolve().parent.parent / "layer2_finetune")
sys.path.insert(0, _layer2)
import qlora_config as C
from utils import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",     default="run3",    help="SFT LoRA checkpoint run name")
    p.add_argument("--dpo-run-name", default=None,      help="DPO checkpoint run name (skipped if omitted)")
    p.add_argument("--n-samples",    type=int, default=1000, help="Stratified sample size from test set")
    p.add_argument("--base-only",    action="store_true", help="Only eval base model")
    p.add_argument("--finetuned-only", action="store_true", help="Only eval SFT+DPO (skip base)")
    p.add_argument("--debug",        action="store_true", help="Print 5 sample generations")
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────────────
def load_test_data(n_samples: int):
    data_dir = Path("./data")
    test_path = data_dir / "test.parquet"

    if not test_path.exists():
        log.info("Downloading test.parquet from S3...")
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        data_dir.mkdir(exist_ok=True)
        s3.download_file(bucket, "data/test.parquet", str(test_path))

    df = pd.read_parquet(test_path)
    log.info("Test set: %d rows", len(df))

    # Stratified sample by subject_name if column exists, else random
    if "subject_name" in df.columns and n_samples < len(df):
        df = (df.groupby("subject_name", group_keys=False)
                .apply(lambda g: g.sample(frac=n_samples / len(df), random_state=42))
                .reset_index(drop=True))
        df = df.iloc[:n_samples]
    else:
        df = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    log.info("Eval sample: %d rows", len(df))
    return df


# ── Model loading ─────────────────────────────────────────────────────────────
def load_base_model():
    cc = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_ID,
        quantization_config=bnb_cfg,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    log.info("Base model loaded: %s", C.MODEL_ID)
    return model, tokenizer


def _download_adapter(run_name: str) -> Path:
    adapter_dir = Path(f"./checkpoints/{run_name}")
    if not adapter_dir.exists():
        log.info("Adapter not found locally — downloading from S3...")
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"checkpoints/{run_name}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                fname = key.split("/", 2)[-1]
                dest = adapter_dir / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
        log.info("Adapter downloaded → %s", adapter_dir)
    return adapter_dir


def load_finetuned_model(run_name: str):
    model, tokenizer = load_base_model()
    adapter_dir = _download_adapter(run_name)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    log.info("SFT model loaded from %s", adapter_dir)
    return model, tokenizer


def load_dpo_model(dpo_run_name: str):
    """Load base + DPO adapter (DPOTrainer saves SFT+DPO weights into one adapter)."""
    model, tokenizer = load_base_model()
    adapter_dir = _download_adapter(dpo_run_name)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    log.info("DPO model loaded from %s", adapter_dir)
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────
def build_prompt(row: dict, tokenizer) -> str:
    # Must match the ChatML format used in training data (layer1_data/02_clean_format.ipynb)
    # NOT tokenizer.apply_chat_template() which outputs Mistral [INST] format
    return (
        "<|im_start|>system\n"
        "You are Vaidya, an expert in Indian medical licensing exams (AIIMS, PGI, USMLE-equivalent). "
        "Answer the question by selecting the single best option.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: {row['question']}\n\n"
        f"Options:\nA. {row['opa']}\nB. {row['opb']}\nC. {row['opc']}\nD. {row['opd']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_answer(text: str) -> str | None:
    """Extract A/B/C/D from model output."""
    patterns = [
        r"correct answer is ([A-D])",
        r"\bThe answer is ([A-D])\b",
        r"^([A-D])[.):] ",
        r"\b([A-D])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


@torch.inference_mode()
def evaluate(model, tokenizer, df: pd.DataFrame, batch_size: int = 8, debug: bool = False) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_subject: dict[str, list[bool]] = {}
    errors = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        prompts = [build_prompt(row, tokenizer) for _, row in batch.iterrows()]
        labels  = [LABEL_MAP[row["cop"]] for _, row in batch.iterrows()]
        subjects = batch.get("subject_name", pd.Series(["unknown"] * len(batch))).tolist()

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        for j, (ids, label, subj) in enumerate(zip(out, labels, subjects)):
            new_tokens = ids[inputs["input_ids"].shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred = extract_answer(generated)
            hit = pred == label
            if debug and total < 5:
                print(f"\n--- Sample {total+1} ---")
                print(f"  Label   : {label}")
                print(f"  Output  : {repr(generated[:120])}")
                print(f"  Pred    : {pred}  {'✅' if hit else '❌'}")
            correct += int(hit)
            total += 1
            per_subject.setdefault(subj, []).append(hit)
            if pred is None:
                errors.append({"prompt": prompts[j][:100], "generated": generated})

        if (i // batch_size) % 10 == 0:
            log.info("  %d/%d — running accuracy %.1f%%", total, len(df), 100 * correct / total)

    subject_acc = {s: sum(v) / len(v) for s, v in per_subject.items()}
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "subject_accuracy": subject_acc,
        "parse_errors": len(errors),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    log.info("eval_accuracy.py v%s  |  sft=%s  dpo=%s  n=%d",
             __version__, args.run_name, args.dpo_run_name or "skip", args.n_samples)
    setup_mlflow("vaidya-eval", C.S3_BUCKET)

    df = load_test_data(args.n_samples)

    results = {}

    if not args.finetuned_only:
        log.info("=== Evaluating BASE model ===")
        model, tokenizer = load_base_model()
        results["base"] = evaluate(model, tokenizer, df, debug=args.debug)
        log.info("Base accuracy: %.2f%%", results["base"]["accuracy"] * 100)
        del model
        torch.cuda.empty_cache()

    if not args.base_only:
        log.info("=== Evaluating SFT model (run=%s) ===", args.run_name)
        model, tokenizer = load_finetuned_model(args.run_name)
        results["finetuned"] = evaluate(model, tokenizer, df, debug=args.debug)
        log.info("SFT accuracy: %.2f%%", results["finetuned"]["accuracy"] * 100)
        del model
        torch.cuda.empty_cache()

        if args.dpo_run_name:
            log.info("=== Evaluating DPO model (run=%s) ===", args.dpo_run_name)
            model, tokenizer = load_dpo_model(args.dpo_run_name)
            results["dpo"] = evaluate(model, tokenizer, df, debug=args.debug)
            log.info("DPO accuracy: %.2f%%", results["dpo"]["accuracy"] * 100)
            del model
            torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────────
    if "base" in results and "finetuned" in results:
        results["sft_delta"] = results["finetuned"]["accuracy"] - results["base"]["accuracy"]
        log.info("Base → SFT delta: %+.2f%%", results["sft_delta"] * 100)
    if "finetuned" in results and "dpo" in results:
        results["dpo_delta"] = results["dpo"]["accuracy"] - results["finetuned"]["accuracy"]
        log.info("SFT  → DPO delta: %+.2f%%", results["dpo_delta"] * 100)
    if "base" in results and "dpo" in results:
        results["total_delta"] = results["dpo"]["accuracy"] - results["base"]["accuracy"]
        log.info("Base → DPO total: %+.2f%%", results["total_delta"] * 100)
    elif "base" in results and "finetuned" in results:
        results["delta"] = results["sft_delta"]

    # ── Save + MLflow ─────────────────────────────────────────────────────────
    out_path = Path("eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved → %s", out_path)

    run_label = args.dpo_run_name or args.run_name
    with mlflow.start_run(run_name=f"eval-{run_label}"):
        if "base" in results:
            mlflow.log_metric("base_accuracy",     results["base"]["accuracy"])
        if "finetuned" in results:
            mlflow.log_metric("sft_accuracy",      results["finetuned"]["accuracy"])
        if "dpo" in results:
            mlflow.log_metric("dpo_accuracy",      results["dpo"]["accuracy"])
        if "sft_delta" in results:
            mlflow.log_metric("sft_delta",         results["sft_delta"])
        if "dpo_delta" in results:
            mlflow.log_metric("dpo_delta",         results["dpo_delta"])
        if "total_delta" in results:
            mlflow.log_metric("total_delta",       results["total_delta"])
        mlflow.log_artifact(str(out_path))

    # ── Push to S3 ───────────────────────────────────────────────────────────
    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        s3.upload_file(str(out_path), bucket, "eval/eval_results.json")
        log.info("eval_results.json → %s/eval/", C.S3_BUCKET)
    except Exception as e:
        log.warning("S3 upload failed: %s", e)

    log.info("✅ Eval complete.")


if __name__ == "__main__":
    main()
