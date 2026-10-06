"""
Vaidya — Phase 5: GPTQ 4-bit quantization.

Loads merged model, calibrates on 128 val samples, saves GPTQ model.

Usage:
    python quantize_gptq.py
"""

__version__ = "1.0"

import os, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer2_finetune"))
import qlora_config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MERGED_DIR = Path("../layer3_advanced/merged_model")
OUT_DIR    = Path("./gptq-4bit")


def download_merged():
    if MERGED_DIR.exists():
        return
    log.info("Downloading merged model from S3...")
    import boto3
    s3 = boto3.client("s3")
    bucket = C.S3_BUCKET.replace("s3://", "")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="models/merged/"):
        for obj in page.get("Contents", []):
            key  = obj["Key"]
            fname = key.replace("models/merged/", "")
            if not fname:
                continue
            dest = MERGED_DIR / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
    log.info("Merged model downloaded → %s", MERGED_DIR)


def load_calibration_data(tokenizer, n=128):
    import pandas as pd
    val_path = Path("./data/val.parquet")
    if not val_path.exists():
        import boto3
        s3 = boto3.client("s3")
        val_path.parent.mkdir(exist_ok=True)
        s3.download_file(C.S3_BUCKET.replace("s3://", ""), "data/val.parquet", str(val_path))
    df = pd.read_parquet(val_path).sample(n, random_state=42)
    texts = []
    for _, row in df.iterrows():
        texts.append(
            f"<|im_start|>user\nQuestion: {row['question']}\n"
            f"A. {row['opa']}\nB. {row['opb']}\nC. {row['opc']}\nD. {row['opd']}\n<|im_end|>\n"
            f"<|im_start|>assistant\nThe correct answer is"
        )
    return [tokenizer(t, return_tensors="pt").input_ids for t in texts]


def main():
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    download_merged()

    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR))
    calib_data = load_calibration_data(tokenizer)

    quant_cfg = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
    )

    log.info("Loading merged model for GPTQ calibration...")
    model = AutoGPTQForCausalLM.from_pretrained(
        str(MERGED_DIR),
        quantize_config=quant_cfg,
    )

    log.info("Calibrating on %d samples...", len(calib_data))
    model.quantize(calib_data)

    OUT_DIR.mkdir(exist_ok=True)
    model.save_quantized(str(OUT_DIR), use_safetensors=True)
    tokenizer.save_pretrained(str(OUT_DIR))
    log.info("GPTQ model saved → %s", OUT_DIR)

    size_mb = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) / 1e6
    log.info("Model size: %.0f MB", size_mb)

    # Push to S3
    import boto3
    s3 = boto3.client("s3")
    bucket = C.S3_BUCKET.replace("s3://", "")
    for f in OUT_DIR.rglob("*"):
        if f.is_file():
            s3.upload_file(str(f), bucket, f"models/gptq-4bit/{f.relative_to(OUT_DIR).as_posix()}")
    log.info("✅ GPTQ model → %s/models/gptq-4bit/", C.S3_BUCKET)


if __name__ == "__main__":
    main()
