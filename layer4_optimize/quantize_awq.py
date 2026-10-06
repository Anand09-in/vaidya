"""
Vaidya — Phase 5: AWQ 4-bit quantization.

Usage:
    python quantize_awq.py
"""

__version__ = "1.0"

import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer2_finetune"))
import qlora_config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MERGED_DIR = Path("../layer3_advanced/merged_model")
OUT_DIR    = Path("./awq-4bit")


def main():
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    import pandas as pd

    if not MERGED_DIR.exists():
        log.info("Downloading merged model from S3...")
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="models/merged/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]; fname = key.replace("models/merged/", "")
                if fname:
                    dest = MERGED_DIR / fname; dest.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, key, str(dest))

    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR))

    # Calibration data
    train_path = Path("./data/train.parquet")
    if not train_path.exists():
        import boto3
        s3 = boto3.client("s3"); train_path.parent.mkdir(exist_ok=True)
        s3.download_file(C.S3_BUCKET.replace("s3://", ""), "data/train.parquet", str(train_path))
    df = pd.read_parquet(train_path).sample(128, random_state=42)
    calib_data = [
        f"<|im_start|>user\nQuestion: {row['question']}\n"
        f"A. {row['opa']}\nB. {row['opb']}\nC. {row['opc']}\nD. {row['opd']}\n<|im_end|>\n"
        f"<|im_start|>assistant\nThe correct answer is"
        for _, row in df.iterrows()
    ]

    log.info("Loading merged model for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(str(MERGED_DIR), device_map="cuda")

    quant_cfg = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    log.info("Quantizing with AWQ...")
    model.quantize(tokenizer, quant_config=quant_cfg, calib_data=calib_data)

    OUT_DIR.mkdir(exist_ok=True)
    model.save_quantized(str(OUT_DIR), safetensors=True)
    tokenizer.save_pretrained(str(OUT_DIR))
    log.info("AWQ model saved → %s", OUT_DIR)

    size_mb = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) / 1e6
    log.info("Model size: %.0f MB", size_mb)

    import boto3
    s3 = boto3.client("s3"); bucket = C.S3_BUCKET.replace("s3://", "")
    for f in OUT_DIR.rglob("*"):
        if f.is_file():
            s3.upload_file(str(f), bucket, f"models/awq-4bit/{f.relative_to(OUT_DIR).as_posix()}")
    log.info("✅ AWQ model → %s/models/awq-4bit/", C.S3_BUCKET)


if __name__ == "__main__":
    main()
