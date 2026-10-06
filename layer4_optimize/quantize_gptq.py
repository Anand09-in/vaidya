"""
Vaidya — Phase 5: GPTQ 4-bit quantization (gptqmodel).

Usage:
    python quantize_gptq.py
"""

__version__ = "1.1"

import sys, logging
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
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="models/merged/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]; fname = key.replace("models/merged/", "")
            if fname:
                dest = MERGED_DIR / fname; dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
    log.info("Merged model downloaded → %s", MERGED_DIR)


def load_calibration_data(n=128):
    import pandas as pd
    train_path = Path("./data/train.parquet")
    if not train_path.exists():
        import boto3
        s3 = boto3.client("s3"); train_path.parent.mkdir(exist_ok=True)
        s3.download_file(C.S3_BUCKET.replace("s3://", ""), "data/train.parquet", str(train_path))
    df = pd.read_parquet(train_path).sample(n, random_state=42)
    return [
        f"<|im_start|>user\nQuestion: {row['question']}\n"
        f"A. {row['opa']}\nB. {row['opb']}\nC. {row['opc']}\nD. {row['opd']}\n<|im_end|>\n"
        f"<|im_start|>assistant\nThe correct answer is"
        for _, row in df.iterrows()
    ]


def main():
    from gptqmodel import GPTQModel, QuantizeConfig, BACKEND

    download_merged()

    calib_data = load_calibration_data()
    log.info("Calibration samples: %d", len(calib_data))

    quant_cfg = QuantizeConfig(bits=4, group_size=128, desc_act=False)

    # Backend.TORCH avoids Marlin/ExllamaV2 JIT compilation (~2 min) that can
    # disconnect Lightning.ai studios.
    log.info("Loading merged model for GPTQ calibration...")
    model = GPTQModel.load(str(MERGED_DIR), quantize_config=quant_cfg,
                           backend=BACKEND.TORCH)

    log.info("Quantizing...")
    model.quantize(calib_data)

    OUT_DIR.mkdir(exist_ok=True)
    model.save(str(OUT_DIR))
    log.info("GPTQ model saved → %s", OUT_DIR)

    size_mb = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) / 1e6
    log.info("Model size: %.0f MB", size_mb)

    import boto3
    s3 = boto3.client("s3"); bucket = C.S3_BUCKET.replace("s3://", "")
    for f in OUT_DIR.rglob("*"):
        if f.is_file():
            s3.upload_file(str(f), bucket, f"models/gptq-4bit/{f.relative_to(OUT_DIR).as_posix()}")
    log.info("✅ GPTQ model → %s/models/gptq-4bit/", C.S3_BUCKET)


if __name__ == "__main__":
    main()
