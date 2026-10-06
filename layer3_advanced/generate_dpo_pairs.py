"""
Vaidya — Phase 4: Generate DPO preference pairs from MedMCQA val set.

For each question:
  prompt   = ChatML user turn (no assistant)
  chosen   = correct answer + explanation (high quality)
  rejected = wrong answer formatted identically (low quality)

Saves dpo_pairs.jsonl → S3.

Usage:
    python generate_dpo_pairs.py --n-pairs 500
"""

__version__ = "1.0"

import sys, json, argparse, logging, random
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer2_finetune"))
import qlora_config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
OPT_MAP   = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}

SYSTEM_PROMPT = (
    "You are Vaidya, an expert in Indian medical licensing exams (AIIMS, PGI, USMLE-equivalent). "
    "Answer the question by selecting the single best option."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_prompt(row: dict) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: {row['question']}\n\n"
        f"Options:\nA. {row['opa']}\nB. {row['opb']}\nC. {row['opc']}\nD. {row['opd']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_response(label: str, option_text: str, explanation: str | None) -> str:
    resp = f"The correct answer is {label}. {option_text}."
    if explanation and len(explanation.strip()) > 10:
        resp += f" {explanation.strip()}"
    resp += "<|im_end|>"
    return resp


def pick_wrong_label(correct: int) -> int:
    options = [i for i in range(4) if i != correct]
    return random.choice(options)


def main():
    args = parse_args()
    random.seed(args.seed)

    # Use train set for DPO pairs (val stays held-out for evaluation)
    data_dir = Path("./data")
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        log.info("Downloading train.parquet from S3...")
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        data_dir.mkdir(exist_ok=True)
        s3.download_file(bucket, "data/train.parquet", str(train_path))

    df = pd.read_parquet(train_path)
    df = df.dropna(subset=["question", "opa", "opb", "opc", "opd", "cop"])
    df = df.sample(min(args.n_pairs, len(df)), random_state=args.seed).reset_index(drop=True)
    log.info("Generating %d DPO pairs from train set", len(df))

    pairs = []
    for _, row in df.iterrows():
        correct_idx  = int(row["cop"])
        correct_lbl  = LABEL_MAP[correct_idx]
        correct_text = row[OPT_MAP[correct_idx]]
        explanation  = row.get("exp", None) if isinstance(row.get("exp"), str) else None

        wrong_idx  = pick_wrong_label(correct_idx)
        wrong_lbl  = LABEL_MAP[wrong_idx]
        wrong_text = row[OPT_MAP[wrong_idx]]

        pairs.append({
            "prompt":   build_prompt(row),
            "chosen":   build_response(correct_lbl, correct_text, explanation),
            "rejected": build_response(wrong_lbl,   wrong_text,   None),
        })

    out_path = Path("dpo_pairs.jsonl")
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    log.info("Saved %d pairs → %s", len(pairs), out_path)

    # Push to S3
    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        s3.upload_file(str(out_path), bucket, "data/dpo_pairs.jsonl")
        log.info("dpo_pairs.jsonl → %s/data/", C.S3_BUCKET)
    except Exception as e:
        log.warning("S3 upload failed: %s", e)

    log.info("✅ Done — %d pairs", len(pairs))


if __name__ == "__main__":
    main()
