"""
Vaidya — Phase 4: Merge LoRA adapter into base model and push to HuggingFace Hub.

Merges the DPO adapter (or SFT adapter if --skip-dpo) into base Mistral-7B,
saves the merged full-precision model, and pushes to HF Hub + S3.

Usage:
    python merge_adapter.py --dpo-run-name dpo-run1
    python merge_adapter.py --sft-run-name run3 --skip-dpo   # merge SFT adapter only
"""

__version__ = "1.0"

import sys, argparse, logging, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer2_finetune"))
import qlora_config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_REPO = "SneakySpidy/vaidya-mistral-7b-medmcqa"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dpo-run-name", default="dpo-run1")
    p.add_argument("--sft-run-name", default="run3")
    p.add_argument("--skip-dpo", action="store_true",
                   help="Merge SFT adapter only (skip DPO layer)")
    p.add_argument("--skip-hub", action="store_true",
                   help="Skip HuggingFace Hub push (S3 only)")
    return p.parse_args()


def download_adapter(run_name: str):
    adapter_dir = Path(f"./checkpoints/{run_name}")
    if not adapter_dir.exists():
        log.info("Downloading %s from S3...", run_name)
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"checkpoints/{run_name}/"):
            for obj in page.get("Contents", []):
                key   = obj["Key"]
                fname = key.split("/", 2)[-1]
                dest  = adapter_dir / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
        log.info("Downloaded → %s", adapter_dir)
    return adapter_dir


def main():
    args = parse_args()
    log.info("merge_adapter.py v%s", __version__)

    # Load base model in full precision for merging
    log.info("Loading base model in BF16 for merge...")
    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # merge on CPU to avoid VRAM limit
    )
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_ID)

    # Apply SFT adapter
    sft_dir = download_adapter(args.sft_run_name)
    log.info("Applying SFT adapter: %s", sft_dir)
    model = PeftModel.from_pretrained(model, str(sft_dir))
    model = model.merge_and_unload()
    log.info("SFT adapter merged.")

    # Apply DPO adapter on top (if not skipped)
    if not args.skip_dpo:
        dpo_dir = download_adapter(args.dpo_run_name)
        log.info("Applying DPO adapter: %s", dpo_dir)
        model = PeftModel.from_pretrained(model, str(dpo_dir))
        model = model.merge_and_unload()
        log.info("DPO adapter merged.")

    # Save merged model
    merged_dir = Path("./merged_model")
    merged_dir.mkdir(exist_ok=True)
    log.info("Saving merged model → %s  (this takes ~5 min)...", merged_dir)
    t0 = time.time()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    log.info("Saved in %.1f s", time.time() - t0)

    # Latency benchmark: merged vs adapter inference
    log.info("Running latency benchmark (50 generations)...")
    model.eval()
    model = model.cuda()
    test_prompt = (
        "<|im_start|>system\nYou are Vaidya.<|im_end|>\n"
        "<|im_start|>user\nQuestion: What is the most common cause of iron deficiency anaemia in India?\n"
        "Options:\nA. Malaria\nB. Hookworm\nC. Poor diet\nD. Bleeding peptic ulcer\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    times = []
    with torch.inference_mode():
        for _ in range(50):
            t0 = time.perf_counter()
            model.generate(**inputs, max_new_tokens=20, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
            times.append(time.perf_counter() - t0)
    import statistics
    p50 = statistics.median(times) * 1000
    p95 = sorted(times)[int(0.95 * len(times))] * 1000
    log.info("Merged model latency — p50: %.1f ms  p95: %.1f ms", p50, p95)

    # Push to S3
    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        uploaded = 0
        log.info("Uploading merged model to S3...")
        for f in merged_dir.rglob("*"):
            if f.is_file():
                key = f"models/merged/{f.relative_to(merged_dir).as_posix()}"
                s3.upload_file(str(f), bucket, key)
                uploaded += 1
        log.info("Uploaded %d files → %s/models/merged/", uploaded, C.S3_BUCKET)
    except Exception as e:
        log.warning("S3 upload failed: %s", e)

    # Push to HuggingFace Hub
    if not args.skip_hub:
        import os
        from huggingface_hub import login as hf_login
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            hf_login(token=hf_token)
        log.info("Pushing to HuggingFace Hub: %s ...", HF_REPO)
        model.push_to_hub(HF_REPO, safe_serialization=True)
        tokenizer.push_to_hub(HF_REPO)
        log.info("✅ Model live at https://huggingface.co/%s", HF_REPO)

    log.info("✅ Phase 4 merge complete.")
    log.info("   Merged model : %s/models/merged/", C.S3_BUCKET)
    if not args.skip_hub:
        log.info("   HF Hub       : https://huggingface.co/%s", HF_REPO)
    log.info("   Latency      : p50=%.1f ms  p95=%.1f ms", p50, p95)
    log.info("   Next         : Phase 5 — quantization (GPTQ / AWQ / GGUF)")


if __name__ == "__main__":
    main()
