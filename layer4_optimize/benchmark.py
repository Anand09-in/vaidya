"""
Vaidya — Phase 5: Benchmark all quantized formats.

Measures inference latency and model size across:
  • BF16 merged (baseline)
  • GPTQ 4-bit
  • AWQ 4-bit
  • GGUF Q4_K_M / Q5_K_M / Q8_0

Usage:
    python benchmark.py [--n-runs 50] [--formats gptq awq gguf]

Saves benchmark_results.json → S3.
"""

__version__ = "1.0"

import os, sys, json, argparse, logging, time, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer2_finetune"))
import qlora_config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEST_PROMPT = (
    "<|im_start|>system\nYou are Vaidya, an expert medical AI.<|im_end|>\n"
    "<|im_start|>user\nQuestion: What is the most common cause of iron deficiency "
    "anaemia in India?\nA. Malaria\nB. Hookworm\nC. Poor diet\nD. Bleeding peptic ulcer"
    "<|im_end|>\n<|im_start|>assistant\n"
)
MAX_NEW_TOKENS = 20
GGUF_DIR = Path("./gguf-models")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--formats", nargs="+",
                   default=["merged", "gptq", "awq", "gguf"],
                   choices=["merged", "gptq", "awq", "gguf"])
    return p.parse_args()


def _latency_stats(times):
    times.sort()
    return {
        "p50_ms":  round(statistics.median(times) * 1000, 1),
        "p95_ms":  round(times[int(0.95 * len(times))] * 1000, 1),
        "mean_ms": round(statistics.mean(times) * 1000, 1),
    }


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 1)


# ── HF-format benchmarks (transformers) ─────────────────────────────────────

def bench_hf(model_dir: Path, n_runs: int, label: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    log.info("[%s] Loading from %s ...", label, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to("cuda")
    times = []
    with torch.inference_mode():
        # warmup
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
            times.append(time.perf_counter() - t0)

    del model
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    stats = _latency_stats(times)
    stats["size_mb"] = dir_size_mb(model_dir)
    log.info("[%s] p50=%.1f ms  p95=%.1f ms  size=%.0f MB",
             label, stats["p50_ms"], stats["p95_ms"], stats["size_mb"])
    return stats


def bench_gptq(n_runs: int):
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer
    import torch

    gptq_dir = Path("./gptq-4bit")
    if not gptq_dir.exists():
        log.warning("GPTQ dir not found, skipping.")
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(gptq_dir))
    model = GPTQModel.load(str(gptq_dir), device="cuda:0")
    model.eval()

    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to("cuda")
    times = []
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
            times.append(time.perf_counter() - t0)

    del model
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    stats = _latency_stats(times)
    stats["size_mb"] = dir_size_mb(gptq_dir)
    log.info("[GPTQ] p50=%.1f ms  p95=%.1f ms  size=%.0f MB",
             stats["p50_ms"], stats["p95_ms"], stats["size_mb"])
    return stats


def bench_awq(n_runs: int):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    import torch

    awq_dir = Path("./awq-4bit")
    if not awq_dir.exists():
        log.warning("AWQ dir not found, skipping.")
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(awq_dir))
    model = AutoAWQForCausalLM.from_quantized(str(awq_dir), fuse_layers=True)
    model.eval()

    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to("cuda")
    times = []
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
            times.append(time.perf_counter() - t0)

    del model
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    stats = _latency_stats(times)
    stats["size_mb"] = dir_size_mb(awq_dir)
    log.info("[AWQ]  p50=%.1f ms  p95=%.1f ms  size=%.0f MB",
             stats["p50_ms"], stats["p95_ms"], stats["size_mb"])
    return stats


def bench_gguf(n_runs: int):
    try:
        from llama_cpp import Llama
    except ImportError:
        log.warning("llama-cpp-python not installed — run: pip install llama-cpp-python")
        return {}

    results = {}
    for gguf_file in sorted(GGUF_DIR.glob("*.gguf")):
        tag = gguf_file.stem.replace("vaidya-", "")
        log.info("[GGUF %s] Loading %s ...", tag, gguf_file)
        llm = Llama(model_path=str(gguf_file), n_gpu_layers=-1, n_ctx=1024, verbose=False)
        times = []
        llm(TEST_PROMPT, max_tokens=MAX_NEW_TOKENS)  # warmup
        for _ in range(n_runs):
            t0 = time.perf_counter()
            llm(TEST_PROMPT, max_tokens=MAX_NEW_TOKENS)
            times.append(time.perf_counter() - t0)
        del llm

        stats = _latency_stats(times)
        stats["size_mb"] = file_size_mb(gguf_file)
        log.info("[GGUF %s] p50=%.1f ms  p95=%.1f ms  size=%.0f MB",
                 tag, stats["p50_ms"], stats["p95_ms"], stats["size_mb"])
        results[f"gguf_{tag}"] = stats

    return results


def print_table(results: dict):
    print("\n" + "=" * 72)
    print(f"{'Format':<22}  {'p50 ms':>8}  {'p95 ms':>8}  {'Size MB':>9}")
    print("-" * 72)
    for fmt, stats in results.items():
        if stats:
            print(f"{fmt:<22}  {stats['p50_ms']:>8.1f}  {stats['p95_ms']:>8.1f}  {stats['size_mb']:>9.0f}")
    print("=" * 72)


def main():
    args = parse_args()
    results = {}

    if "merged" in args.formats:
        merged_dir = Path("../layer3_advanced/merged_model")
        if merged_dir.exists():
            results["merged_bf16"] = bench_hf(merged_dir, args.n_runs, "BF16-merged")
        else:
            log.warning("Merged model not found, skipping BF16 baseline.")

    if "gptq" in args.formats:
        results["gptq_4bit"] = bench_gptq(args.n_runs)

    if "awq" in args.formats:
        results["awq_4bit"] = bench_awq(args.n_runs)

    if "gguf" in args.formats:
        results.update(bench_gguf(args.n_runs))

    print_table(results)

    out = Path("benchmark_results.json")
    out.write_text(json.dumps(results, indent=2))
    log.info("Results saved → %s", out)

    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = C.S3_BUCKET.replace("s3://", "")
        s3.upload_file(str(out), bucket, "models/benchmark_results.json")
        log.info("✅ Benchmark → %s/models/benchmark_results.json", C.S3_BUCKET)
    except Exception as e:
        log.warning("S3 upload failed: %s", e)


if __name__ == "__main__":
    main()
