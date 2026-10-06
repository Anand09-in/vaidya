# Vaidya (वैद्य)

**End-to-end LLM fine-tuning pipeline for Indian medical Q&A**

QLoRA fine-tuned Mistral-7B on 187k Indian medical exam questions (MedMCQA), with DPO preference tuning, three quantization formats (GPTQ / AWQ / GGUF), FastAPI serving, and a Prometheus + Grafana monitoring stack.

> MedMCQA accuracy: **Base 44.8%** → **Fine-tuned 59.3%** (+14.5 pp)

---

## Architecture

```
Raw MedMCQA (187k)
  └── Layer 1: Data Pipeline       → Clean Parquet splits + quality report
        └── Layer 2: QLoRA Training → Mistral-7B + LoRA adapter (MLflow tracked)
              └── Layer 3: DPO      → Preference tuning + merged model on HF Hub
                    └── Layer 4: Quantization → GPTQ / AWQ / GGUF — 3.5× compression
                          └── Layer 5: Production → FastAPI + vLLM + Prometheus + Grafana
```

---

## Repo structure

```
vaidya/
├── layer1_data/          # Data pipeline notebooks
├── layer2_finetune/      # QLoRA training scripts + shared utils
│   └── lightning/        # Self-contained Lightning.ai training copy
├── layer3_advanced/      # DPO training + adapter merge → HF Hub
├── layer4_optimize/      # GPTQ / AWQ / GGUF quantization + benchmark
├── layer5_production/    # FastAPI + vLLM + Prometheus + Grafana (Docker Compose)
├── eval/                 # Accuracy eval script
├── layer3_eval/          # Lightning.ai eval notebook
└── scripts/              # One-time S3 setup
```

Data and model artifacts live in **S3 (`s3://vaidya-artifacts/`)** — not in this repo.

---

## Results

### Accuracy

| Model | MedMCQA Accuracy |
|-------|-----------------|
| Mistral-7B base | 44.8% |
| + QLoRA SFT (run3, 680 steps) | 59.3% |
| + DPO preference tuning | *(pending re-eval)* |

Evaluated on 1,000 stratified test samples across 19 medical subjects.

### Quantization benchmark

| Format | Size | p50 latency | p95 latency | vs BF16 size |
|--------|------|-------------|-------------|--------------|
| BF16 merged | 14.5 GB | 1,233 ms | 1,239 ms | 1× |
| GPTQ 4-bit | 4.2 GB | 2,484 ms* | 2,499 ms* | **3.5×** |
| AWQ 4-bit | 4.2 GB | 2,959 ms* | 2,972 ms* | **3.5×** |
| GGUF Q4_K_M | 4.4 GB | — | — | **3.3×** |
| GGUF Q5_K_M | 5.1 GB | — | — | **2.8×** |
| GGUF Q8_0 | 7.7 GB | — | — | **1.9×** |

*Latency measured with `Backend.TORCH` (no Marlin JIT). Marlin / ExllamaV2 kernels would be ~2× faster. GGUF latency is for CPU/edge inference via Ollama.

---

## Stack

| Layer | Tech |
|-------|------|
| Data | HuggingFace Datasets, Pandas, Parquet |
| Fine-tuning | transformers 4.47, peft 0.14, trl 0.15.2, bitsandbytes, accelerate |
| Experiment tracking | MLflow (file-based, artifact root: S3) |
| Advanced tuning | TRL DPOTrainer, PEFT merge_and_unload |
| Quantization | gptqmodel, AutoAWQ, llama.cpp GGUF |
| Serving | vLLM (GPU), Ollama (CPU/GGUF), FastAPI |
| Monitoring | Prometheus + Grafana |
| Load testing | Locust |
| Infra | Docker Compose |

---

## How to reproduce

### Layer 1 — Data (local or free Colab)

Run `layer1_data/` notebooks 01 → 02 → 03. Needs S3 credentials (`vaidya` AWS profile locally, or env vars).

### Layer 2 — Fine-tuning (Lightning.ai A100 80GB)

1. Open `layer2_finetune/lightning/lightning_train.ipynb` in a Lightning.ai studio
2. Set GPU to **A100 80GB**
3. Run cells 1–9. Training completes in ~3 hours (680 steps, batch 64).

**Also runs on Kaggle T4 ×2** via `layer2_finetune/kaggle_train.ipynb` (~6–8 hours).

### Layer 3 — DPO + merge (Lightning.ai L4)

Open `layer3_advanced/lightning_dpo.ipynb`. Generates 500 DPO pairs from train set, trains ~30 min, merges SFT + DPO adapters, pushes to HF Hub.

### Layer 4 — Quantization (Lightning.ai L4)

Open `layer4_optimize/lightning_quantize.ipynb`. Runs GPTQ → AWQ → GGUF → benchmark table (~1 hour total).

### Layer 5 — Serving (local, Docker required)

```bash
# Start full stack
docker-compose -f layer5_production/docker-compose.yml up

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which vitamin deficiency causes night blindness?",
       "options": {"A": "Vitamin A", "B": "Vitamin B12", "C": "Vitamin C", "D": "Vitamin D"}}'
```

---

## HuggingFace model

[SneakySpidy/vaidya-mistral-7b-medmcqa](https://huggingface.co/SneakySpidy/vaidya-mistral-7b-medmcqa)

---

## Limitations

- Trained on exam questions only — not clinical decision support
- Base model: Mistral-7B-Instruct-v0.3
- Flash Attention 2 requires Ampere (CC ≥ 8.0); Kaggle T4 uses standard attention
- T4 does not have native BF16; training uses FP16 automatically on T4
- GPTQ/AWQ latency benchmarked without Marlin kernels (Lightning.ai CUDA 13 incompatibility)
