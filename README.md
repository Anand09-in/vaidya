# Vaidya (वैद्य)

**End-to-end LLM fine-tuning pipeline for Indian medical Q&A**

QLoRA fine-tuned Mistral-7B on 187k Indian medical exam questions (MedMCQA), with DPO preference tuning, three quantization formats (GPTQ/AWQ/GGUF), vLLM serving, and a Prometheus + Grafana monitoring stack.

> MedMCQA accuracy: **Base → X%** | **Fine-tuned → Y%**

---

## Architecture

```
Raw MedMCQA (187k)
  └── Layer 1: Data Pipeline       → Clean Parquet splits + quality report
        └── Layer 2: QLoRA Training → Mistral-7B + LoRA adapter (MLflow tracked)
              └── Layer 3: DPO      → Preference tuning + merged model on HF Hub
                    └── Layer 4: Quantization → GPTQ / AWQ / GGUF benchmark table
                          └── Layer 5: Production → vLLM + FastAPI + Prometheus + Grafana
```

---

## Repo structure

```
vaidya/
├── layer1_data/          # Data pipeline notebooks
├── layer2_finetune/      # QLoRA training scripts
├── layer3_advanced/      # DPO training + adapter merge
├── layer4_optimize/      # Quantization + profiling
├── layer5_production/    # vLLM serving + monitoring + load tests
├── eval/                 # Accuracy eval + regression check
├── scripts/              # One-time S3 setup
└── .github/workflows/    # CI regression check
```

Data lives in **S3 (`s3://vaidya-artifacts/`)** — not in this repo.

---

## Stack

| Layer | Tech |
|-------|------|
| Data | HuggingFace datasets, Pandas, Parquet |
| Fine-tuning | transformers, peft, trl, bitsandbytes, accelerate |
| Tracking | MLflow (SQLite locally, artifact root: S3) |
| Advanced tuning | TRL DPOTrainer, PEFT merge_and_unload |
| Quantization | AutoGPTQ, AutoAWQ, llama.cpp GGUF |
| Profiling | torch.profiler, pynvml |
| Serving | vLLM (GPU), Ollama (CPU/GGUF) |
| API | FastAPI |
| Monitoring | Prometheus + Grafana |
| Load testing | Locust |
| Infra | Docker Compose |

---

## Benchmark table (fill after Layer 4)

| Format | Size | Latency p50 | Latency p95 | Throughput (tok/s) | MedMCQA Acc |
|--------|------|-------------|-------------|-------------------|-------------|
| Full BF16 | 14 GB | - | - | - | -% |
| GPTQ 4-bit | ~4 GB | - | - | - | -% |
| AWQ 4-bit | ~4 GB | - | - | - | -% |
| GGUF Q4_K_M | ~4 GB | - | - | - | -% |
| GGUF Q8_0 | ~7.7 GB | - | - | - | -% |

---

## Quickstart (API)

```bash
# Pull GGUF model (after Layer 4)
aws s3 cp s3://vaidya-artifacts/models/gguf/vaidya-Q4_K_M.gguf ./models/

# Start full stack
docker-compose -f layer5_production/docker-compose.yml up

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which vitamin deficiency causes night blindness? A. Vitamin A  B. Vitamin B12  C. Vitamin C  D. Vitamin D"}'
```

---

## How to reproduce

### Layer 1 — Data (local or Colab free tier)
Run `layer1_data/` notebooks in order. Needs: S3 credentials (`vaidya` AWS profile locally, or env vars on Colab).

### Layer 2 — Fine-tuning (Kaggle T4 ×2)
1. Upload `layer2_finetune/train.py` and `layer2_finetune/qlora_config.py` as a Kaggle Dataset input
2. Open `layer2_finetune/kaggle_train.ipynb` on Kaggle
3. Settings → Accelerator → **GPU T4 ×2**, Internet → **On**
4. Add-ons → Secrets → add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
5. Run all cells — expected ~6–8 hours

**Environment note:** Kaggle runs Python 3.13 + CUDA 12.8. The notebook installs compatible package versions automatically.

See each layer's folder for detailed instructions.

---

## HuggingFace model
[Anand09-in/vaidya-mistral-7b-medmcqa](https://huggingface.co/Anand09-in/vaidya-mistral-7b-medmcqa)

---

## Limitations
- Trained on exam questions only — not clinical decision support
- Base model: Mistral-7B-Instruct-v0.3
- Flash Attention 2 requires Ampere (CC ≥ 8.0); Kaggle T4 (CC 7.5) uses standard attention
- T4 does not have native BF16 hardware; training uses FP16 automatically
