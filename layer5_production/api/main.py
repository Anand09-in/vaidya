"""
Vaidya — Phase 6: FastAPI serving layer.

Routes /ask to vLLM (GPU) or Ollama (CPU/GGUF) based on BACKEND env var.
Exposes /metrics for Prometheus scraping.
"""

__version__ = "1.0"

import os, re, time, logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
BACKEND      = os.getenv("BACKEND", "vllm")          # "vllm" | "ollama"
VLLM_URL     = os.getenv("VLLM_URL",  "http://vllm:8000/v1")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "vaidya-q4")

SYSTEM_PROMPT = (
    "You are Vaidya, an expert in Indian medical licensing exams (AIIMS, PGI, USMLE-equivalent). "
    "Answer the question by selecting the single best option. "
    "Reply with just the letter: A, B, C, or D."
)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "vaidya_requests_total", "Total requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "vaidya_request_latency_seconds", "Request latency",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
TOKENS_GENERATED = Counter(
    "vaidya_tokens_generated_total", "Total tokens generated"
)
ACTIVE_REQUESTS = Gauge(
    "vaidya_active_requests", "Currently active requests"
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    options:  dict[str, str]   # {"A": "...", "B": "...", "C": "...", "D": "..."}

class AskResponse(BaseModel):
    answer:     str            # "A" | "B" | "C" | "D"
    answer_text: str
    latency_ms: float
    backend:    str


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(req: AskRequest) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in req.options.items())
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question: {req.question}\n\nOptions:\n{opts}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_answer(text: str) -> str:
    text = text.strip().upper()
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else text[:1] if text else "?"


# ── Inference backends ────────────────────────────────────────────────────────
async def infer_vllm(prompt: str, client: httpx.AsyncClient) -> tuple[str, int]:
    resp = await client.post(
        f"{VLLM_URL}/completions",
        json={
            "model":       "vaidya",
            "prompt":      prompt,
            "max_tokens":  8,
            "temperature": 0.0,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    text   = data["choices"][0]["text"]
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return text, tokens


async def infer_ollama(prompt: str, client: httpx.AsyncClient) -> tuple[str, int]:
    resp = await client.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 8},
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data   = resp.json()
    text   = data.get("response", "")
    tokens = data.get("eval_count", 0)
    return text, tokens


# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Vaidya API v%s | backend=%s", __version__, BACKEND)
    async with httpx.AsyncClient() as client:
        app.state.http = client
        yield


app = FastAPI(title="Vaidya API", version=__version__, lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND, "version": __version__}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    try:
        prompt = build_prompt(req)

        if BACKEND == "vllm":
            raw, n_tokens = await infer_vllm(prompt, app.state.http)
        elif BACKEND == "ollama":
            raw, n_tokens = await infer_ollama(prompt, app.state.http)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown backend: {BACKEND}")

        answer = parse_answer(raw)
        latency = (time.perf_counter() - t0) * 1000

        TOKENS_GENERATED.inc(n_tokens)
        REQUEST_COUNT.labels(endpoint="/ask", status="ok").inc()
        REQUEST_LATENCY.labels(endpoint="/ask").observe(latency / 1000)

        return AskResponse(
            answer=answer,
            answer_text=req.options.get(answer, ""),
            latency_ms=round(latency, 1),
            backend=BACKEND,
        )

    except httpx.HTTPStatusError as e:
        REQUEST_COUNT.labels(endpoint="/ask", status="error").inc()
        raise HTTPException(status_code=502, detail=f"Backend error: {e.response.status_code}")
    except httpx.RequestError as e:
        REQUEST_COUNT.labels(endpoint="/ask", status="error").inc()
        raise HTTPException(status_code=503, detail=f"Backend unreachable: {e}")
    finally:
        ACTIVE_REQUESTS.dec()
