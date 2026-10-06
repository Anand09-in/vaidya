#!/usr/bin/env bash
# Vaidya — Phase 5: Convert merged model to GGUF via llama.cpp.
#
# Usage:
#   bash convert_gguf.sh [--quant Q4_K_M|Q5_K_M|Q8_0]
#
# Requirements:
#   git, cmake, python3 (with transformers, sentencepiece)
#   AWS CLI configured (for S3 upload)
#
# Output:
#   ./gguf-models/vaidya-Q4_K_M.gguf  (and other quants)

set -euo pipefail

QUANT="${1:-Q4_K_M}"
MERGED_DIR="../layer3_advanced/merged_model"
OUT_DIR="./gguf-models"
LLAMA_DIR="./llama.cpp"

# ── 1. Pull merged model from S3 if missing ────────────────────────────────
if [ ! -d "$MERGED_DIR" ]; then
    echo "Downloading merged model from S3..."
    python3 - <<'PY'
import sys, os, boto3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'layer2_finetune'))
import qlora_config as C
from pathlib import Path

s3 = boto3.client('s3')
bucket = C.S3_BUCKET.replace('s3://', '')
dest_root = Path('../layer3_advanced/merged_model')
dest_root.mkdir(parents=True, exist_ok=True)
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket, Prefix='models/merged/'):
    for obj in page.get('Contents', []):
        key = obj['Key']; fname = key.replace('models/merged/', '')
        if fname:
            dest = dest_root / fname; dest.parent.mkdir(parents=True, exist_ok=True)
            print(f'  Downloading {key}...')
            s3.download_file(bucket, key, str(dest))
print('Done.')
PY
fi

# ── 2. Clone + build llama.cpp if needed ───────────────────────────────────
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

echo "Building llama.cpp (cmake)..."
cd "$LLAMA_DIR"
cmake -B build -DLLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native 2>/dev/null \
  || cmake -B build -DCMAKE_BUILD_TYPE=Release  # CPU fallback if no CUDA toolkit
cmake --build build --config Release -j "$(nproc)" --target llama-quantize convert_hf_to_gguf 2>&1 | tail -5
cd ..

# ── 3. Install conversion deps ─────────────────────────────────────────────
pip install -q gguf sentencepiece transformers

# ── 4. Convert to GGUF F16 ─────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
GGUF_F16="$OUT_DIR/vaidya-f16.gguf"
echo "Converting HF → GGUF F16..."
python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outtype f16 \
    --outfile "$GGUF_F16"
echo "F16 GGUF: $GGUF_F16 ($(du -h "$GGUF_F16" | cut -f1))"

# ── 5. Quantize to target format ───────────────────────────────────────────
GGUF_QUANT="$OUT_DIR/vaidya-${QUANT}.gguf"
echo "Quantizing to $QUANT..."
"$LLAMA_DIR/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_QUANT" "$QUANT"
echo "$QUANT GGUF: $GGUF_QUANT ($(du -h "$GGUF_QUANT" | cut -f1))"

# Extra quants for benchmark table
for Q in Q5_K_M Q8_0; do
    if [ "$Q" != "$QUANT" ]; then
        OUT="$OUT_DIR/vaidya-${Q}.gguf"
        echo "Quantizing to $Q..."
        "$LLAMA_DIR/build/bin/llama-quantize" "$GGUF_F16" "$OUT" "$Q" \
          && echo "$Q GGUF: $OUT ($(du -h "$OUT" | cut -f1))" \
          || echo "⚠️  $Q failed — skipping"
    fi
done

# ── 6. Upload to S3 ────────────────────────────────────────────────────────
echo "Uploading GGUF models to S3..."
python3 - <<PY
import sys, os, boto3
sys.path.insert(0, os.path.join(os.path.dirname('$0'), '..', 'layer2_finetune'))
import qlora_config as C
from pathlib import Path

s3 = boto3.client('s3')
bucket = C.S3_BUCKET.replace('s3://', '')
out_dir = Path('$OUT_DIR')
uploaded = 0
for f in out_dir.glob('*.gguf'):
    key = f'models/gguf/{f.name}'
    print(f'  Uploading {f.name}...')
    s3.upload_file(str(f), bucket, key)
    uploaded += 1
print(f'Uploaded {uploaded} GGUF file(s) → {C.S3_BUCKET}/models/gguf/')
PY

echo ""
echo "✅ GGUF conversion complete."
echo "   Files: $OUT_DIR"
echo "   S3   : s3://vaidya-artifacts/models/gguf/"
echo "   Next : run benchmark.py to compare all formats"
