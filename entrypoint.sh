#!/bin/bash
# Stop on error
set -e

echo "------------------------------------------------"
echo "🎙️ PVC-Pipe: Piper Voice Creation Suite"
echo "------------------------------------------------"

# 1. Directory Setup
mkdir -p /workspace/data/wavs /workspace/models

# 2. Permissions Check
if [ ! -w "/workspace/models" ]; then
    echo "⚠️  WARNING: /workspace/models is not writable. Fix permissions on your host."
fi

# 3. Environment Check
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN is not set. Hugging Face downloads will be restricted."
else
    echo "✅ HF_TOKEN detected."
fi

# 4. Path Enforcement
export PYTHONPATH="/workspace/piper1-gpl/src:$PYTHONPATH"

# 5. Hand off to the Python UI
echo "🚀 Launching Gradio UI..."

exec python3 /workspace/app.py