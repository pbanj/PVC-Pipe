FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git python3 python3-pip build-essential cmake ninja-build espeak-ng libsndfile1-dev wget zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/OHF-Voice/piper1-gpl.git
WORKDIR /workspace/piper1-gpl

# Install dependencies
RUN pip3 install --no-cache-dir -e '.[train]' lightning==2.1.0 gradio scikit-build cython librosa pandas tqdm
RUN bash build_monotonic_align.sh && python setup.py build_ext --inplace
RUN cp espeakbridge*.so src/piper/espeakbridge.so || true

# PyTorch 2.6 patch
RUN sed -i '1i import torch\nimport pathlib\nif hasattr(torch.serialization, "add_safe_globals"):\n    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])\n' src/piper/__init__.py

ENV PYTHONPATH="/workspace/piper1-gpl/src:${PYTHONPATH}"

# --- BAKE THE CODE IN ---
# These files must exist in the root of your PVC-Pipe repo
COPY train.sh /workspace/train.sh
COPY app.py /workspace/app.py
RUN chmod +x /workspace/train.sh

# Start the Web UI directly
CMD ["python3", "/workspace/app.py"]
