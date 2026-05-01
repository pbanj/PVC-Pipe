FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/OHF-Voice/piper1-gpl.git

COPY . .

RUN cd piper1-gpl && \
    python3 -m pip install -e . && \
    bash build_monotonic_align.sh && \
    python3 setup.py build_ext --inplace

RUN pip3 install --no-cache-dir -r requirements.txt

RUN chmod +x /workspace/entrypoint.sh /workspace/train.sh

ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
