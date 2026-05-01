FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system tools.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    pkg-config \
    autoconf \
    automake \
    libtool \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# 2. Install MODERN CMake + Build Tools via Pip.
RUN pip3 install --no-cache-dir cmake Cython scikit-build setuptools wheel

WORKDIR /workspace
COPY . .

# 3. Clone the piper repo.
RUN git clone https://github.com/OHF-Voice/piper1-gpl.git /workspace/piper1-gpl

# 4. Build.
WORKDIR /workspace/piper1-gpl
RUN pip3 install -e . && \
    python3 setup.py build_ext --inplace

WORKDIR /workspace
RUN pip3 install --no-cache-dir -r requirements.txt
RUN chmod +x /workspace/entrypoint.sh /workspace/train.sh

ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
