FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system tools and build-essential for C++ compilation
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Clone the official repository so it's guaranteed to be there
RUN git clone https://github.com/rhasspy/piper1-gpl.git

# 3. Copy your specific UI, train.sh, and entrypoint.sh
COPY . .

# 4. Compile the C++ extensions (The "Bonder" heart)
RUN cd piper1-gpl && \
    python3 -m pip install -e . && \
    bash build_monotonic_align.sh && \
    python3 setup.py build_ext --inplace

# 5. Install UI dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Ensure scripts are executable (Just in case Git stripped permissions)
RUN chmod +x /workspace/entrypoint.sh /workspace/train.sh

ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
