FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Fix for potential "command not found" - link python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Clone the OHF repo
RUN git clone https://github.com/OHF-Voice/piper1-gpl.git

COPY . .

RUN pip3 install --no-cache-dir Cython wheel setuptools


RUN pip3 install --no-cache-dir -r requirements.txt

RUN cd /workspace/piper1-gpl && \
    pip3 install -e . && \
    python3 setup.py build_ext --inplace
RUN cd /workspace/piper1-gpl && \
    pip3 install -e . && \
    /bin/bash build_monotonic_align.sh && \
    python3 setup.py build_ext --inplace

RUN chmod +x /workspace/entrypoint.sh /workspace/train.sh

ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
