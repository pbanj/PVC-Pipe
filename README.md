# PVC-Pipe: Pbanjs Voice Creation Pipeline.   

I'm back with more horrible puns :D  
This repo serves as a home for the files for training piper voices in Google colab or locally.  
Full disclosure, had geminni help with parts. Mostly formatting and estimates.  
AI is really great at readmes so that's nice at least. a bit emoji heavy but meh.  
I did this because of the hell of dealing with AI trying to get it working. No one should have to argue as much with an AI as I did through all of this.  
It doesnt know I added this ~~as I didn't want it to start delelting things because it felt like a failure.~~  
It deleted shit out of the files anyways and then argued about it when I added them back.


---  


### 💻 Hardware Reference (Local Benchmark)
Benchmark based on: **i7-12650H | 64GB RAM | RTX 4050 (6GB VRAM)**

| Preset | Batch Size | Epochs | Local Time (Estimated) |
| :--- | :--- | :--- | :--- |
| **Medium Quality** | 16 | 5,000 | **~5-7 Hours** |
| **High Quality** | 16 | 8,000 | **~8-10 Hours** |
| **Production Ready** | 8 | 10,000 | **~10-14 Hours** |  

*Note: While local training is free and has no idle timeouts it can be slow. Start a Production run before bed, and it's ready by morning!*  

### 💡 Recommended Settings for Colab (Free Tier)
Colab's T4 GPU is quite capable. Use these presets to balance speed and quality:

| Target Quality | Batch Size | Epochs | Time (Colab T4) | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Fast Prototype** | 32 | 2,000 | **~45 - 60 mins** | Quick check to see if your dataset is working. |
| **Medium Quality** | 16 | 5,000 | **~3 - 4 hours** | **The Sweet Spot.** Great for Home Assistant. |
| **High Quality** | 16 | 8,000 | **~5 - 6 hours** | Solid quality, great for most applications. |
| **Production Ready** | 8 | 10,000 | **~8 - 10 hours** | The Piper Gold Standard. Maximum clarity. |  

> **Tip:** If you get an `Out of Memory (OOM)` error in Colab, drop the **Batch Size** to 8.  
> "Production Ready" isn't recommended on the free tier as it'll most likely time out.  
> But the notbook has a built in resume system so you wont lose your data and can start from where it left off.  

### 💡 Quality Presets & Benchmarks (Cloud)
Choosing the right number of epochs depends on your goal. Here is how they stack up on a standard Colab T4:

| Preset | Batch Size | Epochs | Time (Colab) | Goal |
| :--- | :--- | :--- | :--- | :--- |
| **Fast Prototype** | 32 | 2,000 | ~45 mins | Quick test of audio/text alignment. |
| **Medium Quality** | 16 | 5,000 | ~3 hours | **The Sweet Spot.** Great for Home Assistant. |
| **High Quality** | 16 | 8,000 | ~5 hours | Near-production results with less wait. |
| **Production Ready** | 8 | 10,000 | ~8 hours | **The Gold Standard.** Maximum clarity. |

*Estimates based on a 20-minute clean dataset trained to 10,000 epochs.*

#### **✨ Why "Medium" is the Recommended Start:**
- **Diminishing Returns:** You get about 90% of the voice quality by 5,000 epochs. The leap to 10,000 is for fine nuances.
- **Efficiency:** It sounds significantly better than "Fast" but saves you 5+ hours of compute time.
- **VRAM Stability:** Batch Size 16 is extremely stable and rarely triggers OOM errors.
- **Storage Efficiency:** At ~65MB, it's half the size of a "High" model but sounds almost identical for short assistant replies.

| Version | Link | Description |
| :--- | :--- | :--- |
| **Stable** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Hs2HzfNJhYBUvfl8DIzF0YmZP9tsSpQm?usp=sharing) | Static version that works. |
| **Dev** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pbanj/PVC-Pipe/blob/main/PVC_Pipe_Trainer.ipynb) | Latest code synced directly from this repo. Might not work.|


### 📦 Model Tiers & File Sizes
The final exported `.onnx` file size depends on the quality tier of the base model you chose. Below are the typical sizes and characteristics:

* **X-Low / Low (~16MB – 28MB)**
  * **Audio:** 16kHz sample rate.
  * **Characteristics:** Uses very few "parameters" (internal math connections). While extremely fast and lightweight, the output tends to sound robotic.
* **Medium (~60MB – 65MB)**
  * **Audio:** 22kHz sample rate (much clearer).
  * **Characteristics:** The "Goldilocks" tier. It maintains a relatively small "brain" to ensure fast execution on CPUs (like a Raspberry Pi 4) while sounding significantly more natural.
* **High / Production (~110MB – 120MB)**
  * **Audio:** 22kHz sample rate.
  * **Characteristics:** Uses the same audio quality as Medium but nearly **doubles** the internal parameters. This allows the AI to capture complex nuances and emotional inflections, though it is slightly slower to run on low-end hardware.  

> **Note:** These sizes refer to the final exported `.onnx` voice model. During training, the `.ckpt` checkpoint files in your storage will be much larger (300MB – 800MB) as they contain the full training state and optimizer data.

## 🛠️ Quick Start (Local Docker)

PVC-Pipe is designed to run entirely on your own hardware using Docker.

### 1. Prerequisites  
Don't use the default package manager versions! These can be ancient or missing the GPU hooks. Follow these official guides for your specific OS:

#### **📦 For Ubuntu/Debian (`apt`)** - **[Docker Engine (Official Install)](https://docs.docker.com/engine/install/ubuntu/)**: Use the official Docker repo for the latest `docker-ce`.
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)**: Essential for GPU passthrough on Ubuntu.

#### **📦 For Fedora/RHEL (`dnf`)** - **[Docker Engine (Official Fedora Install)](https://docs.docker.com/engine/install/fedora/)**: Fedora defaults to Podman; use this guide to install the official Docker Engine (`docker-ce`) instead.
- **[NVIDIA Container Toolkit (dnf)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-dnf)**: Requires adding the NVIDIA `rpm` repository to your `dnf` config.

Fedora 41+ is very aggressive about **cgroups v2** and **SELinux**. If You run into an issue where the container can't see the GPU even after following the guide. Ensure the `nvidia-container-toolkit-selinux` package is installed (it usually is by default from the NVIDIA repo).

#### **🚀 Post-Install (Mandatory for All)**
- **[Linux Post-Install Steps](https://docs.docker.com/engine/install/linux-postinstall/)**: Follow this to add your user to the `docker` group. If you skip this, `docker compose` will fail with permission errors.
- **Restart Docker**: After installing the NVIDIA toolkit, run:
  ```bash
  sudo systemctl restart docker

### 2. Launch the Pipeline  
1. Create a project folder.
2. Ensure you have your `/data` (wavs + metadata) and `/models` (base checkpoint) ready.
3. Drop the `docker-compose.yml` into the folder and run:

```bash
docker compose up -d
