import gradio as gr
import subprocess
import os
from huggingface_hub import hf_hub_download

# --- Logic for Download Tab ---
def download_base_model(repo_id, filename):
    token = os.getenv("HF_TOKEN")
    try:
        yield f"⏳ Authenticating and downloading {filename}..."
        path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            local_dir="/workspace/models",
            token=token,
            local_dir_use_symlinks=False
        )
        yield f"✅ Success! Model saved to: {path}\n\n💡 Pro-tip: Rename this file to 'base_model.ckpt' in your local folder to use it for training."
    except Exception as e:
        yield f"❌ Error: {str(e)}"

# --- Logic for Training Tab ---
def start_training(voice_name, batch_size, epochs, checkpoint_interval):
    # Construct the command for your bash script
    cmd = [
        "/bin/bash", "/workspace/train.sh", 
        voice_name, str(batch_size), str(epochs), str(checkpoint_interval)
    ]
    
    # Run the process and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output = ""
    for line in process.stdout:
        output += line
        yield output

# --- Main UI Layout ---
with gr.Blocks(title="PVC-Pipe: Piper Voice Creation") as demo:
    gr.Markdown("# 🎙️ PVC-Pipe: Piper Voice Creation Pipeline")
    gr.Markdown("Created by **pbanj** | Locally hosted, privacy-first voice training.")

    with gr.Tab("🛠️ 1. Setup & Download"):
        gr.Markdown("### Pull Base Models from Hugging Face")
        with gr.Row():
            repo_input = gr.Textbox(label="Hugging Face Repo", value="rhasspy/piper-vits")
            file_input = gr.Textbox(label="Model File (.ckpt)", value="en_US/medium/base_model.ckpt")
        
        dl_btn = gr.Button("Download to /models", variant="secondary")
        dl_logs = gr.Textbox(label="Status", interactive=False, lines=3)
        
        dl_btn.click(download_base_model, [repo_input, file_input], dl_logs)

    with gr.Tab("🚀 2. Training"):
        gr.Markdown("### Configure & Launch Training")
        with gr.Row():
            voice_name = gr.Textbox(label="Voice Name", value="bonder")
            batch_size = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Batch Size (6GB VRAM = 8)")
        
        with gr.Row():
            epochs = gr.Number(label="Max Epochs", value=10000)
            ckpt_interval = gr.Slider(minimum=10, maximum=1000, value=50, step=10, label="Checkpoint Frequency")
        
        start_btn = gr.Button("Start Training", variant="primary")
        train_logs = gr.Textbox(label="Training Output", lines=15, interactive=False)

        start_btn.click(start_training, [voice_name, batch_size, epochs, ckpt_interval], train_logs)

# Launch with server_name 0.0.0.0 so it is accessible outside the container
demo.queue().launch(server_name="0.0.0.0", server_port=7860)
