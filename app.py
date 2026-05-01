import gradio as gr
import subprocess
import os
import pandas as pd
import librosa
from huggingface_hub import hf_hub_download

# --- 1. Dataset Sanitizer Logic ---
def sanitize_dataset(max_sec):
    # Paths are mapped to the standard Docker /workspace structure
    data_dir = "/workspace/data"
    audio_dir = "/workspace/data/wavs"
    output_path = "/workspace/data/metadata.csv"
    
    # Auto-sniff for metadata files
    possible_files = ["metadata.txt", "metadata.csv", "transcript.txt", "transcript.csv"]
    transcript_path = None
    
    for f in possible_files:
        if os.path.exists(os.path.join(data_dir, f)):
            transcript_path = os.path.join(data_dir, f)
            break

    if not transcript_path:
        return "❌ **Error:** No metadata file (.txt or .csv) found in `/data` folder."
    
    try:
        yield f"🔍 Found `{os.path.basename(transcript_path)}`. Sniffing format..."
        
        # Sniff delimiter (handles | or ,)
        with open(transcript_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            delim = "|" if "|" in first_line else ","

        # Load data
        df = pd.read_csv(transcript_path, sep=delim, header=None, names=['file', 'text'])
        
        # --- CLEANUP ENGINE ---
        # Strip rogue double quotes and extra whitespace
        df['text'] = df['text'].str.replace('"', '', regex=False).str.strip()
        df['file'] = df['file'].str.strip()
        
        valid_rows = []
        skipped_long = 0
        skipped_missing = 0
        
        yield "⏱️ Scanning audio durations for VRAM safety..."
        
        for _, row in df.iterrows():
            file_key = str(row['file'])
            if not file_key.endswith('.wav'): file_key += '.wav'
            full_path = os.path.join(audio_dir, file_key)
            
            if os.path.exists(full_path):
                # Duration check prevents OOM crashes
                if librosa.get_duration(path=full_path) <= max_sec:
                    valid_rows.append(row)
                else:
                    skipped_long += 1
            else:
                skipped_missing += 1
            
        final_df = pd.DataFrame(valid_rows)
        # Force save as Pipe-Separated CSV (Piper Requirement)
        final_df.to_csv(output_path, sep="|", header=False, index=False)
        
        yield (
            f"✅ **Sanitization Complete!**\n\n"
            f"- **Ready for Training:** {len(final_df)} files\n"
            f"- **Skipped (Too Long):** {skipped_long}\n"
            f"- **Skipped (Missing Wav):** {skipped_missing}\n\n"
            f"📁 Created: `{output_path}`"
        )
    except Exception as e:
        yield f"❌ **Error during sanitization:** {str(e)}"

# --- 2. Model Download Logic ---
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
        yield f"✅ Success! Saved to: `{path}`\n\n💡 Pro-tip: Rename this to 'base_model.ckpt' to start training."
    except Exception as e:
        yield f"❌ Error: {str(e)}"

# --- Step 3: Training Tab ---
with gr.Tab("🚀 3. Training"):
    with gr.Row():
        v_name = gr.Textbox(label="Voice Name", value="bonder")
        b_size = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Batch Size")
    with gr.Row():
        # Precision Dropdown
        precision = gr.Dropdown(
            choices=["16-mixed", "32"], 
            value="16-mixed", 
            label="Trainer Precision", 
            info="Use 16-mixed for RTX GPUs; 32 for older cards/CPU."
        )
        ep = gr.Number(label="Max Epochs", value=10000)
        intv = gr.Slider(minimum=10, maximum=1000, value=50, step=10, label="Checkpoint Frequency")
    
    start_btn = gr.Button("Start Training", variant="primary")
    train_logs = gr.Textbox(label="Live Training Logs", lines=15, interactive=False)

    # Update the click function to include the precision variable
    start_btn.click(start_training, [v_name, b_size, ep, intv, precision], train_logs)

# --- Update the start_training function ---
def start_training(voice_name, batch_size, epochs, ckpt_interval, precision):
    cmd = [
        "/bin/bash", "/workspace/train.sh", 
        voice_name, str(batch_size), str(epochs), str(ckpt_interval), str(precision)
    ]
    
# --- 4. Main UI Layout (Dark Theme) ---
# We use Soft theme with Purple accent for a clean "Dev" look
with gr.Blocks(title="PVC-Pipe", theme=gr.themes.Soft(primary_hue="purple", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎙️ PVC-Pipe: Piper Voice Creation Pipeline")
    gr.Markdown("Created by **pbanj** | Privacy-First Local Voice Training")

    with gr.Tab("📥 1. Download Base"):
        gr.Markdown("### Pull Base Models from Hugging Face")
        with gr.Row():
            repo_input = gr.Textbox(label="HF Repo", value="rhasspy/piper-vits")
            file_input = gr.Textbox(label="Model Path", value="en_US/medium/base_model.ckpt")
        
        dl_btn = gr.Button("Download to /models", variant="secondary")
        dl_logs = gr.Textbox(label="Download Status", interactive=False)
        dl_btn.click(download_base_model, [repo_input, file_input], dl_logs)

    with gr.Tab("🧹 2. Dataset Sanitizer"):
        gr.Markdown("### 🛠️ Universal Metadata Sanitizer")
        gr.Markdown("Prepares your `.txt` or `.csv` for Piper. Strips quotes and enforces VRAM limits.")
        
        with gr.Row():
            max_sec = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Max Audio Duration (VRAM Safety)")
            
        san_btn = gr.Button("🔍 Scan & Sanitize /data", variant="primary")
        san_logs = gr.Markdown("Status: Waiting for input...")
        
        san_btn.click(sanitize_dataset, [max_sec], san_logs)

    with gr.Tab("🚀 3. Training"):
        gr.Markdown("### Configure & Launch Training")
        with gr.Row():
            voice_name = gr.Textbox(label="Voice Name", value="bonder")
            batch_size = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Batch Size (4050 6GB = 8)")
        
        with gr.Row():
            epochs = gr.Number(label="Max Epochs", value=10000)
            checkpoint_interval = gr.Slider(minimum=10, maximum=1000, value=50, step=10, label="Checkpoint Frequency")
        
        start_btn = gr.Button("Start Training", variant="primary")
        train_logs = gr.Textbox(label="Live Training Logs", lines=15, interactive=False)

        start_btn.click(start_training, [voice_name, batch_size, epochs, checkpoint_interval], train_logs)

# 0.0.0.0 allows access from outside the Docker container
demo.queue().launch(server_name="0.0.0.0", server_port=7860)
