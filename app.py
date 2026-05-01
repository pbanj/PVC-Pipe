import gradio as gr
import subprocess
import os

def start_training(voice_name, batch_size, epochs, checkpoint_interval):
    # This function constructs the command and runs your bash script
    cmd = [
        "/bin/bash", "/workspace/train.sh", 
        voice_name, str(batch_size), str(epochs), str(checkpoint_interval)
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Yield the output to Gradio in real-time
    output = ""
    for line in process.stdout:
        output += line
        yield output

with gr.Blocks(title="PVC-Pipe: Piper Voice Creation") as demo:
    gr.Markdown("# 🎙️ PVC-Pipe: Piper Voice Creation Pipeline")
    
    with gr.Row():
        voice_name = gr.Textbox(label="Voice Name", value="bonder")
        batch_size = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Batch Size")
        
    with gr.Row():
        epochs = gr.Number(label="Max Epochs", value=10000)
        ckpt_interval = gr.Slider(minimum=10, maximum=1000, value=50, step=10, label="Checkpoint Every X Epochs")
    
    start_btn = gr.Button("🚀 Start Training", variant="primary")
    logs = gr.Textbox(label="Training Logs", lines=15, interactive=False)

    start_btn.click(start_training, [voice_name, batch_size, epochs, ckpt_interval], logs)

demo.queue().launch(server_name="0.0.0.0", server_port=7860)
