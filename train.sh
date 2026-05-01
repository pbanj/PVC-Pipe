#!/bin/bash
set -e

# These variables pull from the Gradio UI 
VOICE_NAME=${1:-"bonder"}
BATCH_SIZE=${2:-8}
MAX_EPOCHS=${3:-10000}
CKPT_INTERVAL=${4:-50}
PRECISION=${5:-"16-mixed"}

echo "------------------------------------------------"
echo "🔥 PVC-Pipe: Starting Piper Training"
echo "------------------------------------------------"
echo "🎙️ Voice Name: $VOICE_NAME"
echo "⚙️  Precision:  $PRECISION"
echo "📦 Batch Size: $BATCH_SIZE"
echo "🎯 Max Epochs: $MAX_EPOCHS"
echo "💾 Checkpoint: Every $CKPT_INTERVAL epochs"
echo "------------------------------------------------"

# Change directory to the Piper source
cd /workspace/piper1-gpl

python3 -m piper.train fit \
  --data.voice_name "$VOICE_NAME" \
  --data.csv_path "/workspace/data/metadata.csv" \
  --data.audio_dir "/workspace/data/wavs" \
  --data.cache_dir "/workspace/data/cache" \
  --data.espeak_voice "en-us" \
  --data.config_path "/workspace/piper1-gpl/etc/vits/en_US/medium/config.json" \
  --model.sample_rate 22050 \
  --data.batch_size "$BATCH_SIZE" \
  --trainer.devices 1 \
  --trainer.max_epochs "$MAX_EPOCHS" \
  --trainer.precision "$PRECISION" \
  --trainer.default_root_dir "/workspace/models/$VOICE_NAME" \
  --trainer.check_val_every_n_epoch "$CKPT_INTERVAL" \
  --trainer.log_every_n_steps 1 \
  --ckpt_path "/workspace/models/base_model.ckpt"
