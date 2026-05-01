#!/bin/bash
set -e

VOICE_NAME=${1:-"bonder"}
BATCH_SIZE=${2:-8}
MAX_EPOCHS=${3:-10000}
CKPT_INTERVAL=${4:-50}

python3 -m piper.train fit \
  --data.voice_name "$VOICE_NAME" \
  --data.csv_path "/workspace/data/metadata.csv" \
  --data.audio_dir "/workspace/data" \
  --data.cache_dir "/workspace/data/cache" \
  --data.espeak_voice "en-us" \
  --data.config_path "/workspace/piper1-gpl/etc/vits/en_US/medium/config.json" \
  --model.sample_rate 22050 \
  --data.batch_size "$BATCH_SIZE" \
  --trainer.devices 1 \
  --trainer.max_epochs "$MAX_EPOCHS" \
  --trainer.precision "16-mixed" \
  --trainer.default_root_dir "/workspace/models/$VOICE_NAME" \
  --trainer.check_val_every_n_epoch "$CKPT_INTERVAL" \
  --trainer.log_every_n_steps 1 \
  --trainer.callbacks lightning.pytorch.callbacks.ModelCheckpoint \
  --trainer.callbacks.save_top_k 3 \
  --trainer.callbacks.monitor "val_loss" \
  --trainer.callbacks.mode "min" \
  --trainer.callbacks.filename "epoch_{epoch:04d}-loss_{val_loss:.4f}" \
  --trainer.callbacks.save_last true \
  --ckpt_path "/workspace/models/base_model.ckpt"
