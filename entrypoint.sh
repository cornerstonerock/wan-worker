#!/bin/bash
# Container entrypoint: link network-volume models, boot ComfyUI, start worker.
set -e

# If a RunPod network volume is attached at /runpod-volume with a models/
# tree, symlink it into ComfyUI's model dirs (volume wins over anything local).
if [ -d "/runpod-volume/models" ]; then
  for d in diffusion_models loras text_encoders vae; do
    if [ -d "/runpod-volume/models/$d" ]; then
      rm -rf "/comfyui/models/$d"
      ln -sfn "/runpod-volume/models/$d" "/comfyui/models/$d"
      echo "[entrypoint] linked /runpod-volume/models/$d"
    fi
  done
fi

# Sanity check: the worker needs its model files before accepting jobs.
missing=0
for f in \
  "diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" \
  "diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" \
  "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
  "diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
  "loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors" \
  "loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors" \
  "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" \
  "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" \
  "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "vae/wan_2.1_vae.safetensors" ; do
  if [ ! -f "/comfyui/models/$f" ]; then
    echo "[entrypoint] MISSING MODEL: /comfyui/models/$f" >&2
    missing=1
  fi
done
if [ "$missing" = "1" ]; then
  echo "[entrypoint] One or more models are missing. Populate the network volume" >&2
  echo "[entrypoint] with download_models.sh, then restart the endpoint." >&2
  exit 1
fi

# Boot ComfyUI in API mode (no UI needed on serverless).
cd /comfyui
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!

# Wait for the API to answer (up to ~5 min on slow cold starts).
echo "[entrypoint] waiting for ComfyUI..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
    echo "[entrypoint] ComfyUI is up"
    break
  fi
  if ! kill -0 $COMFY_PID 2>/dev/null; then
    echo "[entrypoint] ComfyUI died during startup. Log tail:" >&2
    tail -50 /tmp/comfyui.log >&2
    exit 1
  fi
  sleep 5
done

# Hand off to the RunPod serverless loop (foreground).
exec python /app/rp_handler.py
