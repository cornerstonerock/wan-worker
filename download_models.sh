#!/bin/bash
# One-time model fetch: populates a RunPod network volume with every file the
# worker needs. Run ONCE from any GPU pod with the volume attached, e.g.:
#
#   ./download_models.sh /runpod-volume/models
#
# Total download is ~64 GB — allow 20-40 min on a fast link.
# All files come from Comfy-Org/Wan_2.2_ComfyUI_Repackaged (Apache-2.0).
set -e

DEST="$(realpath -m "${1:-/runpod-volume/models}")"
# Safety guard: never download 64GB to the pod's small container disk.
# If DEST doesn't resolve under /runpod-volume, fail fast instead of
# silently filling / and burning GPU hours.
case "$DEST" in
  /runpod-volume/*) ;;
  *) echo "ERROR: refusing to download outside /runpod-volume (got: $DEST)" >&2; exit 1 ;;
esac
echo "Downloading to: $DEST"
BASE="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"

mkdir -p "$DEST"/diffusion_models "$DEST"/loras "$DEST"/text_encoders "$DEST"/vae

fetch() { # $1 = subdir, $2 = filename
  local out="$DEST/$1/$2"
  if [ -f "$out" ]; then
    echo "[skip] $2 already present"
    return
  fi
  echo "[fetch] $2"
  # curl, not aria2c: RunPod network volumes report bogus statvfs numbers
  # (and lack fallocate), which makes aria2c abort every download.
  curl -fL --retry 3 --retry-delay 5 --progress-bar -o "$out" "$BASE/$1/$2"
}

# --- Diffusion models: Wan2.2 14B, fp8-scaled (~13.3 GB each) ---
fetch diffusion_models wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
fetch diffusion_models wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors
fetch diffusion_models wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
fetch diffusion_models wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors

# --- LightX2V 4-step LoRAs (~1.1 GB each) ---
fetch loras wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors
fetch loras wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors
fetch loras wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
fetch loras wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors

# --- Text encoder + VAE ---
fetch text_encoders umt5_xxl_fp8_e4m3fn_scaled.safetensors
fetch vae wan_2.1_vae.safetensors

echo "Done. $(du -sh "$DEST" | cut -f1) in $DEST"
ls -la "$DEST"/diffusion_models "$DEST"/loras "$DEST"/text_encoders "$DEST"/vae
