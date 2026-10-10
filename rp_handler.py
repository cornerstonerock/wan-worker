"""
Wan 2.2 RunPod Serverless worker — text-to-video + image-to-video.

Implements the PulseClone worker contract (see README.md):
  input:  { prompt, negative_prompt?, image_url?, num_frames?,
            width?, height?, fps?, num_inference_steps?, guidance_scale?, seed? }
  output: { video: "data:video/mp4;base64,...", metadata: {...} }

Architecture: ComfyUI (API mode) running the official ComfyUI Wan2.2 14B
workflow structure — dual UNETs (high/low noise fp8) + LightX2V 4-step
LoRAs, KSamplerAdvanced cascade (steps 0-2 high noise, 2-4 low noise),
cfg=1.0, euler/simple, ModelSamplingSD3 shift=5.0.

When image_url is present the start frame is VAE-encoded via
Wan22ImageToVideoLatent (image-to-video); otherwise EmptyHunyuanLatentVideo
(text-to-video).
"""

import base64
import io
import json
import os
import random
import time
import uuid

import requests
from PIL import Image

import runpod

COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", "/comfyui/input")
OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT_DIR", "/comfyui/output")

# ----------------------------------------------------------------------------
# Models — Comfy-Org/Wan_2.2_ComfyUI_Repackaged (verified file names)
# ----------------------------------------------------------------------------
REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

T2V_HIGH = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
T2V_LOW = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
T2V_LORA_HIGH = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
T2V_LORA_LOW = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"

I2V_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
I2V_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
I2V_LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
I2V_LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

CLIP_FILE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_FILE = "wan_2.1_vae.safetensors"  # matches the official ComfyUI template

# Lightning LoRAs are distilled for exactly 4 NFE — steps/cfg are fixed.
STEPS = 4
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "simple"
SHIFT = 5.0

DEFAULT_NEGATIVE = (
    "low quality, blurry, watermark, distorted, deformed, bad anatomy, "
    "extra fingers, bad hands, bad face, disfigured"
)

# Poll the local ComfyUI queue for this long before giving up.
GENERATION_TIMEOUT_S = int(os.environ.get("GENERATION_TIMEOUT_S", "1200"))


class WorkerError(Exception):
    pass


# ----------------------------------------------------------------------------
# Input validation / normalization
# ----------------------------------------------------------------------------
def snap_frames(n):
    n = max(9, min(121, int(n)))
    return 8 * round((n - 1) / 8) + 1  # Wan requires 8n+1


def snap_dim(n):
    n = max(256, min(1024, int(n)))
    return (n // 16) * 16


def validate(event_input):
    if not isinstance(event_input, dict):
        raise WorkerError("input must be a JSON object")
    prompt = event_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkerError("prompt is required and must be a non-empty string")
    if len(prompt) > 2000:
        raise WorkerError("prompt must be <= 2000 characters")

    image_url = event_input.get("image_url")
    if image_url is not None and not isinstance(image_url, str):
        raise WorkerError("image_url must be a string")

    return {
        "prompt": prompt.strip(),
        "negative_prompt": str(event_input.get("negative_prompt") or DEFAULT_NEGATIVE)[:1000],
        "image_url": image_url.strip() if isinstance(image_url, str) and image_url.strip() else None,
        "num_frames": snap_frames(event_input.get("num_frames", 81)),
        "width": snap_dim(event_input.get("width", 480)),
        "height": snap_dim(event_input.get("height", 832)),
        "fps": max(8, min(24, int(event_input.get("fps", 16)))),
        "seed": event_input.get("seed"),
    }


# ----------------------------------------------------------------------------
# ComfyUI workflow construction (API format)
# ----------------------------------------------------------------------------
def build_workflow(p):
    mode = "i2v" if p["image_url"] else "t2v"
    if mode == "t2v":
        high, low, lora_high, lora_low = T2V_HIGH, T2V_LOW, T2V_LORA_HIGH, T2V_LORA_LOW
    else:
        high, low, lora_high, lora_low = I2V_HIGH, I2V_LOW, I2V_LORA_HIGH, I2V_LORA_LOW

    seed = p["seed"]
    if not isinstance(seed, int):
        seed = random.randint(0, 2**31 - 1)

    wf = {
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_FILE, "type": "wan", "device": "default"},
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_FILE},
        },
        "unet_high": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": high, "weight_dtype": "default"},
        },
        "unet_low": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": low, "weight_dtype": "default"},
        },
        "lora_high": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet_high", 0],
                "lora_name": lora_high,
                "strength_model": 1.0,
            },
        },
        "lora_low": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet_low", 0],
                "lora_name": lora_low,
                "strength_model": 1.0,
            },
        },
        "shift_high": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["lora_high", 0], "shift": SHIFT},
        },
        "shift_low": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["lora_low", 0], "shift": SHIFT},
        },
        "pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip", 0], "text": p["prompt"]},
        },
        "neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip", 0], "text": p["negative_prompt"]},
        },
        # High-noise expert: steps 0-2 of the 4-step schedule.
        "ks_high": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["shift_high", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["latent", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "control_after_generate": "randomize",
                "steps": STEPS,
                "cfg": CFG,
                "sampler_name": SAMPLER,
                "scheduler": SCHEDULER,
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable",
            },
        },
        # Low-noise expert: steps 2-4, continuing from the high-noise latent.
        "ks_low": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["shift_low", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["ks_high", 0],
                "add_noise": "disable",
                "noise_seed": 0,
                "control_after_generate": "fixed",
                "steps": STEPS,
                "cfg": CFG,
                "sampler_name": SAMPLER,
                "scheduler": SCHEDULER,
                "start_at_step": 2,
                "end_at_step": STEPS,
                "return_with_leftover_noise": "disable",
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["ks_low", 0], "vae": ["vae", 0]},
        },
        "mkvideo": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["decode", 0], "fps": p["fps"]},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["mkvideo", 0],
                "filename_prefix": "wan22",
                "format": "auto",
                "codec": "auto",
            },
        },
    }

    if mode == "i2v":
        wf["loadimg"] = {
            "class_type": "LoadImage",
            "inputs": {"image": p["_image_filename"]},
        }
        wf["latent"] = {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["vae", 0],
                "width": p["width"],
                "height": p["height"],
                "length": p["num_frames"],
                "batch_size": 1,
                "image": ["loadimg", 0],
            },
        }
    else:
        wf["latent"] = {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": p["width"],
                "height": p["height"],
                "length": p["num_frames"],
                "batch_size": 1,
            },
        }
    return wf, mode, seed


# ----------------------------------------------------------------------------
# ComfyUI API helpers
# ----------------------------------------------------------------------------
def comfy_post(path, payload=None, timeout=60):
    r = requests.post(f"{COMFY_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def comfy_get(path, timeout=30):
    r = requests.get(f"{COMFY_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def queue_and_wait(workflow):
    queued = comfy_post("/prompt", {"prompt": workflow})
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise WorkerError(f"ComfyUI did not return a prompt_id: {queued}")

    deadline = time.time() + GENERATION_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(5)
        try:
            history = comfy_get(f"/history/{prompt_id}")
        except requests.RequestException:
            continue  # transient — keep polling
        entry = history.get(prompt_id)
        if not entry:
            continue
        status = entry.get("status", {})
        if status.get("completed") or "outputs" in entry:
            outputs = entry.get("outputs", {})
            for node_id, node_out in outputs.items():
                for key in ("gifs", "videos"):
                    for f in node_out.get(key, []):
                        name = f.get("filename", "")
                        if name.lower().endswith((".mp4", ".webm", ".mov")):
                            return name, f.get("subfolder", "")
            raise WorkerError(f"ComfyUI finished but produced no video file: {outputs}")
        if status.get("status_str") == "error":
            raise WorkerError(f"ComfyUI execution error: {status.get('messages')}")
    raise WorkerError("ComfyUI generation timed out")


# ----------------------------------------------------------------------------
# Image handling (i2v start frame)
# ----------------------------------------------------------------------------
def fetch_start_frame(image_url, width, height):
    try:
        r = requests.get(image_url, timeout=60, headers={"User-Agent": "pulseclone-wan-worker/1.0"})
        r.raise_for_status()
    except requests.RequestException as e:
        raise WorkerError(f"Could not download image_url: {e}")
    try:
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise WorkerError(f"image_url is not a readable image: {e}")
    img = img.resize((width, height), Image.LANCZOS)
    filename = f"i2v_{uuid.uuid4().hex}.png"
    os.makedirs(INPUT_DIR, exist_ok=True)
    img.save(os.path.join(INPUT_DIR, filename))
    return filename


# ----------------------------------------------------------------------------
# RunPod handler
# ----------------------------------------------------------------------------
def handler(job):
    event_input = job.get("input") if isinstance(job, dict) else None
    p = validate(event_input or {})

    if p["image_url"]:
        p["_image_filename"] = fetch_start_frame(p["image_url"], p["width"], p["height"])

    workflow, mode, seed = build_workflow(p)
    filename, _subfolder = queue_and_wait(workflow)

    video_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(video_path):
        # ComfyUI may nest output under a dated subfolder; search for it.
        found = None
        for root, _dirs, files in os.walk(OUTPUT_DIR):
            if filename in files:
                found = os.path.join(root, filename)
                break
        if not found:
            raise WorkerError(f"ComfyUI reported {filename} but the file is missing")
        video_path = found

    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    return {
        "video": f"data:video/mp4;base64,{b64}",
        "metadata": {
            "mode": mode,
            "width": p["width"],
            "height": p["height"],
            "num_frames": p["num_frames"],
            "fps": p["fps"],
            "seed": seed,
            "steps": STEPS,
            "filename": filename,
        },
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
