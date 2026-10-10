# Wan 2.2 RunPod Serverless Worker

Self-hosted text-to-video + image-to-video backend for **PulseClone**, running
Alibaba's open-weights **Wan 2.2 14B** on a rented GPU. Replaces per-render
fal.ai billing (~$0.70–2.00 / 5s clip) with raw GPU time (~$0.05 / 5s clip).

## What it is

A RunPod Serverless worker: ComfyUI (API mode) + a small Python handler
(`rp_handler.py`). Each job builds a ComfyUI workflow from the official
ComfyUI Wan2.2 14B template structure and returns the finished clip.

**Model stack** (all Apache-2.0, from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` —
file names verified against the HuggingFace repo):

| Role | File | Size |
|---|---|---|
| T2V high-noise DiT (fp8) | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | ~13.3 GB |
| T2V low-noise DiT (fp8) | `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | ~13.3 GB |
| I2V high-noise DiT (fp8) | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | ~13.3 GB |
| I2V low-noise DiT (fp8) | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | ~13.3 GB |
| T2V LightX2V 4-step LoRA ×2 | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_{high,low}_noise.safetensors` | ~1.1 GB ea |
| I2V LightX2V 4-step LoRA ×2 | `wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors` | ~1.1 GB ea |
| Text encoder (fp8) | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ~6.3 GB |
| VAE | `wan_2.1_vae.safetensors` | ~0.2 GB |

**Why this variant:** the LightX2V 4-step LoRAs are distilled for exactly 4
denoising steps (no CFG — guidance is baked in, `cfg=1.0`), giving roughly a
5–7× speedup over the 40-step base at quality the LightX2V team rates on par
with the base model. The official ComfyUI template's own benchmark on a
24 GB card: 640×640×81 in ~108 s first run / ~71 s warmed (vs ~536 s without
the LoRA). The fp8-scaled DiTs are what make 14B fit in 24 GB VRAM.

**Sampler cascade** (matches the official ComfyUI Wan2.2 template):
`KSamplerAdvanced` high-noise expert on steps 0–2 → low-noise expert on steps
2–4, `euler`/`simple`, `ModelSamplingSD3` shift 5.0, `VAEDecode` →
`CreateVideo` (fps from input) → `SaveVideo` (mp4).

## GPU requirement

**NVIDIA RTX 4090 (24 GB VRAM)** — the officially benchmarked card for this
exact stack. A40/A100 40 GB+ also work. Do not use <24 GB cards: the dual
fp8 experts + T5-XXL need it (official template reports ~89% VRAM at
640×640 on 24 GB).

## How RunPod builds it from GitHub

1. Push this repo to GitHub.
2. In RunPod Console → Serverless → New Endpoint → **Connect GitHub**, pick
   the repo/branch. RunPod builds the Docker image automatically.
   (Alternative: `docker build -t <you>/wan22-worker . && docker push …` and
   paste the image URL.)
3. **Network volume (required):** create a ~75 GB network volume in the same
   region, attach it to the endpoint at `/runpod-volume`. Populate it **once**
   from any temporary pod with the volume attached:
   `bash download_models.sh /runpod-volume/models` (~64 GB, 20–40 min).
   The entrypoint symlinks the volume into ComfyUI and refuses to start if
   any model file is missing.
4. Endpoint settings:
   - GPU: RTX 4090 · Container disk: 20 GB · Volume mount: `/runpod-volume`
   - **Max workers:** 2–3 to start (queue instead of cost-spiking on bursts)
   - **Idle timeout:** 60 s · **Min workers:** 0 (scale to zero)
   - **Execution timeout:** 1200 s (a 121-frame render can take ~10 min)
5. Test:
   ```bash
   curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
     -H "Authorization: Bearer $RUNPOD_API_KEY" \
     -H "Content-Type: application/json" \
     -d @test_input.json
   # then poll: GET https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<JOB_ID>
   ```
6. Copy the endpoint ID + a RunPod API key into PulseClone's secrets as
   `WAN_RUNPOD_ENDPOINT_ID` / `WAN_RUNPOD_API_KEY`.

## Input / output contract

**Request** — `POST /run`, body `{ "input": { … } }`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string (required) | — | 1–2000 chars |
| `negative_prompt` | string | short English default | ≤1000 chars |
| `image_url` | string | none | **when present → image-to-video** (start frame); otherwise text-to-video |
| `num_frames` | int | 81 | clamped 9–121, snapped to 8n+1 (Wan architecture requirement) |
| `width` / `height` | int | 480 / 832 | clamped 256–1024, snapped to ×16 |
| `fps` | int | 16 | 8–24 |
| `num_inference_steps` | int | 4 | accepted but **forced to 4** — the LoRAs are distilled for 4 NFE |
| `guidance_scale` | float | 1.0 | accepted but **forced to 1.0** — guidance is baked into the distill |
| `seed` | int | random | |

**Response** — standard RunPod `/status/{id}` envelope; on `COMPLETED`:

```json
{ "status": "COMPLETED",
  "output": { "video": "data:video/mp4;base64,…",
              "metadata": { "mode": "t2v", "width": 480, "height": 832,
                            "num_frames": 81, "fps": 16, "seed": 42,
                            "steps": 4, "filename": "wan22_00001.mp4" } } }
```

The client (`wan-video.server.ts` in PulseClone) also accepts a bare URL
string or `{ "video_url" }` — base64 data URI is used here so the worker is
fully self-contained (no external storage dependency). On failure the job
reports `FAILED` with `error`.

## Known limitations

- **Cold start ~60 s** (up to ~5 min worst case): image pull + ComfyUI boot +
  ~60 GB model load into VRAM. First render after idle shows "warming up".
- **Render time (RTX 4090, official-template benchmarks):** 640×640×81 ≈
  70–110 s; 512×512×33 ≈ 30–40 s. A 121-frame 832×480 clip can take ~10 min
  and is near the 24 GB VRAM ceiling — prefer ≤81 frames above 640×640.
- **Quality:** 4-step distilled ≈ base model per LightX2V, but fine facial
  detail and complex multi-object scenes trail full-step Veo 3.1 / Kling 3.0.
  This worker is the cheap workhorse; keep fal.ai for hero shots.
- **No audio:** Wan 2.2 is video-only (unlike Seedance). Add music/SFX in
  PulseClone's post pipeline.
- **One clip per worker at a time;** concurrency = max workers.
- `Wan22ImageToVideoLatent` input names follow current ComfyUI native nodes —
  if a future ComfyUI rev renames them, the i2v path is the first thing to
  check (`/tmp/comfyui.log` in the worker).
- Base image pins `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`;
  ComfyUI itself is cloned at latest — if a ComfyUI update breaks a node,
  pin `git clone --branch` to a known-good commit.

## Cost control recap

Scale-to-zero + 60 s idle timeout means you pay only for active render
seconds (~$1.10/hr on serverless 4090 → ~$0.03–0.06 per 5 s clip). Cap Max
workers to bound burst spend; never enable always-warm workers until daily
volume justifies it.
