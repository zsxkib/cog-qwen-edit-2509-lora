# Qwen Edit 2509 – LoRA Harness

Minimal Cog deployment for `Qwen/Qwen-Image-Edit-2509` with plug-and-play LoRA loading. It ships the same transformer, FA3 attention processor, and optional AOT optimisation used in the Hugging Face space, but leaves the LoRA choice up to the caller.

The goal is to provide a clean, auditable harness that works out-of-the-box with the base model and can optionally mount any `.safetensors` LoRA (local path, Hugging Face slug, or direct URL) without extra plumbing.

## Quickstart

```bash
pip install cog==0.16.8
cog predict \
  -i image=@examples/monkey.jpg \
  -i rotate_degrees=45 \
  -i move_forward=5 \
  -i go_fast=true \  # uses 4 denoise steps unless you override num_inference_steps
  -i lora_weights="owner/repo-or-https-url-if-you-have-one" \
  -i output_format=webp \
  -o outputs/sample.webp
```

- Leave `lora_weights` blank to use the base model, or supply either a Hugging Face slug (`owner/model`) or a direct `.safetensors`/`.zip`/`.tar` URL (`https://.../resolve/.../weights.safetensors`, etc.).
- `go_fast=true` (with no `num_inference_steps`) gives you the 4-step fast preset; set `go_fast=false` or provide a `num_inference_steps` value (1–40) for slower passes.
- `lora_scale` defaults to `1.25` but can be adjusted when you load custom weights.

## Runtime inputs

| Input | Purpose |
|-------|---------|
| `image` | Required source image (`cog.Path`). |
| `prompt` | Optional free-form text appended after the auto-generated camera instruction. |
| `rotate_degrees` | -90 → 90. Positive rotates left, negative rotates right. |
| `move_forward` | 0 → 10. Higher values push toward a close-up. |
| `vertical_tilt` | -1 (bird’s-eye) → 1 (worm’s-eye). |
| `use_wide_angle` | Boolean toggle for the wide-angle instruction. |
| `aspect_ratio` | Preset aspect ratios or `match_input_image`. |
| `go_fast` | When `true` (default) and `num_inference_steps` is omitted, run the 4-step fast preset; set to `false` for the 40-step preset. |
| `num_inference_steps` | Optional explicit denoise count (1–40). Overrides the presets when provided. |
| `true_guidance_scale` | 0–10 true CFG scale passed straight to the pipeline. |
| `lora_weights` | Optional LoRA weights. Accepts Hugging Face repo slugs (`owner/model`), `.../resolve/...` URLs, or arbitrary .safetensors/.zip/.tar URLs. |
| `lora_scale` | Scale applied to the selected LoRA (default 1.25). |
| `seed` | Optional RNG seed for reproducible results. |
| `output_format`, `output_quality` | Post-processing options (`webp`/`jpg`/`png`, quality 0–100). |
| `disable_safety_checker` | Skip NSFW checks (not recommended). |

## Implementation notes

- The transformer weights come from `linoyts/Qwen-Image-Edit-Rapid-AIO` and are paired with the FA3 attention processor exactly as in the HF space.
- The LoRA loader accepts local paths, Hugging Face slugs, or direct download links (`.safetensors`, `.zip`, `.tar.*`) and caches the converted weights for reuse.
- After each prediction we detach any custom adapters so the next call starts from the base model unless you supply another LoRA.
- `optimization.py` mirrors the HF AOT helper. If the runtime lacks Hugging Face’s `spaces` package the call is simply skipped.

## Repository layout

| Path | Description |
|------|-------------|
| `predict.py` | Cog predictor with camera prompt assembly, LoRA management, and safety filtering. |
| `qwenimage/` | Local copy of the HF Qwen edit pipeline (`pipeline_qwenimage_edit_plus.py`, FA3 processor, transformer). |
| `optimization.py` | Optional AOT compile helper (no-op outside Spaces). |
| `cog.yaml` | Build recipe. Only installs what the predictor actually uses. |
| `requirements.txt` | Python dependencies (diffusers, transformers, torchao, kernels, etc.). |

## Development tips

1. Keep the Hugging Face space open while making changes; feature parity is the benchmark.
2. When adjusting inputs, prefer tweaking bounds/descriptions over renaming keys—Replicate clients rely on stable field names.
3. If you load additional LoRAs frequently, drop them into `custom_loras/` to skip future downloads (gitignored).
4. For regression testing, reuse the example images bundled in the HF repo (`tool_of_the_sea.png`, `monkey.jpg`, …) together with representative camera motions.

Questions or follow-ups? Ping @zsakib. Happy camera moves! 🎥
