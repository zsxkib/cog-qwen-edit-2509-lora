import hashlib
import json
import random
import shutil
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path as FsPath
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from cog import BasePredictor, Input, Path, current_scope
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForImageClassification,
    CLIPImageProcessor,
    ViTImageProcessor,
)

from optimization import optimize_pipeline_
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

CONVERSION_VERSION = "v2"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1392, 752),
    "9:16": (752, 1392),
    "4:3": (1184, 880),
    "3:4": (880, 1184),
    "match_input_image": (None, None),
}


class LoRALoader:
    """Downloads and converts LoRA weights into a PEFT-compatible safetensors file."""

    def __init__(self, workspace: FsPath, cache_subdir: str = "lora_cache") -> None:
        self.workspace = workspace
        self.cache_dir = workspace / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._processed_cache: Dict[str, FsPath] = {}
        self._adapter_signatures: Dict[str, Tuple[str, float]] = {}

    def prepare(
        self,
        source: str,
        weight_name: Optional[str],
    ) -> FsPath:
        """Return a local path to a PEFT-compatible safetensors file for the given source."""
        normalized = source.strip()
        if not normalized:
            raise ValueError("LoRA source string cannot be empty.")

        repo_id: Optional[str] = None
        filename: Optional[str] = weight_name.strip() if weight_name else None
        local_candidate: Optional[FsPath] = None

        if normalized.startswith(("http://", "https://")):
            normalized = normalized.split("huggingface.co/", 1)[-1]
        normalized = normalized.strip("/")

        potential_path = FsPath(normalized)
        if potential_path.exists():
            local_candidate = (
                potential_path
                if potential_path.is_absolute()
                else self.workspace / potential_path
            )
        elif "/" in normalized:
            repo_id = normalized
        else:
            raise ValueError(
                f"Unable to interpret LoRA source '{source}'. Provide a local path or Hugging Face repo id."
            )

        if local_candidate is None and repo_id is not None:
            filename = filename or "镜头转换.safetensors"
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            local_candidate = FsPath(downloaded)

        if local_candidate is None or not local_candidate.exists():
            raise FileNotFoundError(
                f"Resolved LoRA weights not found for source '{source}'."
            )

        return self._convert_if_needed(local_candidate)

    def _convert_if_needed(self, src_path: FsPath) -> FsPath:
        """Convert Comfy/LyCORIS-style weights to PEFT format if necessary."""
        key = self._cache_key(src_path)
        if key in self._processed_cache and self._processed_cache[key].exists():
            return self._processed_cache[key]

        state_dict = load_file(str(src_path))
        needs_conversion = any(
            ".lora_A" in name or ".lora_B" in name for name in state_dict.keys()
        )

        if not needs_conversion:
            # Already PEFT style
            self._processed_cache[key] = src_path
            return src_path

        processed_path = self.cache_dir / f"{key}.safetensors"
        if processed_path.exists():
            self._processed_cache[key] = processed_path
            return processed_path

        converted: Dict[str, torch.Tensor] = {}
        alpha_defaults: Dict[str, Tuple[torch.dtype, torch.device, int]] = {}
        target_modules: set[str] = set()

        for name, tensor in state_dict.items():
            clean_name = name.replace(".default", "")
            if clean_name.endswith(".lora_A.weight"):
                base = clean_name.replace(".lora_A.weight", "")
                converted[f"{base}.lora_down.weight"] = tensor
                alpha_defaults.setdefault(
                    base,
                    (tensor.dtype, tensor.device, tensor.shape[0]),
                )
                target_modules.add(self._target_from_module_path(base))
            elif clean_name.endswith(".lora_B.weight"):
                base = clean_name.replace(".lora_B.weight", "")
                converted[f"{base}.lora_up.weight"] = tensor
                target_modules.add(self._target_from_module_path(base))
            else:
                converted[clean_name] = tensor

        for base, (dtype, device, rank) in alpha_defaults.items():
            alpha_key = f"{base}.alpha"
            if alpha_key not in converted:
                converted[alpha_key] = torch.tensor(
                    float(rank), dtype=dtype, device=device
                )

        representative_rank = (
            next(iter(alpha_defaults.values()))[2] if alpha_defaults else 16
        )
        peft_config = {
            "base_model_name_or_path": "Qwen/Qwen-Image-Edit-2509",
            "bias": "none",
            "lora_alpha": representative_rank,
            "lora_dropout": 0.0,
            "peft_type": "LORA",
            "target_modules": sorted(target_modules),
            "task_type": "UNDEFINED",
        }

        save_file(
            converted,
            str(processed_path),
            metadata={"peft.config": json.dumps(peft_config)},
        )
        self._processed_cache[key] = processed_path
        return processed_path

    def ensure_loaded(
        self,
        pipe: QwenImageEditPlusPipeline,
        adapter_name: str,
        processed_path: FsPath,
    ) -> None:
        """Load or hot-swap the adapter into the given pipeline."""
        mtime = processed_path.stat().st_mtime
        signature = (str(processed_path), mtime)
        if self._adapter_signatures.get(adapter_name) == signature:
            return

        available = pipe.get_list_adapters() or {}
        transformer_adapters = available.get("transformer", [])
        hotswap = adapter_name in transformer_adapters
        pipe.load_lora_weights(
            str(processed_path),
            adapter_name=adapter_name,
            hotswap=hotswap,
        )
        self._adapter_signatures[adapter_name] = signature

    def _cache_key(self, path: FsPath) -> str:
        digest = hashlib.sha256()
        digest.update(CONVERSION_VERSION.encode("utf-8"))
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(path.stat().st_mtime_ns).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _target_from_module_path(module_path: str) -> str:
        parts = module_path.split(".")
        if len(parts) >= 3 and parts[0] == "transformer_blocks" and parts[1].isdigit():
            return ".".join(parts[2:])
        return module_path


class Predictor(BasePredictor):
    """Qwen-Image-Edit-2509 predictor with Lightning + Multi-Angle LoRA support."""

    def __init__(self) -> None:
        super().__init__()
        self._workspace = FsPath(__file__).resolve().parent
        self._lora_loader = LoRALoader(self._workspace)

        self._custom_adapter_name = "custom_lora"

    def setup(self) -> None:
        """Load the model and fast-path optimisations into memory."""
        setup_start_time = time.time()
        model_path = "Qwen/Qwen-Image-Edit-2509"
        transformer_repo = "linoyts/Qwen-Image-Edit-Rapid-AIO"
        dtype = torch.bfloat16

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.cuda.empty_cache()

        print("Load Qwen-Image-Edit pipeline ...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            transformer_repo,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map="cuda",
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=dtype,
        ).to("cuda")

        # Match the Hugging Face space upgrades (FA3 attention + fused transformer class).
        pipe.transformer.__class__ = QwenImageTransformer2DModel
        pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

        try:
            optimize_pipeline_(
                pipe,
                image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))],
                prompt="camera move",
            )
        except Exception as exc:  # pragma: no cover - optimisation is optional
            print(f"Warning: Failed to apply optimisation helper ({exc}).")

        self.pipe = pipe

        self.custom_lora_cache = self._workspace / "custom_loras"
        self.custom_lora_cache.mkdir(parents=True, exist_ok=True)

        print("Load Stable Diffusion Safety Checker ...")
        safety_checker_start_time = time.time()
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        print("Load Falcon Safety Checker ...")
        self.falcon_model = AutoModelForImageClassification.from_pretrained(
            "Falconsai/nsfw_image_detection"
        )
        self.falcon_processor = ViTImageProcessor.from_pretrained(
            "Falconsai/nsfw_image_detection"
        )
        print(
            f"Safety checkers loaded in {time.time() - safety_checker_start_time:.2f} seconds"
        )

        print(f"Total setup took {time.time() - setup_start_time:.2f} seconds")
        self.pipe.set_progress_bar_config(disable=True)

    def _load_input_images(self, uploads: List[Path]) -> List[Image.Image]:
        """Normalize Cog paths into RGB PIL images."""
        if not uploads:
            raise ValueError("At least one input image must be provided.")

        pil_images: List[Image.Image] = []
        for item in uploads:
            source_path = FsPath(str(item))
            with Image.open(source_path) as im:
                pil_images.append(im.convert("RGB"))

        if not pil_images:
            raise ValueError("No valid images were provided.")

        return pil_images

    def _build_camera_prompt(
        self,
        rotate_degrees: int,
        move_forward: int,
        vertical_tilt: int,
        use_wide_angle: bool,
    ) -> str:
        instructions: List[str] = []

        if rotate_degrees != 0:
            direction = "left" if rotate_degrees > 0 else "right"
            instructions.append(
                f"将镜头向{'左' if direction == 'left' else '右'}旋转{abs(rotate_degrees)}度 Rotate the camera {abs(rotate_degrees)} degrees to the {direction}."
            )

        if move_forward > 5:
            instructions.append("将镜头转为特写镜头 Turn the camera to a close-up.")
        elif move_forward >= 1:
            instructions.append("将镜头向前移动 Move the camera forward.")

        if vertical_tilt <= -1:
            instructions.append("将镜头转为俯视 Turn the camera to a top-down view.")
        elif vertical_tilt >= 1:
            instructions.append("将镜头转为仰视 Turn the camera to a low-angle view.")

        if use_wide_angle:
            instructions.append(
                "将镜头转为广角镜头 Turn the camera to a wide-angle lens."
            )

        return " ".join(instructions).strip()

    # -- Custom LoRA helpers -------------------------------------------------

    def _normalize_hf_url(self, url: str) -> str:
        if "huggingface.co" in url and "/blob/" in url:
            return url.replace("/blob/", "/resolve/", 1)
        return url

    def _custom_lora_cache_path(self, source: str) -> FsPath:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return self.custom_lora_cache / f"{digest}.safetensors"

    def _download_to_path(self, url: str, destination: FsPath) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(destination, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    def _extract_safetensors(self, archive_path: FsPath) -> FsPath:
        tmp_dir = FsPath(tempfile.mkdtemp(prefix="lora_extract_"))

        try:
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(tmp_dir)
            else:
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(tmp_dir)

            candidates = sorted(
                tmp_dir.rglob("*.safetensors"),
                key=lambda p: ("lora" not in p.name.lower(), p.name.lower()),
            )
            if not candidates:
                raise FileNotFoundError("Archive did not contain a .safetensors file.")
            return candidates[0]
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _download_lora_from_repo(self, repo_id: str) -> FsPath:
        files = list_repo_files(repo_id)
        safes = [f for f in files if f.lower().endswith(".safetensors")]
        if not safes:
            raise FileNotFoundError(
                f"No .safetensors files found in Hugging Face repo '{repo_id}'."
            )
        safes.sort(key=lambda name: ("lora" not in name.lower(), name.lower()))
        filename = safes[0]
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        return FsPath(downloaded)

    def _resolve_custom_lora(self, source: str) -> FsPath:
        source = source.strip()
        if not source:
            raise ValueError("Provide a non-empty value for lora_weights.")

        cache_dest = self._custom_lora_cache_path(source)
        if cache_dest.exists():
            return cache_dest

        # Local filesystem path
        candidate = FsPath(source)
        if candidate.exists():
            shutil.copy2(candidate, cache_dest)
            return self._lora_loader._convert_if_needed(cache_dest)

        # Remote sources
        if source.startswith(("http://", "https://")):
            normalized = self._normalize_hf_url(source)
            parsed = urlparse(normalized)
            lower_path = parsed.path.lower()

            with tempfile.TemporaryDirectory(prefix="lora_dl_") as td:
                tmp_dir = FsPath(td)
                download_path = tmp_dir / "downloaded"
                self._download_to_path(normalized, download_path)

                if lower_path.endswith(".safetensors"):
                    shutil.copy2(download_path, cache_dest)
                    return self._lora_loader._convert_if_needed(cache_dest)

                if any(
                    lower_path.endswith(ext)
                    for ext in (
                        ".zip",
                        ".tar",
                        ".tar.gz",
                        ".tgz",
                        ".tar.xz",
                        ".tar.bz2",
                    )
                ):
                    extracted = self._extract_safetensors(download_path)
                    shutil.copy2(extracted, cache_dest)
                    return self._lora_loader._convert_if_needed(cache_dest)

                raise ValueError(
                    "Unsupported file type for lora_weights URL. Provide a .safetensors file or an archive."
                )

        # Hugging Face repo id (owner/repo[/subpath])
        if "/" in source and "." not in source.split("/")[-1]:
            lora_path = self._download_lora_from_repo(source)
            shutil.copy2(lora_path, cache_dest)
            return self._lora_loader._convert_if_needed(cache_dest)

        # Hugging Face repo file path owner/repo/file.safetensors
        if "/" in source and source.split("/")[-1].lower().endswith(".safetensors"):
            repo_id = "/".join(source.split("/")[:-1])
            filename = source.split("/")[-1]
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copy2(downloaded, cache_dest)
            return self._lora_loader._convert_if_needed(cache_dest)

        raise ValueError(
            "Unsupported lora_weights value. Provide a local path, a Hugging Face repo id, or a direct URL."
        )

    def _activate_lora(self, source: str, scale: float) -> str:
        """Load and activate the requested LoRA, returning the adapter name."""
        effective_source = source.strip() if source else ""
        transformer_adapters = self.pipe.get_list_adapters() or {}
        transformer_names = transformer_adapters.get("transformer", [])

        if not effective_source:
            self.pipe.disable_lora()
            if self._custom_adapter_name in transformer_names:
                self.pipe.delete_adapters(self._custom_adapter_name)
            return "base"

        custom_path = self._resolve_custom_lora(effective_source)
        self._lora_loader.ensure_loaded(
            self.pipe,
            self._custom_adapter_name,
            FsPath(custom_path),
        )
        self.pipe.disable_lora()
        self.pipe.set_adapters(
            [self._custom_adapter_name],
            adapter_weights=[float(scale)],
        )
        return self._custom_adapter_name

    def _restore_default_lora(self) -> None:
        """Disable any custom LoRA adapters."""
        self.pipe.disable_lora()
        adapters = self.pipe.get_list_adapters() or {}
        if self._custom_adapter_name in adapters.get("transformer", []):
            try:
                self.pipe.delete_adapters(self._custom_adapter_name)
            except Exception as exc:  # pragma: no cover - defensive cleanup
                print(f"Warning: Failed to delete custom adapter ({exc}).")

    def run_safety_checker(self, images, np_images):
        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(
            "cuda"
        )
        image, has_nsfw_concept = self.safety_checker(
            images=np_images,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def run_falcon_safety_checker(self, image):
        with torch.no_grad():
            inputs = self.falcon_processor(images=image, return_tensors="pt")
            outputs = self.falcon_model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            result = self.falcon_model.config.id2label[predicted_label]
        return result == "normal"

    def postprocess(
        self,
        images: List[Image.Image],
        disable_safety_checker: bool,
        output_format: str,
        output_quality: int,
        np_images: List[np.ndarray],
    ) -> List[Path]:
        has_nsfw_content = [False] * len(images)

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(images, np_images)

        output_paths = []
        for i, (img, is_nsfw) in enumerate(zip(images, has_nsfw_content)):
            if is_nsfw:
                try:
                    falcon_is_safe = self.run_falcon_safety_checker(img)
                except Exception as e:
                    print(f"Error running safety checker: {e}")
                    falcon_is_safe = False
                if not falcon_is_safe:
                    print(f"NSFW content detected in image {i}")
                    continue

            output_path = f"out-{i}.{output_format}"
            save_params = (
                {"quality": output_quality, "optimize": True}
                if output_format != "png"
                else {}
            )
            img.save(output_path, **save_params)
            output_paths.append(Path(output_path))

        if not output_paths:
            raise Exception(
                "All generated images contained NSFW content. Try running it again with a different prompt."
            )

        print(f"Total safe images: {len(output_paths)} out of {len(images)}")
        return output_paths

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Image file uploaded to Cog (jpeg, png, gif, or webp).",
        ),
        prompt: str = Input(
            description="Optional text instruction appended after the camera directive.",
            default="",
        ),
        rotate_degrees: int = Input(
            description="Camera rotation in degrees. Positive rotates left, negative rotates right.",
            default=0,
            ge=-90,
            le=90,
        ),
        move_forward: int = Input(
            description="Move the camera forward (zoom in). Higher values push toward a close-up framing.",
            default=0,
            ge=0,
            le=10,
        ),
        vertical_tilt: int = Input(
            description="Vertical camera tilt. -1 = bird's-eye, 0 = level, 1 = worm's-eye.",
            default=0,
            ge=-1,
            le=1,
        ),
        use_wide_angle: bool = Input(
            description="Switch to a wide-angle lens instruction.",
            default=False,
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=list(ASPECT_RATIOS.keys()),
            default="match_input_image",
        ),
        go_fast: bool = Input(
            description="If num_inference_steps is omitted, true runs the 4-step fast preset and false runs the 40-step detailed preset.",
            default=True,
        ),
        num_inference_steps: Optional[int] = Input(
            description="Explicit denoising step count (1-40). Leave blank to use the go_fast presets (4 or 40 steps).",
            default=None,
            ge=1,
            le=40,
        ),
        lora_weights: str = Input(
            description=(
                "LoRA weights to apply. Pass a Hugging Face repo slug "
                "(for example 'owner/model') or a direct .safetensors/zip/tar URL "
                "such as 'https://huggingface.co/flymy-ai/qwen-image-lora/resolve/main/pytorch_lora_weights.safetensors'. "
                "Leave blank to run without a LoRA."
            ),
            default="",
        ),
        lora_scale: float = Input(
            description="Strength applied to the selected LoRA.",
            default=1.25,
            ge=0.0,
            le=4.0,
        ),
        true_guidance_scale: float = Input(
            description="True classifier-free guidance scale passed to the pipeline.",
            default=1.0,
            ge=0.0,
            le=10.0,
        ),
        seed: Optional[int] = Input(
            description="Random seed. Set for reproducible generation.",
            default=None,
        ),
        output_format: str = Input(
            description="Format of the output images.",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs.",
            default=95,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images.",
            default=False,
        ),
    ) -> List[Path]:
        if seed is None:
            seed = random.randint(0, 1_000_000)
        print(f"Using seed: {seed}")

        width, height = ASPECT_RATIOS[aspect_ratio]

        pil_images = self._load_input_images([image])

        auto_camera_prompt = self._build_camera_prompt(
            rotate_degrees=rotate_degrees,
            move_forward=move_forward,
            vertical_tilt=vertical_tilt,
            use_wide_angle=use_wide_angle,
        )
        prompt_components = [prompt.strip()]
        if auto_camera_prompt:
            prompt_components.insert(0, auto_camera_prompt)
        composed_prompt = " ".join(part for part in prompt_components if part) or prompt

        requested_lora = (lora_weights or "").strip()
        try:
            active_adapter = self._activate_lora(requested_lora, lora_scale)
        except Exception as exc:
            raise ValueError(f"Failed to load LoRA '{lora_weights}': {exc}") from exc
        print(f"Active adapter: {active_adapter}")

        call_kwargs = {
            "image": pil_images,
            "prompt": composed_prompt,
            "true_cfg_scale": float(true_guidance_scale),
            "width": width,
            "height": height,
        }

        resolved_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else (4 if go_fast else 40)
        )
        call_kwargs["num_inference_steps"] = resolved_steps

        if width is None:
            call_kwargs.pop("width")
        if height is None:
            call_kwargs.pop("height")

        call_kwargs["generator"] = torch.Generator("cuda").manual_seed(seed)

        start_time = time.time()
        with torch.inference_mode():
            img = self.pipe(**call_kwargs).images[0]
        end_time = time.time()
        print(f"Generation took {end_time - start_time:.2f} seconds")

        used_fast_path = go_fast if num_inference_steps is None else resolved_steps <= 8
        current_scope().record_metric("go_fast", bool(used_fast_path))
        current_scope().record_metric("image_output_count", 1)

        if requested_lora:
            self._restore_default_lora()

        np_imgs = [np.asarray(img, dtype=np.uint8)]
        return self.postprocess(
            [img],
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )
