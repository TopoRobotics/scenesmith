"""Amazon Bedrock image-generation backend (on-demand, pay-per-request).

A drop-in ``BaseImageGenerator`` that routes the same three calls the OpenAI and
Gemini backends serve to Bedrock's on-demand ``InvokeModel`` API — no provisioned
throughput, nothing running when idle, billed per image. Amazon Nova Canvas by
default (Titan Image works with the same request shape via ``model_id``):

- ``generate_images``            -> Nova Canvas TEXT_IMAGE
- ``generate_furniture_context`` -> Nova Canvas IMAGE_VARIATION (edit a render)
- ``generate_manipuland_context``-> Nova Canvas IMAGE_VARIATION (edit a render)

Auth is standard AWS (the task's IAM role needs ``bedrock:InvokeModel`` for the
model); region comes from config or ``AWS_REGION``. Kept in its own module so the
core ``image_generation.py`` stays free of the boto3 import.
"""

import base64
import json
import logging
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scenesmith.agent_utils.image_generation import BaseImageGenerator
from scenesmith.prompts import PROMPTS_DATA_DIR
from scenesmith.prompts.manager import PromptManager
from scenesmith.prompts.registry import ImageGenerationPrompts

console_logger = logging.getLogger(__name__)

# Nova Canvas caps the prompt at 1024 characters; SceneSmith's composed prompts
# can exceed that, so truncate defensively rather than let the API 400.
_MAX_PROMPT_CHARS = 1024


def _parse_size(size: str | None) -> tuple[int, int]:
    """"1024x1024" -> (1024, 1024); default square. Nova Canvas takes ints."""
    if not size:
        return 1024, 1024
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 1024, 1024


class BedrockImageGenerator(BaseImageGenerator):
    """Image generation via Amazon Bedrock on-demand InvokeModel."""

    def __init__(
        self,
        model_id: str = "amazon.nova-canvas-v1:0",
        region: str | None = None,
        quality: str = "standard",  # Nova Canvas: "standard" | "premium"
        cfg_scale: float = 6.5,
        similarity_strength: float = 0.7,  # IMAGE_VARIATION: how close to the reference
        client=None,
    ) -> None:
        import boto3  # local import: only this backend needs it

        self.model_id = model_id
        self.quality = quality
        self.cfg_scale = cfg_scale
        self.similarity_strength = similarity_strength
        self.client = client or boto3.client(
            "bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1")
        )
        self.prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)

    # -- low-level ---------------------------------------------------------------

    def _invoke(self, body: dict, output_path: Path, label: str) -> Path:
        start = time.time()
        resp = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        if payload.get("error"):
            raise RuntimeError(f"Bedrock image generation error for {label}: {payload['error']}")
        images = payload.get("images") or []
        if not images:
            raise RuntimeError(f"Bedrock returned no image for {label}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(images[0]))
        console_logger.info(f"Generated image for {label} in {time.time() - start:.2f}s (Bedrock {self.model_id})")
        return output_path

    def _image_config(self, width: int, height: int) -> dict:
        return {"numberOfImages": 1, "width": width, "height": height,
                "quality": self.quality, "cfgScale": self.cfg_scale}

    # -- text -> image -----------------------------------------------------------

    def generate_images(
        self,
        style_prompt: str,
        object_descriptions: list[str],
        output_paths: list[Path],
        size: str | None = None,
        labels: list[str] | None = None,
    ) -> None:
        if len(object_descriptions) != len(output_paths):
            raise ValueError("Number of descriptions must match number of output paths")
        width, height = _parse_size(size)
        effective_labels = labels if labels else object_descriptions
        console_logger.info(f"Generating {len(object_descriptions)} images (Bedrock {self.model_id})")

        def one(description: str, output_path: Path, label: str) -> None:
            prompt = self.prompt_manager.get_prompt(
                ImageGenerationPrompts.ASSET_IMAGE_INITIAL,
                description=description, style_prompt=style_prompt,
            )
            body = {"taskType": "TEXT_IMAGE",
                    "textToImageParams": {"text": prompt[:_MAX_PROMPT_CHARS]},
                    "imageGenerationConfig": self._image_config(width, height)}
            self._invoke(body, output_path, label)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(one, d, p, l)
                       for d, p, l in zip(object_descriptions, output_paths, effective_labels)]
            for future in as_completed(futures):
                future.result()

    # -- image edit (reference render -> edited image) ---------------------------

    def _edit_image(self, prompt: str, reference_image_path: Path, output_path: Path,
                    size: str = "1024x1024") -> Path:
        width, height = _parse_size(size)
        ref_b64 = base64.b64encode(Path(reference_image_path).read_bytes()).decode("utf-8")
        body = {"taskType": "IMAGE_VARIATION",
                "imageVariationParams": {"images": [ref_b64], "text": prompt[:_MAX_PROMPT_CHARS],
                                         "similarityStrength": self.similarity_strength},
                "imageGenerationConfig": self._image_config(width, height)}
        console_logger.info(f"Editing image {reference_image_path} (Bedrock {self.model_id})")
        return self._invoke(body, output_path, "edited image")

    def generate_furniture_context_image(
        self,
        reference_image_path: Path,
        scene_description: str,
        width_m: float,
        length_m: float,
        output_path: Path,
    ) -> Path:
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
            scene_description=scene_description, width_m=width_m, length_m=length_m,
        )
        console_logger.info("Generating furniture placement context image")
        return self._edit_image(prompt, reference_image_path, output_path)

    def generate_manipuland_context_image(
        self,
        reference_image_path: Path,
        furniture_description: str,
        furniture_dimensions: str,
        suggested_items: str,
        prompt_constraints: str,
        style_notes: str,
        output_path: Path,
    ) -> Path:
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.MANIPULAND_CONTEXT_IMAGE,
            furniture_description=furniture_description,
            furniture_dimensions=furniture_dimensions,
            suggested_items=suggested_items,
            prompt_constraints=prompt_constraints,
            style_notes=style_notes,
        )
        console_logger.info("Generating manipuland placement context image")
        return self._edit_image(prompt, reference_image_path, output_path)
