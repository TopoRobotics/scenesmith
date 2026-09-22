"""Amazon Bedrock image-generation backend (on-demand, pay-per-request).

A drop-in ``BaseImageGenerator`` that routes the same three calls the OpenAI and
Gemini backends serve to Bedrock's on-demand ``InvokeModel`` API — no provisioned
throughput, nothing running when idle, billed per image. Uses Stability's
**SD3.5 Large** by default, which serves both of SceneSmith's needs with one
model: text→image for asset images, and image-to-image for the two
context-image edits.

    generate_images                 -> mode "text-to-image"
    generate_furniture_context      -> mode "image-to-image" (edit a render)
    generate_manipuland_context     -> mode "image-to-image" (edit a render)

Auth is standard AWS (the task's IAM role needs ``bedrock:InvokeModel`` for the
model); the image region defaults to us-west-2 (where the Stability base
generators are offered) and is independent of where the GPU/data live. Kept in
its own module so the core ``image_generation.py`` stays free of the boto3
import.
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


def _aspect(size: str | None) -> str:
    """Map an OpenAI-style size to a Stability aspect ratio (its size knob)."""
    if not size:
        return "1:1"
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except (ValueError, AttributeError):
        return "1:1"
    if w > h * 1.3:
        return "16:9"
    if h > w * 1.3:
        return "9:16"
    return "1:1"


class BedrockImageGenerator(BaseImageGenerator):
    """Image generation via Amazon Bedrock on-demand InvokeModel (Stability)."""

    def __init__(
        self,
        model_id: str = "stability.sd3-5-large-v1:0",
        region: str | None = None,
        strength: float = 0.5,  # image-to-image: how far the edit moves from the reference
        client=None,
    ) -> None:
        import boto3  # local import: only this backend needs it

        self.model_id = model_id
        self.strength = strength
        self.client = client or boto3.client(
            "bedrock-runtime",
            region_name=region or os.environ.get("AWS_BEDROCK_REGION", "us-west-2"),
        )
        self.prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)

    # -- low-level ---------------------------------------------------------------

    def _invoke(self, body: dict, output_path: Path, label: str) -> Path:
        start = time.time()
        resp = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        images = payload.get("images") or []
        if not images:
            raise RuntimeError(f"Bedrock returned no image for {label}: {payload.get('finish_reasons')}")
        # Stability reports content filtering per image; a non-null reason means no usable image.
        reasons = payload.get("finish_reasons") or [None]
        if reasons[0]:
            raise RuntimeError(f"Bedrock image for {label} was not produced: {reasons[0]}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(images[0]))
        console_logger.info(f"Generated image for {label} in {time.time() - start:.2f}s (Bedrock {self.model_id})")
        return output_path

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
        aspect = _aspect(size)
        effective_labels = labels if labels else object_descriptions
        console_logger.info(f"Generating {len(object_descriptions)} images (Bedrock {self.model_id})")

        def one(description: str, output_path: Path, label: str) -> None:
            prompt = self.prompt_manager.get_prompt(
                ImageGenerationPrompts.ASSET_IMAGE_INITIAL,
                description=description, style_prompt=style_prompt,
            )
            body = {"prompt": prompt, "mode": "text-to-image", "aspect_ratio": aspect, "output_format": "png"}
            self._invoke(body, output_path, label)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(one, d, p, l)
                       for d, p, l in zip(object_descriptions, output_paths, effective_labels)]
            for future in as_completed(futures):
                future.result()

    # -- image edit (reference render -> edited image) ---------------------------

    def _edit_image(self, prompt: str, reference_image_path: Path, output_path: Path,
                    size: str = "1024x1024") -> Path:
        ref_b64 = base64.b64encode(Path(reference_image_path).read_bytes()).decode("utf-8")
        body = {"prompt": prompt, "mode": "image-to-image", "image": ref_b64,
                "strength": self.strength, "output_format": "png"}
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
