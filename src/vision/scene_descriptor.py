import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import base64
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as cv2
import asyncio
import time


class SceneDescriptor:
    """Generates detailed scene descriptions from camera frames using either Florence-2 or GPT-4o-mini.

    Can use either Microsoft's Florence-2 model locally or OpenAI's GPT-4o-mini through API calls.
    """

    def __init__(self, use_local_model=True):
        """Initialize the scene descriptor with either local or OpenAI model."""

        self.use_local_model = use_local_model
        if use_local_model:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )

            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
            )

            self.model = self.model.to(self.device)

    def _encode_image(self, image):
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def describe_frame(self, frame, openai_client=None):
        """Generate a detailed description of the current camera frame."""

        try:
            image = Image.fromarray(frame)

            if self.use_local_model:

                loop = asyncio.get_event_loop()
                description = await loop.run_in_executor(
                    None, self._process_local_frame, image
                )
                return description
            else:
                return await self._process_openai_frame(image, openai_client)

        except Exception as e:
            print(f"Error describing frame: {e}")
            return None

    def _process_local_frame(self, image):
        """Process frame using local Florence model (runs in executor)."""

        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        description = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )

        return description["<MORE_DETAILED_CAPTION>"]

    async def _process_openai_frame(self, image, openai_client):
        """Process frame using OpenAI model (runs in executor)."""

        if openai_client is None:
            raise ValueError(
                "OpenAI client is required when use_local_model=False"
            )

        base64_image = self._encode_image(image)

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are K-Bot, a robot with camera vision. Describe what you see in natural, first-person language as if you are experiencing it directly through your camera. Be concise but detailed, focusing on the most important elements of the scene.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see through your camera? Describe it from your perspective as K-Bot.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=50,
        )

        return response.choices[0].message.content
