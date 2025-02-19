import torch
import base64
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as cv2


class SceneDescriptor:
    """Generates detailed scene descriptions from camera frames using either Florence-2 or GPT-4o-mini.

    Can use either Microsoft's Florence-2 model locally or OpenAI's GPT-4o-mini through API calls.
    """

    def __init__(self, use_local_model=False):
        """Initialize vision model.

        Args:
            use_local_model (bool): If True, uses local Florence model. If False, uses OpenAI GPT-4V.
        """
        self.use_local_model = use_local_model

        if use_local_model:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )

            print(f"Loading Florence model on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base",
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", trust_remote_code=True
            )
            print("Florence model loaded!")

    def _encode_image(self, image):
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def describe_frame(self, frame, openai_client=None):
        """Generate a detailed description of the current camera frame.

        Args:
            frame: RGB numpy array from camera
            openai_client: AsyncOpenAI client instance (required if use_local_model=False)

        Returns:
            str: Detailed natural language description of the scene
        """
        image = Image.fromarray(frame)

        if self.use_local_model:
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(self.device, self.torch_dtype)

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
        else:
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
