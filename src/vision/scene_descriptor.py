import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as cv2


class SceneDescriptor:
    """Generates detailed scene descriptions from camera frames using Florence-2.

    Uses Microsoft's Florence-2 model to generate rich, natural language descriptions
    of scenes without requiring API calls to external services.
    """

    def __init__(self):
        """Initialize Florence model and processor."""

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

    def describe_frame(self, frame):
        """Generate a detailed description of the current camera frame.

        Args:
            frame: RGB numpy array from camera

        Returns:
            str: Detailed natural language description of the scene
        """

        image = Image.fromarray(frame)

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
            generated_text, task=prompt, image_size=(image.width, image.height)
        )

        return description["<MORE_DETAILED_CAPTION>"]
