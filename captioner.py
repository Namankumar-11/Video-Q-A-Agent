"""
captioner.py
------------
Every CAPTION_EVERY_N_SECONDS, we caption a frame in plain English. This is
what lets the system answer *semantic* questions ("was anyone wearing a
helmet?") that raw object-detection labels alone can't cover — YOLO gives
you "person", a caption can give you "a person wearing a red helmet
crossing the street."

NOTE: Groq's current model lineup has no vision-capable chat model (its
earlier vision preview models were deprecated with no direct replacement
as of this writing). So captioning runs locally instead, using a small
BLIP image-captioning model via `transformers` — no API key needed for
this part, and it removes a dependency on Groq's model availability
changing again later. If a Groq vision model becomes available again,
swap this out for an API call the same way the original design did.
"""

import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import config


class SceneCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        # Base BLIP model — ~990MB, runs fine on CPU, no GPU required.
        # First run downloads the weights; cached locally after that.
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def caption_frame(self, frame, active_tracks):
        """
        active_tracks: list of class_names currently visible (from detector.py).
        BLIP's base captioning model doesn't take a free-text grounding prompt
        the way a chat-style VLM does, so instead of injecting detected objects
        into the model's input, we generate the caption independently and then
        append the tracker's detections as a separate, clearly-labeled fact —
        this keeps the tracker's ground-truth objects from being diluted or
        contradicted by whatever the caption model happens to describe.
        """
        # OpenCV frames are BGR numpy arrays; PIL/transformers expect RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        inputs = self.processor(pil_image, return_tensors="pt")
        output_ids = self.model.generate(**inputs, max_new_tokens=40)
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

        detected_objects = ", ".join(sorted(set(active_tracks))) or "none"
        return f"{caption} (Objects detected by tracker: {detected_objects})"
