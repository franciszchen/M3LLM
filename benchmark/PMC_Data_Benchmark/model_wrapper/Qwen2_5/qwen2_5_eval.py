import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from model_wrapper.vlm_base import VLMBase

def _norm_one_label(x, fallback_i):
    if x is None:
        return f"Fig-{fallback_i}"
    s = str(x).strip()
    if s.lower().startswith("fig-"):
        s = s.split("-", 1)[1]
    try:
        return f"Fig-{int(s)}"
    except Exception:
        return s if s.lower().startswith("fig-") else f"Fig-{s}"

def _norm_fig_labels(labels, n):
    labels = [] if labels is None else list(labels)
    out = []
    for i in range(n):
        lab = labels[i] if i < len(labels) else None
        out.append(_norm_one_label(lab, i))
    return out

def _open_image_safe(img):
    """Return a PIL.Image in RGB or None if it cannot be opened."""
    try:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, (bytes, bytearray, io.BytesIO)):
            return Image.open(io.BytesIO(img if isinstance(img, (bytes, bytearray)) else img.getvalue())).convert("RGB")
        if isinstance(img, str):
            # local path or URL – Qwen can also accept URLs; here we try local open
            return Image.open(img).convert("RGB")
        return None
    except Exception as e:
        print(f"[Warning] failed to load image {img!r}: {e}")
        return None

class Qwen2_5_VL(VLMBase):
    def __init__(self, model_path, args):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path
        )
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def generate_output(self, messages: dict) -> str:
        user_text = messages.get("prompt", "") or ""
        system_text = messages.get("system", "") or ""

        # Gather images (normalize to list)
        imgs_in = []
        if "images" in messages:
            imgs_in = messages["images"] if isinstance(messages["images"], list) else [messages["images"]]
        elif "image" in messages:
            imgs_in = [messages["image"]]

        # Open images safely; skip ones that fail to load
        pil_images = []
        for img in imgs_in:
            pil = _open_image_safe(img)
            if pil is not None:
                pil_images.append(pil)

        content = []

        # If we have at least one image
        if pil_images:
            labels = _norm_fig_labels(messages.get("image_indices"), len(pil_images))
            # Add a label line (plain text) followed by the image for each image
            for lab, pil in zip(labels, pil_images):
                content.append({"type": "text", "text": f"{lab}:"})
                content.append({"type": "image", "image": pil})
            if user_text:
                content.append({"type": "text", "text": user_text})
        else:
            # Text-only fallback
            content.append({"type": "text", "text": user_text})

        full_messages = []
        if system_text:
            full_messages.append({"role": "system", "content": system_text})
        full_messages.append({"role": "user", "content": content})

        # Build prompt & vision inputs
        prompt = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(full_messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0
        )

        # Trim prompt tokens
        cut_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            cut_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]