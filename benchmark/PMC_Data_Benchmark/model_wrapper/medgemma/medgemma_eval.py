import os
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForImageTextToText,
    AutoProcessor,
)
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

#from .utils import load_image  # optional; we fall back to PIL if needed
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
    labels = labels or []
    out = []
    for i in range(n):
        lab = labels[i] if i < len(labels) else None
        out.append(_norm_one_label(lab, i))
    return out

def _to_pil(img_obj):
    """
    Accepts a path/URL/PIL/tensor and returns a PIL.Image or raises.
    Prefer PIL for MedGemma (processor handles conversion to tensors).
    """
    # Try our utils first (may return tensor); if tensor, convert via PIL
    try:
        t = load_image(img_obj)
        if isinstance(t, torch.Tensor):
            # NHWC or CHW -> PIL
            if t.dim() == 4:
                t = t[0]
            if t.dim() == 3:
                # assume CHW in [0,1] or [0,255]
                c, h, w = t.shape
                if c == 3:
                    arr = t.detach().cpu().float()
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).clamp(0, 255)
                    arr = arr.byte().permute(1, 2, 0).numpy()
                    return Image.fromarray(arr)
        elif isinstance(t, Image.Image):
            return t
    except Exception:
        pass

    # Fallback pure PIL
    if isinstance(img_obj, Image.Image):
        return img_obj
    if isinstance(img_obj, (str, os.PathLike)):
        return Image.open(img_obj).convert("RGB")
    raise ValueError(f"Unsupported image type for MedGemma: {type(img_obj)}")


class MedGemma(VLMBase):
    """
    MedGemma wrapper for your evaluation codebase.

    - Multimodal variants:  google/medgemma-4b-it, google/medgemma-27b-it
      (loaded with AutoModelForImageTextToText + AutoProcessor)
    - Text-only variant:    google/medgemma-27b-text-it
      (loaded with AutoModelForCausalLM + AutoTokenizer)

    """

    def __init__(self, model_path, args):
        super().__init__()
        self.model_id = model_path
        self.is_text_only = ("-text-" in model_path) or ("text" in model_path.split("/")[-1])

        device_map = getattr(args, "device_map", "auto")
        use_4bit = bool(getattr(args, "use_4bit", False))

        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if use_4bit and _HAS_BNB:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        if self.is_text_only:
            # Text-only
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.processor = None
        else:
            # Vision+Language
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tokenizer = None

        self.generation_config = {
            "max_new_tokens": getattr(args, "max_new_tokens", 512),
            "repetition_penalty": getattr(args, "repetition_penalty", 1.0),
            "temperature": getattr(args, "temperature", 0.0),
            "top_p": getattr(args, "top_p", 1.0),
        }

    # ----- message preparation -----

    def _build_chat_messages(self, messages):
        """
        Convert your evaluation message dict into MedGemma chat messages:
          [
            {"role":"system","content":[{"type":"text","text": "..."}]},
            {"role":"user","content":[{"type":"text","text":"Fig-1:"},{type:"image","image": PIL}, ... , {"type":"text","text": prompt}]}
          ]
        """
        system_text = (messages.get("system") or "").strip()
        user_text = (messages.get("prompt") or "").strip()

        # Collect images (if any)
        raw_imgs = []
        if "image" in messages:
            raw_imgs = [messages["image"]]
        elif "images" in messages:
            imgs = messages["images"]
            raw_imgs = imgs if isinstance(imgs, (list, tuple)) else [imgs]

        pil_images = []
        for x in raw_imgs:
            try:
                pil_images.append(_to_pil(x))
            except Exception as e:
                print(f"[Warning] MedGemma: failed to load image: {e}")

        # Build chat messages
        chat = []
        if system_text:
            chat.append({"role": "system", "content": [{"type": "text", "text": system_text}]})

        user_content = []
        if pil_images and not self.is_text_only:
            labels = _norm_fig_labels(messages.get("image_indices"), len(pil_images))
            for lab, pil in zip(labels, pil_images):
                user_content.append({"type": "text", "text": f"{lab}:"})
                user_content.append({"type": "image", "image": pil})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            if pil_images and self.is_text_only:
                print("[Warning] MedGemma text-only variant: images provided but will be ignored.")
            # Text only
            combined = (system_text + ("\n" if system_text and user_text else "") + user_text).strip()
            # Put in the user turn (system is already added above)
            user_content.append({"type": "text", "text": (user_text or combined)})

        chat.append({"role": "user", "content": user_content})
        return chat

    # ----- core inference -----

    def _gen_kwargs(self):
        # Respect provided generation params; set do_sample only if temperature > 0
        return dict(
            max_new_tokens=self.generation_config["max_new_tokens"],
            do_sample=self.generation_config["temperature"] > 0.0,
            temperature=self.generation_config["temperature"],
            top_p=self.generation_config["top_p"],
            repetition_penalty=self.generation_config["repetition_penalty"],
        )

    def generate_output(self, messages):
        chat_messages = self._build_chat_messages(messages)

        if self.is_text_only:
            tokenizer = self.tokenizer
            inputs = tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out_ids = self.model.generate(**inputs, **self._gen_kwargs())
                out_ids = out_ids[0][input_len:]
            response = tokenizer.decode(out_ids, skip_special_tokens=True)
        else:
            processor = self.processor
            # NOTE: dtype=bfloat16 like the official example
            inputs = processor.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out_ids = self.model.generate(**inputs, **self._gen_kwargs())
                out_ids = out_ids[0][input_len:]
            response = processor.decode(out_ids, skip_special_tokens=True)

        try:
            sys_prompt = messages.get("system", "") or ""
            q_text = messages.get("prompt", "") or ""
            header = sys_prompt + ("\n" if sys_prompt and q_text else "") + q_text
        except Exception:
            pass

        return response

    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]
