import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from model_wrapper.vlm_base import VLMBase


class Llava(VLMBase):
    def __init__(self, model_path, args):
        self.processor = LlavaNextProcessor.from_pretrained(
            model_path, cache_dir=args.cache_dir
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=args.cache_dir
        ).to("cuda")

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    # --- helpers for labels ---
    @staticmethod
    def _norm_one_label(x, fallback_i: int) -> str:
        """Return a 'Fig-<n>' label from x (accepts 'Fig-3', '3', 3, etc.)."""
        if x is None:
            return f"Fig-{fallback_i}"
        s = str(x).strip()
        if s.lower().startswith("fig-"):
            s = s.split("-", 1)[1]
        try:
            return f"Fig-{int(s)}"
        except Exception:
            return s if s.lower().startswith("fig-") else f"Fig-{s}"

    def _norm_fig_labels(self, labels, n: int):
        """Normalize/pad/truncate labels list to length n."""
        labels = [] if labels is None else list(labels)
        out = []
        for i in range(n):
            lab = labels[i] if i < len(labels) else None
            out.append(self._norm_one_label(lab, i))
        return out

    def generate_output(self, messages: dict) -> str:
        from PIL import Image

        system = messages.get("system", "") or ""
        user_prompt = messages.get("prompt", "") or ""
        prompt = system + ("\n" if (system and user_prompt) else "") + user_prompt

        # collect images into a list (even if single)
        raw_imgs = []
        if "images" in messages:
            raw_imgs = messages["images"] if isinstance(messages["images"], list) else [messages["images"]]
        elif "image" in messages:
            raw_imgs = [messages["image"]]

        # no images → text only
        if not raw_imgs:
            messages_input = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            input_text = self.processor.apply_chat_template(messages_input, add_generation_prompt=True)
            inputs = self.processor(None, input_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        else:
            pil_images = []
            for img in raw_imgs:
                pil_images.append(Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB"))

            n = len(pil_images)

            labels = list(messages.get("image_indices", []))
            if len(labels) < n:
                labels.extend([f"Fig-{i}" for i in range(len(labels), n)])
            elif len(labels) > n:
                labels = labels[:n]

            # Build content
            content = []
            for lab in labels:
                content.append({"type": "text", "text": f"{lab}:"})
                content.append({"type": "image"}) 
            if prompt:
                content.append({"type": "text", "text": prompt})

            messages_input = [{"role": "user", "content": content}]
            images_for_processor = pil_images if n > 1 else pil_images[0]

            input_text = self.processor.apply_chat_template(messages_input, add_generation_prompt=True)
            print(input_text)
            inputs = self.processor(
                images_for_processor,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p
        )

        out_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            out_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
