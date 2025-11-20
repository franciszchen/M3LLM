import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from .utils import load_image
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
    if not labels:
        labels = []
    out = []
    for i in range(n):
        lab = labels[i] if i < len(labels) else None
        out.append(_norm_one_label(lab, i))
    return out

def safe_load_to_bf16_cuda(img):
    try:
        t = load_image(img).to(torch.bfloat16).to("cuda")
        if t.dim() == 3:  # (C,H,W) -> (1,C,H,W)
            t = t.unsqueeze(0)
        return t
    except Exception as e:
        print(f"[Warning] failed to load image {img}: {e}")
        return None

class InternVL(VLMBase):
    def __init__(self,model_path,args):
        super().__init__()
        self.model =  AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="cuda",
                    #attn_implementation="flash_attention_2",
                    cache_dir = '/gpfs/radev/home/yf329/scratch/hf_models'
                    )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, cache_dir = '/gpfs/radev/home/yf329/scratch/hf_models')
        self.generation_config ={ 
            'max_new_tokens': args.max_new_tokens,
            'repetition_penalty': args.repetition_penalty,
            'temperature' : args.temperature,
            'top_p' : args.top_p
        }
    

    def process_messages(self, messages):
        prompt = messages.get("system", "") or ""
        text = messages.get("prompt", "") or ""

        # No images at all -> text-only
        if ("image" not in messages) and ("images" not in messages):
            return {"prompt": (prompt + ("\n" if prompt and text else "") + text), "pixel_values": None}

        # SINGLE IMAGE
        if "image" in messages and "images" not in messages:
            t = safe_load_to_bf16_cuda(messages["image"])
            if t is None:
                # fall back to text-only
                return {"prompt": (prompt + ("\n" if prompt and text else "") + text), "pixel_values": None}

            labels = messages.get("image_indices") or ["Fig-0"]
            label = _norm_fig_labels(labels, 1)[0]
            full_prompt = f"{prompt}\n{label}: <image>" + ("\n" + text if text else "")
            return {"prompt": full_prompt, "pixel_values": t}

        # MULTIPLE IMAGES
        raw_imgs = messages.get("images") or []
        if not isinstance(raw_imgs, (list, tuple)):
            raw_imgs = [raw_imgs]

        # Try to load all; keep only successful ones
        loaded = []
        for img in raw_imgs:
            t = safe_load_to_bf16_cuda(img)
            if t is not None:
                loaded.append(t)

        # Nothing loaded -> text-only
        if not loaded:
            print("[Warning] 'images' provided but none could be loaded; running text-only.")
            return {"prompt": (prompt + ("\n" if prompt and text else "") + text), "pixel_values": None}

        # Normalize labels to the number of successfully loaded images
        labels = _norm_fig_labels(messages.get("image_indices"), len(loaded))

        # Compose prompt header: one <image> token per loaded tensor
        lines = [f"{lab}: <image>" for lab in labels]
        full_prompt = (prompt + ("\n" if prompt else "") + "\n".join(lines) +
                    (("\n" + text) if text else ""))

        pixel_values = torch.cat(loaded, dim=0)  # (B,C,H,W)
        return {"prompt": full_prompt, "pixel_values": pixel_values}



    def generate_output(self, messages):
        llm_inputs = self.process_messages(messages)
        question = llm_inputs["prompt"]
        pixel_values = llm_inputs["pixel_values"]

        #If no images -> text-only inference
        if pixel_values is None:

            try:
                resp, _ = self.model.chat(
                    self.tokenizer,
                    None,                      # no images
                    question,
                    self.generation_config,
                    history=None,
                    return_history=True
                )
                return resp
            except Exception:
                inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=getattr(self, "max_new_tokens", 512),
                    temperature=getattr(self, "temperature", 0.0),
                    top_p=getattr(self, "top_p", 1.0),
                    repetition_penalty=getattr(self, "repetition_penalty", 1.0),
                )
                cut = inputs["input_ids"].shape[1]
                out = self.tokenizer.decode(gen_ids[0][cut:], skip_special_tokens=True)
                return out

        # With images
        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            history=None,
            return_history=True
        )
        return response

    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
