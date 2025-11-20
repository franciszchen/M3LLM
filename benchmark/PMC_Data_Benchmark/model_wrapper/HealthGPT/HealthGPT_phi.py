import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import transformers
import tokenizers

from PIL import Image
from tqdm import tqdm
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


from .llava.constants import  IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from .llava.peft import LoraConfig, get_peft_model
from .llava import conversation as conversation_lib
from .llava.model import *
from .llava.mm_utils import tokenizer_image_token,process_images
from .llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM
from .utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square,com_vision_args
from model_wrapper.vlm_base import VLMBase

def _norm_one_label(x, fallback_i):
    """Return 'Fig-<n>' from x (accepts 'Fig-3', '3', 3, etc.)."""
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
    """Normalize/pad/truncate labels to length n."""
    labels = [] if labels is None else list(labels)
    out = []
    for i in range(n):
        lab = labels[i] if i < len(labels) else None
        out.append(_norm_one_label(lab, i))
    return out

def _open_to_rgb(img):
    """Open path/PIL to RGB; return None if it fails."""
    try:
        if isinstance(img, str):
            return Image.open(img).convert('RGB')
        elif isinstance(img, Image.Image):
            return img.convert('RGB')
        else:
            return None
    except Exception as e:
        print(f"[Warning] failed to open image {img}: {e}")
        return None

class HealthGPT(VLMBase):
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path = 'microsoft/phi-4',
        #attn_implementation="flash_attention_2",
        torch_dtype= torch.float16,
        device_map="cuda",
        cache_dir = '/gpfs/radev/home/yf329/scratch/hf_models'
    )
        print("load model done")
        lora_config = LoraConfig(
            r= 32,
            lora_alpha=64,
            target_modules=find_all_linear_names(self.llm),
            lora_dropout=0.0,
            bias='none',
            task_type="CAUSAL_LM",
            lora_nums=4,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        print("load lora done")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            'microsoft/phi-4',
            padding_side="right",
            use_fast=False,
            cache_dir = '/gpfs/radev/home/yf329/scratch/hf_models'
        )
        print("load tokenizer done")

        num_new_tokens = add_special_tokens_and_resize_model(self.tokenizer, self.llm, 8192)
        print(f"Number of new tokens added for unified task: {num_new_tokens}")

        com_vision_args.model_name_or_path = model_path
        com_vision_args.vision_tower = '/gpfs/radev/home/yf329/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
        com_vision_args.version = "phi4_instruct"

        self.llm.get_model().initialize_vision_modules(model_args=com_vision_args)
        self.llm.get_vision_tower().to(dtype=torch.float16)
        self.llm.get_model().mm_projector.to(dtype=torch.float16)
        print("load vision tower done")

        self.llm = load_weights(self.llm, "/gpfs/radev/home/yf329/scratch/hf_models/models--lintw--HealthGPT-L14/snapshots/a868fe949f4552281f8784d668b353826c4b0c38/com_hlora_weights_phi4.bin")
        print("load weights done")
        self.llm.eval()
        self.llm.to(dtype=torch.float16).cuda()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens

    def process_messages(self, messages):
        conv = conversation_lib.conv_templates["phi4_instruct"].copy()
        conv.messages = []
        if "system" in messages:
            conv.system = messages["system"]

        user_text = messages.get("prompt", "") or ""

        # No images
        if "image" not in messages and "images" not in messages:
            conv.append_message(conv.roles[0], user_text)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt(), None

        # Gather images into a list
        raw_imgs = []
        if "images" in messages:
            raw_imgs = messages["images"]
        else:  # "image"
            raw_imgs = [messages["image"]]

        # Normalize labels to the intended number of images
        labels = _norm_fig_labels(messages.get("image_indices"), len(raw_imgs))

        # Load images; ONLY emit a DEFAULT_IMAGE_TOKEN line if the image actually loads
        imgs = []
        lines = []
        for img, lab in zip(raw_imgs, labels):
            pil = _open_to_rgb(img)
            if pil is None:
                print(f"[Warning] skipping missing/bad image paired with label {lab}")
                continue
            imgs.append(pil)
            # IMPORTANT: DEFAULT_IMAGE_TOKEN appears exactly once per loaded image
            lines.append(f"{lab}: {DEFAULT_IMAGE_TOKEN}")

        # If none loaded, just send text
        if not imgs:
            conv.append_message(conv.roles[0], user_text)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt(), None

        # Compose final prompt: one labeled token per loaded image, then the user text
        prompt = ("\n".join(lines) + ("\n" if user_text else "") + user_text)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        final_prompt = conv.get_prompt()
        return final_prompt, imgs


    def generate_output(self,messages):
        prompt,imgs = self.process_messages(messages)
        print(prompt)
        if imgs:
            # imgs = imgs[0]
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze_(0).cuda()
            imgs = [expand2square(img, tuple(int(x*255) for x in self.llm.get_vision_tower().image_processor.image_mean)) for img in imgs]
            imgs = self.llm.get_vision_tower().image_processor.preprocess(imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.float16, device='cuda', non_blocking=True)

        else:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze_(0).cuda()
            imgs = None

        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            output_ids = self.llm.base_model.model.generate(input_ids,images=imgs,do_sample=do_sample,num_beams=5,max_new_tokens=self.max_tokens,temperature = self.temperature,top_p = self.top_p,repetition_penalty = self.repetition_penalty,use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_outputs(self,messages_list):
        outputs = []
        for messages in tqdm(messages_list):
            output = self.generate_output(messages)
            outputs.append(output)
        return outputs