# from transformers import AutoModelForSequenceClassification

# # Loading a model will automatically download and cache it
# model = AutoModelForSequenceClassification.from_pretrained("lintw/HealthGPT-M3", cache_dir='/gpfs/radev/home/yf329/scratch/hf_models')

from huggingface_hub import hf_hub_download

# Download to a custom cache directory
custom_cache_path = "/gpfs/radev/home/yf329/scratch/hf_models"
file_path = hf_hub_download(repo_id="lintw/HealthGPT-L14", filename="com_hlora_weights_phi4.bin", cache_dir=custom_cache_path)
print(f"File downloaded to: {file_path}")
