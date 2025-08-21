import torch
from diffusers import WanImageToVideoPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from PIL import Image

model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Ottimizzazioni
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload(pipe)

apply_group_offloading(
    pipe.text_encoder,
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=4
)

apply_group_offloading(
    pipe.transformer,
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    num_blocks_per_group=4
)

# Prompt + immagine
prompt = "a girl walking"
image = Image.open("./image_flux_w.png").convert("RGB").resize((512, 512))

# Inference
video = pipe(prompt=prompt, image=image).frames

# Salvataggio (es. GIF)
from diffusers.utils import export_to_gif
export_to_gif(video, "output.gif")