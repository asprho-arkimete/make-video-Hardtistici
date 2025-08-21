import torch
import numpy as np
from PIL import Image
import imageio

from diffusers import (
    DDIMScheduler,
    MotionAdapter,
    PIAPipeline,EulerDiscreteScheduler
)
from diffusers.utils import export_to_video, load_image  # nota: corretto 'exort_video_mp4' in 'export_to_video'

# Percorsi
input_image_path = "./h//image_flux2.png"
output_gif_path = "animated_output.gif"
model_base1= "SG161222/Realistic_Vision_V6.0_B1_noVAE"
model_base2= "stablediffusionapi/epicbabes_realistic"

# Prompt
prompt = "blowjob,a woman sucks a dick"
negative_prompt = "wrong white balance, dark, sketches, worst quality, low quality"

# Caricamento immagine
image = Image.open(input_image_path).convert("RGB")
image = image.resize((512, 512))

# Caricamento adapter e pipeline
adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter")
pipe = PIAPipeline.from_pretrained(model_base2, motion_adapter=adapter)

# Caricamento LoRA
lora_path = "./"  # Cartella dove si trova il LoRA
lora_weights = "TongueOut-CM.safetensors"
pipe.load_lora_weights(lora_path, weights=lora_weights, adapter_name="bj", adapter_weight=0.7)
pipe.set_adapters(["bj"])  # Solo il nome
pipe.fuse_lora()

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)# Ottimizzazioni memoria
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Generazione
generator = torch.Generator("cpu").manual_seed(0)
output = pipe(image=image, prompt=prompt, negative_prompt=negative_prompt, generator=generator)

# Salvataggio video
frames = output.frames[0]
export_to_video(frames, "pia-freeinit-animation.mp4")  # Corretto nome funzione
