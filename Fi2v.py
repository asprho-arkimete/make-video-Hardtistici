import torch
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import SiglipImageProcessor, SiglipVisionModel
from PIL import Image
import os
import numpy as np
from moviepy import ImageSequenceClip

def make_divisible(x, divisor=8):
    return x - (x % divisor)

# === Configurazione iniziale ===
dtype = torch.float16
onload_device = "cuda"
offload_device = "cpu"

# === Attiva TensorFloat-32 (se disponibile) ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# === Caricamento componenti pesanti ===
transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePack_F1_I2V_HY_20250503", torch_dtype=dtype
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=dtype
)
feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
)

# === Creazione pipeline ===
pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    image_encoder=image_encoder,
    feature_extractor=feature_extractor,
    torch_dtype=dtype,
)

# === Offloading intelligente solo su componenti sicuri ===
for module in [pipe.text_encoder, pipe.text_encoder_2, pipe.transformer]:
    apply_group_offloading(
        module,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
        use_stream=True,
        low_cpu_mem_usage=True,
    )

# === Spostamento su CUDA senza offloading per VAE e image_encoder ===
pipe.image_encoder.to(onload_device)
pipe.vae.to(onload_device)
pipe.vae.enable_slicing()  # 🔥 Riduce l'uso di VRAM durante il decoding

# === Caricamento immagine ===
image_source = Image.open("image_flux_w.png")
w, h = image_source.size
max_side = max(w, h)
min_side = min(w, h)
ris = 480  # Risoluzione massima: 480p
new_min_side = int(ris * min_side / max_side)
new_w, new_h = (ris, new_min_side) if w >= h else (new_min_side, ris)
new_w = make_divisible(new_w)
new_h = make_divisible(new_h)
image_source = image_source.resize((new_w, new_h), Image.BICUBIC)

# === Prompt personalizzato ===
prompt = "A girl walking"

# === Parametri video ottimizzati ===
secondi = 2  # Durata del video in secondi
fps = 16       # Fotogrammi al secondo
num_frames = int(secondi * fps)  # → 24 frames
print(f"Genererò {num_frames} fotogrammi ({secondi} secondi a {fps} FPS)")

# === Generazione video con ottimizzazioni ===
with torch.inference_mode():
    with torch.autocast("cuda"):
        output = pipe(
            image=image_source,
            prompt=prompt,
            height=new_h,
            width=new_w,
            num_frames=num_frames,
            num_inference_steps=3,  # Ridotto per velocità e memoria
            guidance_scale=9.0,
            generator=torch.Generator().manual_seed(0),
            sampling_type="vanilla",
        ).frames[0]

# === Pulizia cache CUDA ===
torch.cuda.empty_cache()

print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")

# === Salvataggio fotogrammi come immagini PNG ===
os.makedirs("frames", exist_ok=True)

# Il tensor è in formato: [num_frames, channels, height, width]
# Lo convertiamo a formato immagine: [height, width, channels] (HWC)
output_images = (output * 255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()

for idx, frame in enumerate(output_images):
    frame = frame.transpose(1, 2, 0)  # CHW → HWC
    frame_image = Image.fromarray(frame)
    frame_image.save(f"frames/frame_{idx:04d}.png")

# === Creazione del video con moviepy ===
image_files = [f"frames/frame_{i:04d}.png" for i in range(len(output_images))]
clip = ImageSequenceClip(image_files, fps=fps)
clip.write_videofile("output_video.mp4", codec="libx264", fps=fps)

print("Video esportato correttamente come 'output_video.mp4'")