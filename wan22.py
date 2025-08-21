# Esempio: usare una versione FP16 "diffusers-ready" (ipotetica)
import torch
from diffusers import WanImageToVideoPipeline
from PIL import Image
from diffusers.utils import export_to_video   # requires imageio-ffmpeg

MODEL_ID = "Phr00t/WAN2.2-14B-Rapid-AllInOne"   # es. repo FP16 (diffusers-ready)
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)

# Low-VRAM tweaks
pipe.enable_sequential_cpu_offload()   # scarica su CPU quando possibile
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Input
image = Image.open("image.png").convert("RGB")


# WAN 2.2 preferisce multipli di 64 — 720x720 funziona
w,h= image.size
wg,wg=720,720

if w>h:
    # w * 720: h*X
    hg= (720*h)//w
else:
    wg=(720*w)//h

image = image.resize((wg,hg), Image.LANCZOS)

# Parametri (mappati dalle raccomandazioni ComfyUI)
steps = 4             # come consigliato
cfg = 1               # in ComfyUI => qui corrisponde a guidance_scale (usa 1.0 per estrema fedeltà)
guidance_scale = float(cfg)   # 1.0
num_frames = 81       # es. 81 (modifica come vuoi)
sampler = "euler_a"   # Se il repo fornisce scheduler, default va bene

out = pipe(
    prompt="a girl walking",
    image=image,
    negative_prompt="low resolution, low quality",
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    num_frames=num_frames
)

# Salvataggio in mp4 (richiede imageio[ffmpeg])
export_to_video(out.frames, "wan22_out.mp4", fps=24)


