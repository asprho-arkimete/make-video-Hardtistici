# !pip install opencv-python transformers accelerate diffusers controlnet_aux

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
import torch
import random
import argparse
import os

# --- Argparse ---
parser = argparse.ArgumentParser()
parser.add_argument("--path_image", type=str, required=True, help="Path to input image")
parser.add_argument("--prompt", type=str, required=True, help="Positive prompt")
parser.add_argument("--model", type=str, required=True, help="Base model path (diffusers format)")
parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale")
parser.add_argument("--modifica", type=float, default=0.6, help="Strength (modifica)")
parser.add_argument("--seed", type=int, default=random.randint(1000, 999999), help="Random seed")
parser.add_argument("--risoluzione", type=str, default="512,512", help="Resolution: width,height")
parser.add_argument("--lora", type=str, default="No_Lora", help="LoRA name")
parser.add_argument("--scale_lora", type=float, default=0.8, help="Scale for LoRA")
parser.add_argument("--ip_adapter", type=str, default=None, help="Path to image for IP-Adapter")  # <--- AGGIUNGI QUESTO
parser.add_argument("--scale_ip_adapter", type=float, default=0.5, help="Scale for IP-Adapter")   # <--- E QUESTO

args = parser.parse_args()

# --- Estrazione argomenti ---
imagepath = args.path_image
prompt = args.prompt
steps = args.steps
cfg = args.cfg
modifica = args.modifica
seed = args.seed
base_model = f"./modelli/{args.model}"
width, height = map(int, args.risoluzione.split(','))

lorapath = f"./Lora/{args.lora}"
scale_lora = args.scale_lora

ip_adapter_path = args.ip_adapter
ip_adapter_scale = args.scale_ip_adapter

negative_prompt = (
    "low quality, blurry, distorted anatomy, deformed face, unrealistic nipples, bad hands, missing limbs, "
    "bad proportions, extra limbs, text, watermark"
)

checkpoint = "lllyasviel/control_v11p_sd15_openpose"
print(f"[INFO] Generazione con seed: {seed}")
print("[INFO] Uso ControlNet SD1.5 con OpenPose")

# --- Carica immagine e genera control image con OpenPose ---
image_pose = Image.open(imagepath).convert("RGB")
w, h = image_pose.size

if w >= h:
    hc = (960 * h) // w
    wc = 960
else:
    wc = (960 * w) // h
    hc = 960

image_pose = image_pose.resize((wc, hc), Image.BICUBIC)

processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cuda")
control_image = processor(image_pose, hand_and_face=True)

if isinstance(control_image, torch.Tensor):
    control_image = control_image.squeeze().cpu().numpy()
    control_image = Image.fromarray((control_image * 255).astype("uint8"))
control_image.save("control.png")

# --- Carica ControlNet e pipeline ---
controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# --- LoRA ---
if args.lora.lower() != "no_lora":
    pipe.load_lora_weights(lorapath, adapter_name="lora")
    pipe.set_adapters("lora", adapter_weights=scale_lora)
    pipe.fuse_lora(adapter_names=["lora"], lora_scale=scale_lora)

# --- IP Adapter ---
ip_image = None
if ip_adapter_path is not None:
    print("[INFO] Uso IP-Adapter")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-full-face_sd15.bin"
    )
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    ip_image = Image.open(ip_adapter_path).convert("RGB")
    w, h = ip_image.size
    if w >= h:
        hip = (960 * h) // w
        wip = 960
    else:
        wip = (960 * w) // h
        hip = 960
    ip_image = ip_image.resize((wip, hip), Image.BICUBIC)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
# --- Generazione immagine ---
generator = torch.manual_seed(seed)
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image_pose,
    control_image=control_image,
    num_inference_steps=steps,
    guidance_scale=cfg,
    strength=modifica,
    generator=generator,
    width=width,
    height=height,
    ip_adapter_image=ip_image,
).images[0]

result.save("sd_img2img.png")
print("[INFO] Immagine salvata come sd_img2img.png")
