import torch
import random
from PIL import Image
import os
import argparse
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
from controlnet_aux import OpenposeDetector

# ====== NOISE SCHEDULER ======
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# ====== ARGPARSE ======
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--risoluzione", type=str, required=True)
parser.add_argument("--output", type=str, default="output.png")
parser.add_argument("--lorascale", type=float, default=0.8)
parser.add_argument("--scaleIP", type=float, default=0.5)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--cfg", type=float, default=7.5)
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--model", type=str, default="hyperRealism_30.safetensors")
parser.add_argument("--lora", type=str, default="No_lora")
parser.add_argument("--ipadapter", type=str, default=None)
parser.add_argument("--pose", type=str, default=None)
args = parser.parse_args()

# ====== PARAMETRI BASE ======
scalalora = args.lorascale
scalaIP = args.scaleIP
steps_value = args.steps
cfg_value = args.cfg
seed = args.seed if args.seed != -1 else random.randint(1, 100000)
generator = torch.Generator(device="cuda").manual_seed(seed)

# ====== PERCORSI ======
model_path = f"./modelli/{args.model}"
lora_path = f"./Lora/{args.lora}"
ip_adapter_image_path = args.ipadapter
image_pose_path = args.pose
checkpoint = "lllyasviel/control_v11p_sd15_openpose"

prompt = args.prompt
negative_prompt = "low quality, hands"

print(f"[INFO] Prompt: {prompt}")
print(f"[INFO] Seed: {seed}")

# ====== CONTROLNET ======
pipeline = None
control_image = None

if image_pose_path and os.path.isfile(image_pose_path):
    print("[INFO] Uso ControlNet SD1.5 con OpenPose")
    image_pose = Image.open(image_pose_path)
    processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cuda")
    control_image = processor(image_pose, hand_and_face=True)

    if isinstance(control_image, torch.Tensor):
        control_image = control_image.squeeze().cpu().numpy()
        control_image = Image.fromarray((control_image * 255).astype("uint8"))

    control_image.save("control.png")

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipeline = StableDiffusionControlNetPipeline.from_single_file(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

# ====== PIPELINE BASE (senza ControlNet) ======
if pipeline is None:
    pipeline = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    print(f"[INFO] Model caricato: {model_path}")

# ====== LoRA ======
if args.lora and args.lora != "No_lora" and os.path.isfile(lora_path):
    print(f"[INFO] Applico LoRA: {args.lora} con scala {scalalora}")
    lora_name = os.path.splitext(os.path.basename(lora_path))[0].replace(".", "_")
    pipeline.load_lora_weights(lora_path, adapter_name=lora_name)
    pipeline.set_adapters(adapter_names=lora_name, adapter_weights=scalalora)
    pipeline.fuse_lora()

# ====== IP ADAPTER SD1.5 ======
has_ip_adapter = False
ip_image = None

if ip_adapter_image_path and os.path.isfile(ip_adapter_image_path):
    print("[INFO] Carico IP-Adapter SD1.5")
    ip_image = Image.open(ip_adapter_image_path).convert("RGB")
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-full-face_sd15.bin"
    )
    pipeline.set_ip_adapter_scale(scalaIP)
    has_ip_adapter = True

# ====== SCHEDULER & CUDA ======
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.use_karras_sigmas = True
pipeline.to("cuda")

# ====== GENERAZIONE ======
try:
    width_str, height_str = args.risoluzione.split(',')
    w, h = int(width_str.strip()), int(height_str.strip())

    # Prepara gli argomenti comuni
    gen_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": cfg_value,
        "width": w,
        "height": h,
        "num_inference_steps": steps_value,
        "generator": generator,
    }

    # Aggiungi ControlNet image se presente
    if control_image is not None:
        gen_args["control_image"] = control_image

    # Aggiungi IP-Adapter image se presente
    if has_ip_adapter and ip_image is not None:
        gen_args["ip_adapter_image"] = ip_image

    print("[INFO] Generazione con parametri:", list(gen_args.keys()))
    result = pipeline(**gen_args).images[0]

    result.save(args.output)
    print(f"[SUCCESS] Immagine generata: {args.output}")

except Exception as e:
    print(f"[ERROR] Errore durante la generazione: {e}")
    import traceback
    traceback.print_exc()
