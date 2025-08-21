import torch
from diffusers import FluxInpaintPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize
from PIL import Image
import os
import argparse

# Configurazione iniziale
repo_id = "black-forest-labs/FLUX.1-dev"
path_model = "trongg/FLUX.1-dev_nsfw_FLUXTASTIC-v3.0"
dtype = torch.bfloat16

# Parser dei parametri da riga di comando
parser = argparse.ArgumentParser(description="Script per inpainting con FLUX.1 Dev + LoRA")
parser.add_argument("--prompt", type=str, required=True, help="Prompt testuale per la generazione")
parser.add_argument("--source", type=str, required=True, help="Percorso dell'immagine sorgente")
parser.add_argument("--lorafile", type=str, default=None, help="Nome del file LoRA da caricare (se presente)")
parser.add_argument("--lorascale", type=float, default=1.0, help="Scala applicata al LoRA")
parser.add_argument("--steps", type=int, default=28, help="Numero di passi di inferenza")
parser.add_argument("--cfg", type=float, default=7.0, help="Guidance scale")
parser.add_argument("--modifica", type=float, default=0.95, help="Strength del refusione immagine")
parser.add_argument("--risoluzione", type=str, default="1024,1024", help="Risoluzione di output (es: 1280,720)")
parser.add_argument("--ip_adapter", action="store_true", help="Attiva IP_adapter")
parser.add_argument("--scale_ip_adapter", type=float, default=0.7, help="Scala IP_adapter")
args = parser.parse_args()

# Caricamento pipeline base
pipe = FluxInpaintPipeline.from_pretrained(path_model, torch_dtype=dtype)

# Gestione LoRA
if args.lorafile:
    lora_path = os.path.join(".", "Lora", args.lorafile)
    if os.path.exists(lora_path):
        adapter_name = os.path.splitext(args.lorafile)[0]
        try:
            pipe.load_lora_weights("./Lora", weight_name=args.lorafile, adapter_name=adapter_name)
            pipe.set_adapters(adapter_name, adapter_weights=args.lorascale)
            pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=args.lorascale)
            print(f"✅ LoRA '{args.lorafile}' fuso con successo.")
        except Exception as e:
            print(f"❌ Errore nel caricamento/fusione LoRA: {e}")
    else:
        print(f"❌ File LoRA '{args.lorafile}' non trovato nella cartella ./Lora.")

# Quantizzazione transformer
print("🔄 Caricamento e quantizzazione del transformer...")
transformer = FluxTransformer2DModel.from_pretrained(path_model, subfolder="transformer", torch_dtype=dtype)
quantize(transformer, weights=qfloat8)
freeze(transformer)

# Quantizzazione text_encoder_2
print("🔄 Caricamento e quantizzazione di text_encoder_2...")
text_encoder_2 = T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_2", torch_dtype=dtype)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

# Sostituzione moduli nella pipeline

pipe.transformer = transformer

pipe.text_encoder_2 = text_encoder_2

# Gestione IP Adapter
img_IP_adapter = None
ip_adapter_active = False

if args.ip_adapter:
    try:
        pipe.load_ip_adapter(
            "XLabs-AI/flux-ip-adapter",
            weight_name="ip_adapter.safetensors",
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
        )
        pipe.set_ip_adapter_scale(args.scale_ip_adapter)
        img_IP_adapter = Image.open("image_riferimento.png").convert("RGB")
        ip_adapter_active = True
    except Exception as e:
        raise RuntimeError(f"❌ Errore nel caricamento dell'IP Adapter o immagine riferimento: {e}")



# Ottimizzazione uso memoria
pipe.enable_model_cpu_offload()

# Caricamento immagine e maschera
try:
    source_image = Image.open(args.source).convert("RGB")
    mask_image = Image.open('./mask.png').convert("L")
except Exception as e:
    raise RuntimeError(f"❌ Errore nel caricamento dell'immagine o della maschera: {e}")

# Parsing della risoluzione
w,h= source_image.size

if w>=h:
    #w*1024,h*X
    h= (1024*h)//w
    w=1024
else:
    w=(1024*w)//h
    h=1024

# Esecuzione dell'inpainting
kwargs = {
    "prompt": args.prompt,
    "image": source_image,
    "mask_image": mask_image,
    "num_inference_steps": args.steps,
    "guidance_scale": args.cfg,
    "strength": args.modifica,
    "width": w,
    "height": h,
}

if ip_adapter_active and img_IP_adapter:
    kwargs["ip_adapter_image"] = img_IP_adapter

image = pipe(**kwargs).images[0]

# Salvataggio risultato
output_path = "flux_inpainting.png"
image.save(output_path)
print(f"✅ Immagine generata salvata in: {output_path}")
