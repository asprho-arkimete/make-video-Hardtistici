import argparse
import os
import torch
from PIL import Image

from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize


def main(args):
    repo_id = "black-forest-labs/FLUX.1-dev"
    dtype = torch.bfloat16

    # 1. Caricamento pipeline base
    pipe = FluxPipeline.from_pretrained(repo_id, torch_dtype=dtype)

    # 2. Caricamento e quantizzazione del transformer
    pathmodel = "./modelli/fluxedUpFluxNSFW_41DevFp8.safetensors"
    transformer = FluxTransformer2DModel.from_single_file(pathmodel, torch_dtype=dtype)
    quantize(transformer, weights=qfloat8)
    freeze(transformer)
    pipe.transformer = transformer

    # 3. Caricamento e quantizzazione del text encoder 2
    text_encoder_2 = T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_2", torch_dtype=dtype)
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)
    pipe.text_encoder_2 = text_encoder_2

    # 4. Caricamento IP Adapter (se richiesto)
    image_reference = None
    if args.use_ip_adapter and args.ip_adapter_image:
        if not os.path.exists(args.ip_adapter_image):
            print(f"❌ Immagine IP Adapter non trovata: {args.ip_adapter_image}")
            return

        print(f"🔌 Caricamento IP Adapter da: {args.ip_adapter_image}")
        pipe.load_ip_adapter(
            "XLabs-AI/flux-ip-adapter",
            weight_name="ip_adapter.safetensors",
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
        )
        image_reference = Image.open(args.ip_adapter_image).convert("RGB")
        pipe.set_ip_adapter_scale(args.scale_ip_adapter)
        print(f"🧠 IP Adapter attivo con scala: {args.scale_ip_adapter}")

    # 5. Caricamento LoRA
    lora_file = args.lora.strip()
    if lora_file.lower() != "no_lora":
        adapter_name = os.path.splitext(lora_file)[0]
        lora_scale = max(0.0, min(float(args.scale), 2.0))
        print(f"🔗 Caricamento LoRA: {lora_file} (scala={lora_scale})")

        try:
            pipe.load_lora_weights("./Lora", weight_name=lora_file, adapter_name=adapter_name)
            print("✅ Caricamento LoRA OK")
        except Exception as e:
            print(f"❌ Caricamento LoRA fallito: {e}")

        try:
            pipe.set_adapters(adapter_name, adapter_weights=lora_scale)
            print("✅ Attivazione LoRA riuscita")
        except Exception as e1:
            print(f"❌ Attivazione LoRA fallita: {e1}")

        try:
            pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=lora_scale)
            print("✅ Fusione LoRA riuscita")
        except Exception as e2:
            print(f"⚠️ Fusione LoRA non riuscita, fallback a modalità adapter dinamico: {e2}")
           
    # 6. Offload su CPU opzionale
    pipe.enable_model_cpu_offload()

    # 7. Prompt check
    if args.prompt.strip().lower() == "prompt vuoto":
        print("❌ Nessun prompt fornito. Uscita.")
        return

    # 8. Generazione immagine
    print("🖼️ Generazione immagine in corso...")
    try:
        generator = torch.Generator().manual_seed(args.seed)

        pipe_kwargs = {
            "prompt": args.prompt,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "generator": generator,
        }

        if args.use_ip_adapter and image_reference is not None:
            pipe_kwargs["ip_adapter_image"] = image_reference
            pipe_kwargs["true_cfg_scale"] = 4.0  # solo se richiesto per IP Adapter

        result = pipe(**pipe_kwargs)

        image = result[0] if isinstance(result, list) else result.images[0]
        image.save(args.output)
        print(f"✅ Immagine salvata in: {args.output}")

    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generazione immagine con FLUX")
    parser.add_argument("--prompt", type=str, default="prompt vuoto", help="Prompt da generare")
    parser.add_argument("--steps", type=int, default=20, help="Numero di passi di inferenza")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="CFG scale")
    parser.add_argument("--height", type=int, default=1024, help="Altezza immagine")
    parser.add_argument("--width", type=int, default=1024, help="Larghezza immagine")
    parser.add_argument("--output", type=str, default="image_flux.png", help="Percorso salvataggio immagine")
    parser.add_argument("--lora", type=str, default="No_lora", help="Nome del file LoRA (cartella ./Lora)")
    parser.add_argument("--scale", type=float, default=0.7, help="Scala del LoRA (0.0 - 2.0)")
    parser.add_argument("--use_ip_adapter", action="store_true", help="Attiva IP Adapter se presente")
    parser.add_argument("--scale_ip_adapter", type=float, default=0.7, help="Scala IP Adapter (0.0 - 2.0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la generazione")
    parser.add_argument("--ip_adapter_image", type=str, default=None, help="Percorso immagine per IP Adapter")
    args = parser.parse_args()

    main(args)
