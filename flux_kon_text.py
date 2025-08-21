import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize
from PIL import Image
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The subject's clothon is completely removed. She has medium breasts, visible nipples ,visible pussy, pussy lips, pubic hair. Keep subject positioning, camera angle, framing and perspective identical. Remove her bikini to reveal her breasts and cunt with hairy pubes.")
    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--risoluzione", type=str, default='1024,1024')
    parser.add_argument("--lora", type=str, default="No_Lora")
    parser.add_argument("--scale_lora", type=float, default=0.7)
    args = parser.parse_args()

    # Setup
    repo = "black-forest-labs/FLUX.1-Kontext-dev"
    dtype = torch.bfloat16

    # Carica la pipeline
    pipe = FluxKontextPipeline.from_pretrained(repo, torch_dtype=dtype)

    # Applica LoRA PRIMA della quantizzazione
    if args.lora != "No_Lora":
        adapter_name = Path(args.lora).stem
        try:
            print(f"🔗 Caricamento LoRA: {args.lora}")
            pipe.load_lora_weights("./Lora", weight_name=args.lora, adapter_name=adapter_name)
            pipe.set_adapters(adapter_name)
            pipe.fuse_lora(lora_scale=args.scale_lora)
            pipe.unload_lora_weights()
            print("✅ LoRA caricato, fuso e rimosso dalla memoria")
        except Exception as e:
            print(f"❌ Errore nel caricamento/fusione LoRA: {e}")
            exit(1)

    # Quantizzazione modelli
    transformer = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dtype)
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    text_encoder_2 = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder_2", torch_dtype=dtype)
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2
    pipe.enable_model_cpu_offload()

     # Caricamento immagine e resize
    input_path = Path(args.input_image)
    output_path = Path("modifica_flux_kon.png")
    input_image = Image.open(input_path)
    w, h = input_image.size

    # Parsing risoluzione (es: "1024,1024" → max_res = 1024)
    max_res = max(map(int, args.risoluzione.split(',')))

    if w >= h:
        wg = max_res
        hg = (max_res * h) // w
    else:
        hg = max_res
        wg = (max_res * w) // h

    # Arrotonda a multipli di 8
    def round_to_multiple(x, base=64):
        return (x // base) * base

    wg = round_to_multiple(wg)
    hg = round_to_multiple(hg)

    input_image = input_image.resize((wg, hg), Image.BICUBIC)

    # Generazione immagine
    image = pipe(
        image=input_image,
        prompt=args.prompt,
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        width=wg,
        height=hg
    ).images[0]

    image.save(output_path)
    print(f"✅ Output salvato: {output_path}")

if __name__ == "__main__":
    main()


"The subject's clothon is completely removed. She has medium breasts, visible nipples ,visible pussy, pussy lips, pubic hair. Keep subject positioning, camera angle, framing and perspective identical. Remove her bikini to reveal her breasts and cunt with hairy pubes."
    

