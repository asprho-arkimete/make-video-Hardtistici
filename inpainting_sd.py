from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
import torch
import argparse
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a beauty girl")
    parser.add_argument("--ip_use", action="store_true", help="Usa IP-Adapter se attivo")
    parser.add_argument("--sorce_image", type=str, default='image_flux.png', help="Immagine sorgente")
    parser.add_argument("--reference", type=str, default='image_riferimento.png', help="Immagine di riferimento IP-Adapter")
    parser.add_argument("--risoluzione", type=str, default="1024", help="Risoluzione massima (es: 1024 o 1024,768)")
    parser.add_argument("--steps", type=int, default=30, help="Numero di passi di inferenza")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--modifica", type=float, default=0.5, help="Forza modifica IP-Adapter")
    parser.add_argument("--scale_ipadapter", type=float, default=0.5, help="Scala per IP-Adapter (0-1)")


    args = parser.parse_args()

    # Caricamento immagini
    init_image = Image.open(args.sorce_image).convert("RGB")
    mask_image = Image.open('mask.png').convert("L")

    image_reference = None
    if args.ip_use:
        try:
            image_reference = Image.open(args.reference).convert("RGB")
        except Exception as e:
            print(f"⚠️ Immagine di riferimento non trovata: {e}")
            args.ip_use = False

    generator = torch.Generator(device="cpu").manual_seed(1)

    # OpenPose
    checkpoint = "lllyasviel/control_v11p_sd15_openpose"
    processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    control_image = processor(init_image, hand_and_face=True)
    control_image = control_image.resize(init_image.size, Image.BICUBIC)
    control_image.save("posa_catturata.png")

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)

    pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
        "./modelli/epicrealism_pureEvolutionV5-inpainting.safetensors",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        safety_checker=None
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if args.ip_use and image_reference:
        try:
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter-full-face_sd15.bin"
            )
            pipe.set_ip_adapter_scale(args.scale_ipadapter)

        except Exception as error:
            print(f"⚠️ Errore caricamento IP-Adapter: {error}")
            args.ip_use = False

    pipe.enable_model_cpu_offload()

    # Gestione risoluzione
    try:
        if ',' in args.risoluzione:
            max_res = max(int(x) for x in args.risoluzione.split(','))
        else:
            max_res = int(args.risoluzione)
    except:
        max_res = 1024

    w, h = init_image.size
    if w >= h:
        wg = max_res
        hg = (max_res * h) // w
    else:
        hg = max_res
        wg = (max_res * w) // h
        
    # Assicurati che width e height siano divisibili per 8
    wg -= wg % 8
    hg -= hg % 8
    
    prompt = args.prompt

    try:
        result = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=generator,
            eta=1.0,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            ip_adapter_image=image_reference if args.ip_use else None,
            strength=args.modifica, 
            width=wg,
            height=hg
        )
        image = result.images[0]
        image.save("Inpainting_sd.png")
        print("✅ Immagine generata: Inpainting_sd.png")

    except Exception as error:
        print(f"❌ Errore durante la generazione: {error}")


if __name__ == "__main__":
    main()
