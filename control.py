from tkinter import *
from tkinter import ttk
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os 
import shutil
import webbrowser
from deep_translator import GoogleTranslator
import backgroundremover 
 

window = Tk()
window.title("Control Panel")
window.geometry("950x350")  # Altezza aumentata per contenere tutti i widget
window.resizable(False, False)

# Dizionario di abbigliamento per soggetto
abbigliamento_per_soggetto = {
    "Donna": ["vestito elegante", "blusa e jeans", "tailleur", "abito da sera"],
    "ragazza": ["top e shorts", "maglietta e jeans", "felpa e leggings", "gonna e maglietta"],
    "petite": ["abito corto", "gonna a pieghe", "crop top", "salopette"],
    "teen": ["maglietta oversize", "jeans strappati", "hoodie e sneakers", "pantaloncini sportivi"],
    "Signora Anziana": ["golfino e gonna lunga", "vestito floreale", "blusa e pantaloni comodi", "tailleur sobrio"],
    "ragazzo": ["t-shirt e jeans", "felpa e pantaloni sportivi", "camicia e pantaloni", "maglietta e shorts"],
    "Uomo": ["camicia e jeans", "giacca e pantaloni", "maglietta e chinos", "abito elegante"],
    "Uomo Anziano": ["pullover e pantaloni", "camicia e cardigan", "giacca in tweed", "gilet e pantaloni classici"]
}

# Funzione per aggiornare l'abbigliamento in base al soggetto selezionato
def aggiorna_abbigliamento(event, soggetto_combo, abbigliamento_combo):
    soggetto = soggetto_combo.get()
    abbigliamento_combo['values'] = abbigliamento_per_soggetto.get(soggetto, [])

# Categoria
categoria = ttk.Combobox(window, values=["Trio", "Lesbiche", "Etero"])
categoria.grid(row=0, column=0, padx=5, pady=10)
categoria.set("Categoria")

costum_categoria = Text(window, height=1, width=15)
costum_categoria.grid(row=1, column=0, padx=5)
costum_categoria.insert(END, "Categoria pers.")

# Location
location = ttk.Combobox(window, values=["Camera da letto", "Divano soggiorno", "Doccia", "Cucina", "Salotto", "Spiaggia"])
location.grid(row=0, column=1, padx=5)
location.set("Luogo")

costum_location = Text(window, height=1, width=15)
costum_location.grid(row=1, column=1, padx=5)
costum_location.insert(END, "Luogo pers.")

# Soggetto 1
soggetto1 = ttk.Combobox(window, values=list(abbigliamento_per_soggetto.keys()))
soggetto1.grid(row=0, column=2, padx=5)
soggetto1.set("Soggetto 1")

soggetto1.set('Donna')

costum_soggetto1 = Text(window, height=1, width=15)
costum_soggetto1.grid(row=1, column=2, padx=5)
costum_soggetto1.insert(END, "Soggetto 1 pers.")

capelli1 = ttk.Combobox(window, values=['capelli biondi', 'capelli neri', 'capelli rossi', 'capelli castani', 'capelli azzurri'])
capelli1.grid(row=2, column=2, padx=5)
capelli1.set("Capelli 1")

capelli1.set('capelli neri')

costum_capelli1 = Text(window, height=1, width=15)
costum_capelli1.grid(row=3, column=2, padx=5)
costum_capelli1.insert(END, "Capelli 1 pers.")

posa1 = ttk.Combobox(window, values=['in piedi', 'seduta', 'sdraiata'])
posa1.grid(row=2, column=3, padx=5)
posa1.set("Posa 1")

posa1.set('in piedi')

costum_posa1 = Text(window, height=1, width=15)
costum_posa1.grid(row=3, column=3, padx=5)
costum_posa1.insert(END, "Posa 1 pers.")

Seno1= ttk.Combobox(window,values=['seno piccolo','seno medio','seno grande'])
Seno1.grid(row=4,column=3)
Seno1.set("Seno 1")

Seno1.set('seno grande')

costum_Seno1 = Text(window, height=1, width=15)
costum_Seno1.grid(row=5, column=3, padx=5)
costum_Seno1.insert(END, "Seno 1 pers.")

Pube1= ttk.Combobox(window,values=['Pube rasato','Pube peloso','Pube con striscia di peli'])
Pube1.grid(row=6,column=3)
Pube1.set("Pube 1")

Pube1.set('Pube peloso')

costum_Pube1 = Text(window, height=1, width=15)
costum_Pube1.grid(row=7, column=3, padx=5)
costum_Pube1.insert(END, "Pube 1 pers.")


typecapelli1 = ttk.Combobox(window, values=['mossi', 'lisci', 'ricci', 'ondulati'])
typecapelli1.grid(row=4, column=2, padx=5)
typecapelli1.set("Tipo Capelli 1")

typecapelli1.set('ondulati')

costum_type1 = Text(window, height=1, width=15)
costum_type1.grid(row=5, column=2, padx=5)
costum_type1.insert(END, "Type 1 pers.")

abbigliamento1 = ttk.Combobox(window)
abbigliamento1.grid(row=0, column=3, padx=5)
abbigliamento1.set("Abbigliamento 1")

abbigliamento1.set('blusa e jeans')

costum_abbigliamento1 = Text(window, height=1, width=15)
costum_abbigliamento1.grid(row=1, column=3, padx=5)
costum_abbigliamento1.insert(END, "Abbigliamento 1 pers.")

soggetto1.bind("<<ComboboxSelected>>", lambda e: aggiorna_abbigliamento(e, soggetto1, abbigliamento1))

# Soggetto 2
soggetto2 = ttk.Combobox(window, values=list(abbigliamento_per_soggetto.keys()))
soggetto2.grid(row=0, column=4, padx=5)
soggetto2.set("Soggetto 2")

soggetto2.set('ragazza')

costum_soggetto2 = Text(window, height=1, width=15)
costum_soggetto2.grid(row=1, column=4, padx=5)
costum_soggetto2.insert(END, "Soggetto 2 pers.")

capelli2 = ttk.Combobox(window, values=['capelli biondi', 'capelli neri', 'capelli rossi', 'capelli castani', 'capelli azzurri'])
capelli2.grid(row=2, column=4, padx=5)
capelli2.set("Capelli 2")

capelli2.set('capelli biondi')

costum_capelli2 = Text(window, height=1, width=15)
costum_capelli2.grid(row=3, column=4, padx=5)
costum_capelli2.insert(END, "Capelli 2 pers.")

posa2 = ttk.Combobox(window, values=['in piedi', 'seduta', 'sdraiata'])
posa2.grid(row=2, column=5, padx=5)
posa2.set("Posa 2")

posa2.set('in piedi')

costum_posa2 = Text(window, height=1, width=15)
costum_posa2.grid(row=3, column=5, padx=5)
costum_posa2.insert(END, "Posa 2 pers.")

Seno2= ttk.Combobox(window,values=['seno piccolo','seno medio','seno grande'])
Seno2.grid(row=4,column=5)
Seno2.set("Seno 2")

Seno2.set('seno grande')

costum_Seno2 = Text(window, height=1, width=15)
costum_Seno2.grid(row=5, column=5, padx=5)
costum_Seno2.insert(END, "Seno 2 pers.")

Pube2= ttk.Combobox(window,values=['Pube rasato','Pube peloso','Pube con striscia di peli'])
Pube2.grid(row=6,column=5)
Pube2.set("Pube 1")

Pube2.set('Pube peloso')

costum_Pube2 = Text(window, height=1, width=15)
costum_Pube2.grid(row=7, column=5, padx=5)
costum_Pube2.insert(END, "Pube 2 pers.")

typecapelli2 = ttk.Combobox(window, values=['mossi', 'lisci', 'ricci', 'ondulati'])
typecapelli2.grid(row=4, column=4, padx=5)
typecapelli2.set("Tipo Capelli 2")

typecapelli2.set('ondulati')

costum_type2 = Text(window, height=1, width=15)
costum_type2.grid(row=5, column=4, padx=5)
costum_type2.insert(END, "Type 2 pers.")

abbigliamento2 = ttk.Combobox(window)
abbigliamento2.grid(row=0, column=5, padx=5)
abbigliamento2.set("Abbigliamento 2")

abbigliamento2.set('maglietta e jeans')

costum_abbigliamento2 = Text(window, height=1, width=15)
costum_abbigliamento2.grid(row=1, column=5, padx=5)
costum_abbigliamento2.insert(END, "Abbigliamento 2 pers.")

soggetto2.bind("<<ComboboxSelected>>", lambda e: aggiorna_abbigliamento(e, soggetto2, abbigliamento2))

# Steps + Checkbox Ideogram
Steps_making_var = StringVar(value='Step0')
Steps_making = ttk.Combobox(window, values=['Step0', 'Step1', 'Step2','Step3','Step4','Step5','Step6'], textvariable=Steps_making_var)
Steps_making.grid(row=6, column=1)

id_var = BooleanVar(value=False)
Ideogram = ttk.Checkbutton(window, text='Use Ideogram', variable=id_var)
Ideogram.grid(row=6, column=2)

import os
import torch
import random
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler
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

def stablediffuser15(
    path_model="hyperRealism_30.safetensors",
    image_pose='nessuna',           # percorso immagine per ControlNet OpenPose
    prompt="una ragazza in piedi",
    negative="low quality",
    image_IP='image_riferimento.png',  # immagine per IP-Adapter
    steps=30,
    cfg=7.5
):
    import random
    import torch
    import os
    from PIL import Image
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
    )
    from controlnet_aux import OpenposeDetector

    checkpoint = "lllyasviel/control_v11p_sd15_openpose"
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2", subfolder="scheduler"
    )

    pipe = None
    control_image = None
    image_pose_pil = None

    # CONTROLNET: se c'è immagine pose valida
    print(f"immagine di posa: {image_pose}")
    if image_pose != 'nessuna' and os.path.exists(image_pose):
        try:
            image_pose_pil = Image.open(image_pose).convert("RGB")
            processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cuda")
            control_image = processor(image_pose_pil, hand_and_face=True)
            control_image.save("image posa.png")

            if isinstance(control_image, torch.Tensor):
                control_image = control_image.squeeze().cpu().numpy()
                control_image = Image.fromarray((control_image * 255).astype("uint8"))

            controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16).to("cuda")

            pipe = StableDiffusionControlNetPipeline.from_single_file(
                f"modelli/{path_model}",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                scheduler=noise_scheduler,
            )
            print(f"uso immagine control {image_pose}")
        except Exception as e:
            print(f"[ERROR] Errore nel processing con OpenposeDetector: {e}")
            control_image = None
            pipe = None

    # PIPELINE base se ControlNet non attivo o fallito
    if pipe is None:
        pipe = StableDiffusionPipeline.from_single_file(
            f"modelli/{path_model}",
            torch_dtype=torch.float16,
            safety_checker=None,
            scheduler=noise_scheduler,
        )

    # IP-Adapter
    ip_image = None
    print(f"IP_ image{image_IP}")
    if image_IP and os.path.isfile(image_IP):
        print("[INFO] Carico IP-Adapter SD1.5")
        ip_image = Image.open(image_IP).convert("RGB")
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-full-face_sd15.bin"
        )
        pipe.set_ip_adapter_scale(0.6)
        print(f"uso IP: {image_IP}")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    generator = torch.Generator("cuda").manual_seed(random.randint(1, 100000))

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        guidance_scale=cfg,
        width=1024,
        height=1024,
        num_inference_steps=steps,
        generator=generator,
    )

    if ip_image is not None:
        kwargs["ip_adapter_image"] = ip_image

    if isinstance(pipe, StableDiffusionControlNetPipeline) and control_image is not None:
        kwargs["control_image"] = control_image
        kwargs["image"] = image_pose_pil  # immagine base per ControlNet
    
    try:
        result = pipe(**kwargs).images[0]
        return result
    except Exception as e:
        print(f"[ERROR] Errore nella generazione dell'immagine: {e}")
    return None



import cv2
import torch
from diffusers import FluxInpaintPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize

os.makedirs("frames", exist_ok=True)

def SD15ControlInpainting(f, source_path, output_path):
    global sog,cap_col,cap_type,posa,Seno,pube

    import os
    import time
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from PIL import Image

    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
    from controlnet_aux import OpenposeDetector
    from deep_translator import GoogleTranslator  # se usi questa libreria

    # === Step 1: Segmentazione e creazione maschera ===
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    image = Image.open(source_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    CLOTH_LABELS = {
        1: 'hat', 2: 'hair', 3: 'glove', 4: 'sunglasses',
        5: 'upper_clothes', 6: 'dress', 7: 'coat', 8: 'socks',
        9: 'pants', 10: 'jumpsuits', 11: 'scarf', 12: 'skirt'
    }
    cloth_ids = [5, 6, 7, 9, 10, 12]

    # Creazione maschera binaria
    if os.path.exists("mask.png"):
        os.remove("mask.png")
    mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for cid in cloth_ids:
        mask |= (pred_seg == cid)

    mask_np = mask.cpu().numpy().astype(np.uint8) * 255

    # DILATAZIONE della maschera
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)

    mask_img = Image.fromarray(mask_dilated, mode="L")
    mask_img.save("mask.png")
    print("🧤 Maschera estesa salvata come mask.png")
    time.sleep(2)

    # === Step 2: Generazione posa con ControlNet ===
    init_image = image
    mask_image = mask_img

    processor_pose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    control_image = processor_pose(init_image, hand_and_face=True)
    control_image = control_image.resize(init_image.size, Image.BICUBIC)
    control_image.save("posa_catturata.png")

    # === Step 3: Caricamento pipeline SD 1.5 Inpainting con ControlNet ===
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.bfloat16
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
        "./modelli/epicrealism_pureEvolutionV5-inpainting.safetensors",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        safety_checker=None
    )

    # Caricamento LoRA
    lora_path = "./Lora/RealPussyAIOv1.safetensors"
    if os.path.exists(lora_path):
        try:
            adapter_name = os.path.splitext(os.path.basename(lora_path))[0].replace(" ", "_")
            pipe.load_lora_weights(
                "./Lora",
                weight_name=os.path.basename(lora_path),
                adapter_name=adapter_name
            )
            pipe.set_adapters(adapter_name, adapter_weights=0.8)
            pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=0.8)
            print(f"🧩 LoRA '{adapter_name}' caricata e fusa con successo.")
        except Exception as e:
            print(f"❌ Errore nel caricamento/fusione LoRA: {e}")
    else:
        print(f"❌ File LoRA '{lora_path}' non trovato.")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # === Step 5: Calcolo risoluzione finale ===
    w, h = init_image.size
    if w >= h:
        hg = (1024 * h) // w
        wg = 1024
    else:
        wg = (1024 * w) // h
        hg = 1024
    wg -= wg % 8
    hg -= hg % 8

    # === Step 6: Inpainting finale ===
    print(f"f: {f}")

    # ATTENZIONE: le variabili sog, cap_col, cap_type, posa, seno, pube
    # devono essere definite e passate alla funzione o accessibili globalmente
    prompt = None
    if f == 'intera':
        prompt = f"""sfondo green screen con davanti una {sog},{cap_col},{cap_type},{posa}, gambe distanziate,
                    (totalmente nuda:1.9),({Seno}:1.9),(Seno Visibile:1.9),(capezzoli visibili:1.9), (figa con {pube}:1.9),(labia of pussy:1.7),braccia lungo i fianchi,"""
    elif f == 'meta':
        prompt = f"""sfondo green screen con davanti una {sog},{cap_col},{cap_type},{posa}, gambe distanziate,
                    (totalmente nuda:1.9),({Seno}:1.9),(Seno Visibile:1.9),(capezzoli visibili:1.9),braccia lungo i fianchi,"""
    else:
        prompt = None

    eng_prompt = ''
    if prompt is not None:
        eng_prompt = GoogleTranslator(source='it', target='en').translate(prompt)
    print(f"prompt engl: step 2: {eng_prompt}")

    import random
    numero_casuale = random.randint(1, 100000)

    try:
        result = pipe(
            prompt=eng_prompt,
            negative_prompt="low quality, low resolution, bad anatomy, bad hands, bad breast, bad pussy",
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=0.95,
            width=wg,
            height=hg,
            generator=torch.Generator(device="cpu").manual_seed(numero_casuale)
        )
        result.images[0].save(output_path)
        print(f"✅ Inpainting completato: {output_path}")
    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")
    
 # Stato flip
flipped1 = False
flipped2 = False
def make_video():
    global soggetto1, soggetto2, location, capelli1, capelli2, typecapelli1, typecapelli2
    global abbigliamento1, abbigliamento2, posa1, posa2, Seno1, Seno2, Pube1, Pube2
    global costum_soggetto1, costum_soggetto2, costum_abbigliamento1, costum_abbigliamento2
    global costum_Seno1, costum_Seno2, costum_Pube1, costum_Pube2
    global costum_location, costum_capelli1, costum_capelli2, costum_type1, costum_type2
    global costum_posa1, costum_posa2
    global Steps_making
    global sog,cap_col,cap_type,posa,Seno,pube
    global location,costum_location
    global flipped1,flipped2


    only_face = False
    ideogram=False
    if id_var.get()==True:
        ideogram=True
    else:
        ideogram=False
    if Steps_making.get() == 'Step0':
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            id2label = {
                0: 'background', 1: 'hat', 2: 'hair', 3: 'glove', 4: 'sunglasses',
                5: 'upper_clothes', 6: 'dress', 7: 'coat', 8: 'socks', 9: 'pants',
                10: 'jumpsuits', 11: 'scarf', 12: 'skirt', 13: 'face', 14: 'left_arm',
                15: 'right_arm', 16: 'left_leg', 17: 'right_leg', 18: 'left_shoe', 19: 'right_shoe'
            }

            processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

            # === FUNZIONE INTERNA PER PROCESSARE I SOGGETTI ===
            def process_subject(image_path, output_path, subject_idx):
                if not os.path.exists(image_path):
                    print(f"❌ Immagine riferimento {subject_idx} non esiste.")
                    return

                # 1. Rilevamento volto
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) == 0:
                    print(f"❌ Nessun volto rilevato per soggetto {subject_idx}")
                    return
                print(f"✅ Volto rilevato per soggetto {subject_idx}")

                # 2. Segmentazione abbigliamento
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits.cpu()
                upsampled_logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
                pred_seg = upsampled_logits.argmax(dim=1)[0]
                class_ids = torch.unique(pred_seg).tolist()
                labels_presenti = [id2label.get(cid, f"id_{cid}") for cid in class_ids]
                print(f"🎨 Classi rilevate (soggetto {subject_idx}):", labels_presenti)

                # 3. Salvataggio o generazione
                if 'pants' in labels_presenti or 'skirt' in labels_presenti:
                    print(f"🧍 Figura intera soggetto {subject_idx}")
                    shutil.copyfile(image_path, output_path)
                    only_face= False
                elif 'upper_clothes' in labels_presenti:
                    print(f"🙆 Mezza figura soggetto {subject_idx}")
                    shutil.copyfile(image_path, output_path)
                    only_face=False
                else:
                    only_face=True
                    print(f"🫣 Primo piano soggetto {subject_idx}")
                    if ideogram:
                        webbrowser.open("https://ideogram.ai/character")
                        input("Premi Invio per continuare...")
                        print("➡️ Esc Step 0")
                    else:
                        print("✨ Generazione con SD + IP Adapter (soggetto {})".format(subject_idx))
                        # Prompt dinamico soggetto 1 o 2
                        if subject_idx == 1:
                            sog = costum_soggetto1.get("1.0", "end").strip() if costum_soggetto1.get("1.0", "end").strip() != "Soggetto 1 pers." else soggetto1.get()
                            cap_col = costum_capelli1.get("1.0", "end").strip() if costum_capelli1.get("1.0", "end").strip() != "Capelli 1 pers." else capelli1.get()
                            cap_type = costum_type1.get("1.0", "end").strip() if costum_type1.get("1.0", "end").strip() != "Type 1 pers." else typecapelli1.get()
                            abb = costum_abbigliamento1.get("1.0", "end").strip() if costum_abbigliamento1.get("1.0", "end").strip() != "Abbigliamento 1 pers." else abbigliamento1.get()
                            posa = costum_posa1.get("1.0", "end").strip() if costum_posa1.get("1.0", "end").strip() != "Posa 1 pers." else posa1.get()
                        else:
                            sog = costum_soggetto2.get("1.0", "end").strip() if costum_soggetto2.get("1.0", "end").strip() != "Soggetto 2 pers." else soggetto2.get()
                            cap_col = costum_capelli2.get("1.0", "end").strip() if costum_capelli2.get("1.0", "end").strip() != "Capelli 2 pers." else capelli2.get()
                            cap_type = costum_type2.get("1.0", "end").strip() if costum_type2.get("1.0", "end").strip() != "Type 2 pers." else typecapelli2.get()
                            abb = costum_abbigliamento2.get("1.0", "end").strip() if costum_abbigliamento2.get("1.0", "end").strip() != "Abbigliamento 2 pers." else abbigliamento2.get()
                            posa = costum_posa2.get("1.0", "end").strip() if costum_posa2.get("1.0", "end").strip() != "Posa 2 pers." else posa2.get()
                        prompt = f"sfondo blue screen, una {sog} con {cap_col}, {cap_type}, che indossa {abb}, di colore differente dal blue, in posa {posa}, braccia lungo il corpo"
                        prompt_eng= GoogleTranslator(source='it',target='en').translate(prompt)
                        print(f"Prompt Engl: {prompt_eng}")
                        negative = "low resolution, blurry, bad anatomy"

                        image = stablediffuser15(
                            path_model="hyperRealism_30.safetensors",
                            image_pose=None,
                            prompt=prompt_eng,
                            negative=negative,
                            image_IP=image_path,
                            steps=100,
                            cfg=7.5,
                        )
                        image.save(output_path)

            # Esegui per entrambi i soggetti
            process_subject("image_riferimento.png", "frames/frame0sog1.png", subject_idx=1)
            process_subject("immagine_riferimento_2.png", "frames/frame0sog2.png", subject_idx=2)

        except Exception as e:
            print("❌ Errore durante la segmentazione:", str(e))

    if Steps_making.get() in ['Step0', 'Step1']:
        import os
        import warnings
        from PIL import Image
        from deep_translator import GoogleTranslator

        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

        # Process subjects 1 and 2
        for i in range(1, 3):  # 1 e 2 inclusi
            print(f"\n=== Processing subject {i} ===")

            # Assegna immagini di riferimento e output
            if i == 1:
                image_path = "image_riferimento.png"
                output_path = "./frames/frame1sog1.png"
            elif i == 2:
                image_path = "immagine_riferimento_2.png"
                output_path = "./frames/frame1sog2.png"

            print(f"Input image: {image_path}")
            print(f"Output path: {output_path}")

            if not os.path.exists(image_path):
                print(f"❌ Immagine riferimento {i} non trovata: {image_path}")
                continue

            # Prepara prompt dinamico per il soggetto
            if i == 1:
                sog = costum_soggetto1.get("1.0", "end").strip()
                sog = sog if sog != "Soggetto 1 pers." else soggetto1.get()

                cap_col = costum_capelli1.get("1.0", "end").strip()
                cap_col = cap_col if cap_col != "Capelli 1 pers." else capelli1.get()

                cap_type = costum_type1.get("1.0", "end").strip()
                cap_type = cap_type if cap_type != "Type 1 pers." else typecapelli1.get()

                Seno = costum_Seno1.get("1.0", "end").strip()
                Seno = Seno if Seno != "Seno 1 pers." else Seno1.get()

                posa = costum_posa1.get("1.0", "end").strip()
                posa = posa if posa != "Posa 1 pers." else posa1.get()

                pube = costum_Pube1.get("1.0", "end").strip()
                pube = pube if pube != "Pube 1 pers." else Pube1.get()

            elif i == 2:
                sog = costum_soggetto2.get("1.0", "end").strip()
                sog = sog if sog != "Soggetto 2 pers." else soggetto2.get()

                cap_col = costum_capelli2.get("1.0", "end").strip()
                cap_col = cap_col if cap_col != "Capelli 2 pers." else capelli2.get()

                cap_type = costum_type2.get("1.0", "end").strip()
                cap_type = cap_type if cap_type != "Type 2 pers." else typecapelli2.get()

                Seno = costum_Seno2.get("1.0", "end").strip()
                Seno = Seno if Seno != "Seno 2 pers." else Seno2.get()

                posa = costum_posa2.get("1.0", "end").strip()
                posa = posa if posa != "Posa 2 pers." else posa2.get()

                pube = costum_Pube2.get("1.0", "end").strip()
                pube = pube if pube != "Pube 2 pers." else Pube2.get()

            prompt_it = (
                f"sfondo green screen, una {sog} con {cap_col}, {cap_type}, "
                f"(totalmente nuda:1.9),{Seno},(capezzoli visivili:1.8),({pube}:1.9) "
                f"in posa {posa},braccia lungo il corpo"
            )
            prompt_eng = GoogleTranslator(source="it", target="en").translate(prompt_it)
            negative = "low resolution, blurry, bad anatomy"

            figura = 'intera'  # default figura

            def determina_figura(img_path):
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                import torch



                image = Image.open(img_path).convert("RGB")
                processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
                model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
                id2label = model.config.id2label

                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits.cpu()

                upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=image.size[::-1], mode="bilinear", align_corners=False
                )
                pred_seg = upsampled_logits.argmax(dim=1)[0]
                class_ids = torch.unique(pred_seg).tolist()

                labels_presenti = [id2label.get(cid, f"id_{cid}") for cid in class_ids]
                labels_normalizzate = [label.lower().replace(" ", "_") for label in labels_presenti]

                META_BUSTO = {"upper_clothes", "coat"}
                FIGURA_INTERA = {"pants", "skirt", "dress", "jumpsuits"}

                if any(label in FIGURA_INTERA for label in labels_normalizzate):
                    print(f"funzione : imagine analizata {img_path}: intera")
                    return 'intera'
                    
                elif any(label in META_BUSTO for label in labels_normalizzate):
                    print(f"funzione : fimagine analizata {img_path}: meta busto")
                    return 'meta'
                else:
                    print(f"funzione : imagine analizata {img_path}: face")
                    return 'face'

            try:
                figura = determina_figura(image_path)
                print(f"🎨 Classi rilevate (soggetto {i}), figura determinata: {figura}")
            except Exception as e:
                print(f"⚠️ Errore controllo figura soggetto {i}: {e}")
                figura = 'face'

            figura_ideogram = 'nessuna'  # inizializzo con valore di default

            if ideogram:
                import webbrowser
                webbrowser.open("https://ideogram.ai/character")

                input(f"""Salva immagine Ideogram come 'v{i}.jpeg' o 'V{i}.jpeg' nella cartella 'frames'.\nPremi Invio per continuare per soggetto {i}...Prompt da utilizzare in Ideogram: 'sfondo green screen con davanti una ragazza {cap_col}, capelli media lunghezza, capelli {cap_type}, {posa}, gambe distanziate, indossa un vestitino corto, scollatura al seno, il vestitino è di colore rosso, braccia lungo i fianchi, vestitino bene visibile non coperto dalle braccia, {Seno}, i capelli dietro le spalle che non coprano in nessun modo il vestito e il seno""")
                try:
                    v1 = v2 = ''
                    if os.path.exists("frames/v1.jpeg"):
                        v1 = "frames/v1.jpeg"
                    elif os.path.exists("frames/V1.jpeg"):
                        v1 = "frames/V1.jpeg"

                    if os.path.exists("frames/v2.jpeg"):
                        v2 = "frames/v2.jpeg"
                    elif os.path.exists("frames/V2.jpeg"):
                        v2 = "frames/V2.jpeg"

                    if i == 1:
                        print(f"file v1:{v1}")
                        figura_ideogram = determina_figura(v1)
                    elif i == 2:
                        print(f"file v1:{v1}")
                        figura_ideogram = determina_figura(v2)

                    print(f"🎨 Figura da immagine Ideogram soggetto {i}: {figura_ideogram}")
                except Exception as e:
                    print(f"⚠️ Errore controllo figura immagine Ideogram soggetto {i}: {e}")
                    figura_ideogram = 'face'  # fallback sicuro
                if i == 1 and figura_ideogram in ['meta', 'intera']:
                    print(f"🖌️ Eseguo SD15ControlInpainting su {v1}")
                    SD15ControlInpainting(figura_ideogram, v1, output_path)
                elif i == 2 and figura_ideogram in ['meta', 'intera']:
                    print(f"🖌️ Eseguo SD15ControlInpainting su {v2}")
                    SD15ControlInpainting(figura_ideogram, v2, output_path)

            elif figura == 'face' and not ideogram:
                # Solo faccia senza ideogram: usa stable diffuser subito
                try:
                    print(f"🖌️ Eseguo stable diffuser su immagine facciale {image_path} senza ideogram")
                    image = stablediffuser15(
                        path_model="hyperRealism_30.safetensors",
                        image_pose=None,
                        prompt=prompt_eng,
                        negative=negative,
                        image_IP=image_path,
                        steps=100,
                        cfg=7.5
                    )
                    image.save(output_path)
                    print(f"✅ Immagine salvata (stable diffuser): {output_path}")
                except Exception as e:
                    print(f"❌ Errore generazione immagine soggetto {i} stable diffuser: {e}")

            else:
                # figura meta o intera dall'inizio, usa inpainting direttamente
                try:
                    print(f"🖌️ Eseguo SD15ControlInpainting su {image_path} (figura: {figura})")
                    SD15ControlInpainting(figura, image_path, output_path)
                    print(f"✅ Immagine salvata (inpainting): {output_path}")
                except Exception as e:
                    print(f"❌ Errore generazione immagine soggetto {i} inpainting: {e}")
                
    if Steps_making.get() in ['Step0', 'Step1', 'Step2']:
        for subject_idx in range(1, 3):
            if subject_idx == 1:
                sog = costum_soggetto1.get("1.0", "end").strip()
                sog = sog if sog != "Soggetto 1 pers." else soggetto1.get()

                cap_col = costum_capelli1.get("1.0", "end").strip()
                cap_col = cap_col if cap_col != "Capelli 1 pers." else capelli1.get()

                cap_type = costum_type1.get("1.0", "end").strip()
                cap_type = cap_type if cap_type != "Type 1 pers." else typecapelli1.get()

                abb = costum_abbigliamento1.get("1.0", "end").strip()
                abb = abb if abb != "Abbigliamento 1 pers." else abbigliamento1.get()

                posa = costum_posa1.get("1.0", "end").strip()
                posa = posa if posa != "Posa 1 pers." else posa1.get()

            else:
                sog = costum_soggetto2.get("1.0", "end").strip()
                sog = sog if sog != "Soggetto 2 pers." else soggetto2.get()

                cap_col = costum_capelli2.get("1.0", "end").strip()
                cap_col = cap_col if cap_col != "Capelli 2 pers." else capelli2.get()

                cap_type = costum_type2.get("1.0", "end").strip()
                cap_type = cap_type if cap_type != "Type 2 pers." else typecapelli2.get()

                abb = costum_abbigliamento2.get("1.0", "end").strip()
                abb = abb if abb != "Abbigliamento 2 pers." else abbigliamento2.get()

                posa = costum_posa2.get("1.0", "end").strip()
                posa = posa if posa != "Posa 2 pers." else posa2.get()

            if ideogram:
                import webbrowser
                webbrowser.open("https://ideogram.ai/character")
                input(
                    f"""Prompt da utilizzare in Ideogram: '(Primo piano della testa) di una {sog} in piedi, totalmente con il 
    corpo di (profilo sinistro) con {cap_col} e {cap_type} di media lunghezza, la bocca spalancata in modo estremo e la lingua fuori, su sfondo verde uniforme, 
    nello stile di una fotografia segnaletica della polizia.'\nSalva immagine Ideogram come 'frame2sog{subject_idx}.jpeg' nella cartella 'frames', se il profilo è sbagliato riflettilo con Photoshop.
    \nPremi Invio per continuare per soggetto {subject_idx}..."""
                )
            else:
                import os
                from deep_translator import GoogleTranslator

                output_path = os.path.join("frames", f"frame2sog{subject_idx}.jpg")

                prompt_it = f"""(Primo piano della testa:1.9) di una {sog} in piedi, totalmente con il 
    corpo di (profilo sinistro:1.9) con {cap_col} e {cap_type} di media lunghezza, la bocca spalancata in modo estremo e la lingua fuori"""

                prompt_eng = GoogleTranslator(source='it', target='en').translate(prompt_it)
                print(f"prompt: {prompt_eng}")

                negative = """low quality, low resolution, blurry, artifacts, 
    distorted proportions, frontal view, one-third view, 3/4 view"""

                def img2imag(imagepath, ip_adapter_path, ip_adapter_scale,mod,l,prompt, negative_prompt):
                    print("img 2 img")
                    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
                    from controlnet_aux import OpenposeDetector
                    from PIL import Image
                    import torch
                    import random

                    seed = random.randint(1, 10000)
                    base_model = "./modelli/hyperRealism_30.safetensors"
                    checkpoint = "lllyasviel/control_v11p_sd15_openpose"
                    print(f"[INFO] Generazione con seed: {seed}")
                    print("[INFO] Uso ControlNet SD1.5 con OpenPose")

                    # Carica immagine e genera control image con OpenPose
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

                    # Carica ControlNet e pipeline
                    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16).to("cuda")
                    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                        base_model,
                        controlnet=controlnet,
                        torch_dtype=torch.float16
                    ).to("cuda")

                    ip_image = None
                    if ip_adapter_path:
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
                    if l=='S':
                        pipe.load_lora_weights("./Lora//sd_tongue28.safetensors",adapter_name="sd_tongue28")
                        pipe.set_adapters("sd_tongue28", adapter_weights=0.8)
                        pipe.fuse_lora(adapter_names=["sd_tongue28"],lora_scale=0.8)
                    else:
                        pipe.load_lora_weights("./Lora//sd_Long_Tongue_Lora.safetensors",adapter_name="sd_Long_Tongue_Lora")
                        pipe.set_adapters("sd_Long_Tongue_Lora", adapter_weights=0.8)
                        pipe.fuse_lora(adapter_names=["sd_Long_Tongue_Lora"],lora_scale=0.8)


                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                    pipe.enable_model_cpu_offload()
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image_pose,
                        control_image=control_image,
                        num_inference_steps=100,
                        guidance_scale=7.5,
                        strength=mod,
                        generator=torch.manual_seed(seed),
                        width=1024,
                        height=1024,
                        ip_adapter_image=ip_image if ip_image else None
                    ).images[0]

                    return result

                # Imposta il percorso dell'IP-Adapter corretto
                ip_adapter = "./image_riferimento.png" if subject_idx == 1 else "./immagine_riferimento_2.png"

                # Scegli immagine di partenza in base al tipo di capelli
                if cap_type.lower() == 'lisci':
                    pose_img = os.path.join("pose breastfeeding", "teen.png")
                else:
                    pose_img = os.path.join("pose breastfeeding", "donna_matura.png")

                out = img2imag(pose_img, ip_adapter, 0.6,0.7,'S',prompt_eng, negative)
                out.save(output_path)
        
    if Steps_making.get() in ['Step0', 'Step1', 'Step2', 'Step3']:
        def img2imag2(path_lora, scale, imagepath, mod, prompt, negative_prompt):
            print("img 2 img")
            from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
            from controlnet_aux import OpenposeDetector
            from PIL import Image
            import torch
            import random
            import os

            seed = random.randint(1, 10000)
            base_model = "./modelli/hyperRealism_30.safetensors"
            checkpoint = "lllyasviel/control_v11p_sd15_openpose"
            print(f"[INFO] Generazione con seed: {seed}")
            print("[INFO] Uso ControlNet SD1.5 con OpenPose")

            # Carica immagine posa
            image_pose = Image.open(imagepath).convert("RGB")
            w, h = image_pose.size
            if w >= h:
                hc = (960 * h) // w
                wc = 960
            else:
                wc = (960 * w) // h
                hc = 960
            image_pose = image_pose.resize((wc, hc), Image.BICUBIC)

            # OpenPose
            processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cuda")
            control_image = processor(image_pose, hand_and_face=True)
            if isinstance(control_image, torch.Tensor):
                control_image = control_image.squeeze().cpu().numpy()
                control_image = Image.fromarray((control_image * 255).astype("uint8"))
            control_image.save("control.png")

            # Pipeline
            controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16).to("cuda")
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                base_model,
                controlnet=controlnet,
                torch_dtype=torch.float16
            ).to("cuda")

            if path_lora != "nessuno" and os.path.exists(path_lora):
                pipe.load_lora_weights(path_lora, adapter_name="lora")
                pipe.set_adapters("lora", adapter_weights=scale)
                pipe.fuse_lora(adapter_names=["lora"], lora_scale=scale)

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()

            print(f"immagine posa: {imagepath}")
            print(f"prompt: {prompt}")

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pose,
                control_image=control_image,
                num_inference_steps=40,
                guidance_scale=9.0,
                strength=mod,
                generator=torch.manual_seed(seed),
                width=1024,
                height=1024,
            ).images[0]

            return result

        from deep_translator import GoogleTranslator
        import os

        for subject_idx in range(1, 3):
            # Soggetto
            if subject_idx == 1:
                sog = costum_soggetto1.get("1.0", "end").strip()
                sog = sog if sog != "Soggetto 1 pers." else soggetto1.get()

                cap_col = costum_capelli1.get("1.0", "end").strip()
                cap_col = cap_col if cap_col != "Capelli 1 pers." else capelli1.get()

                cap_type = costum_type1.get("1.0", "end").strip()
                cap_type = cap_type if cap_type != "Type 1 pers." else typecapelli1.get()

                Seno = costum_Seno1.get("1.0", "end").strip()
                Seno = Seno if Seno != "Seno 1 pers." else Seno1.get()

                Pube = costum_Pube1.get("1.0", "end").strip()
                Pube = Pube if Pube != "Pube 1 pers." else Pube1.get()

            elif subject_idx == 2:
                sog = costum_soggetto2.get("1.0", "end").strip()
                sog = sog if sog != "Soggetto 2 pers." else soggetto2.get()

                cap_col = costum_capelli2.get("1.0", "end").strip()
                cap_col = cap_col if cap_col != "Capelli 2 pers." else capelli2.get()

                cap_type = costum_type2.get("1.0", "end").strip()
                cap_type = cap_type if cap_type != "Type 2 pers." else typecapelli2.get()

                Seno = costum_Seno2.get("1.0", "end").strip()
                Seno = Seno if Seno != "Seno 2 pers." else Seno2.get()

                Pube = costum_Pube2.get("1.0", "end").strip()
                Pube = Pube if Pube != "Pube 2 pers." else Pube2.get()

            negative = "low quality, low resolution, blurry, artifacts, distorted proportions, extra nipples, multiple nipples, duplicate breasts,unnatural skin texture, bad anatomy, asymmetrical breasts, extra arms, extra body parts, out of frame, text, watermark"

            mod=0.25
            # frame 3
            prompt = f"fultra realistic photograph, right side profile of a woman with {cap_col}, {cap_type}, medium length hair, standing, with {Seno}, visible nipple, skin pores, natural lighting, high resolution, soft shadows, detailed skin texture"

            prompt_eng = GoogleTranslator(source='it', target='en').translate(prompt)
            imagecontrol = None
            if Seno == "seno grande":
                imagecontrol = "pose breastfeeding//seno_grande.png"
            elif Seno == 'seno medio':
                imagecontrol = "pose breastfeeding//seno_medio.png"
            elif Seno == 'seno piccolo':
                imagecontrol = "pose breastfeeding//seno_piccolo.png"
            out = img2imag2("nessuno", 0.0, imagecontrol, mod, prompt_eng, negative)
            out.save(os.path.join("frames", f"frame3sog{subject_idx}.jpg"))

            # frame 4
            negative = "low quality, low resolution, blurry, artifacts, distorted proportions, extra nipples, multiple nipples, extra ass, extra anu,duplicate breasts,unnatural skin texture, bad anatomy, asymmetrical breasts, extra arms, extra body parts, out of frame, text, watermark"

            prompt = f"primo piano del profilo di una figa aperta, di una {sog} sdraiata a pancia in su con le gambe spalancate,vulva,labbra figa,{Pube},ano"
            prompt_eng = GoogleTranslator(source='it', target='en').translate(prompt)
            if Pube == "Pube rasato":
                imagecontrol = "pose breastfeeding//lick_pussyO1.png"
            else:
                imagecontrol = "pose breastfeeding//lick_pussyO0.png"
            out = img2imag2("./Lora//RealPussyAIOv1.safetensors", 0.8, imagecontrol, mod, prompt_eng, negative)
            out.save(os.path.join("frames", f"frame4sog{subject_idx}.jpg"))

            # frame 5
            prompt = f"primo piano di una figa aperta con {Pube} e dell'ano, di una {sog} sdraiata a pancia in su con le gambe totalmente sollevate al in su,vulva,labbra figa spalancata,{Pube},ano aperto"
            prompt_eng = GoogleTranslator(source='it', target='en').translate(prompt)
            if Pube == "Pube rasato":
                imagecontrol = "pose breastfeeding//lick_analO1.png"
            else:
                imagecontrol = "pose breastfeeding//lick_analO0.png"
            out = img2imag2("./Lora//RealPussyAIOv1.safetensors", 0.8, imagecontrol, mod, prompt_eng, negative)
            out.save(os.path.join("frames", f"frame5sog{subject_idx}.jpg"))

            # frame 6
            prompt = f"primo piano di gambe allunga di una {sog},piede,5 dita del piedi"
            prompt_eng = GoogleTranslator(source='it', target='en').translate(prompt)
            imagecontrol = "pose breastfeeding//feet.png"
            out = img2imag2("nessuno", 0.0, imagecontrol, mod, prompt_eng, negative)
            out.save(os.path.join("frames", f"frame6sog{subject_idx}.jpg"))

    if Steps_making.get() in ['Step0', 'Step1', 'Step2', 'Step3', 'Step4']:
        print("Genera immagine di sfondo con Flux")

        from deep_translator import GoogleTranslator
        import torch, random
        from diffusers import FluxTransformer2DModel, FluxPipeline
        from transformers import T5EncoderModel
        from optimum.quanto import freeze, qfloat8, quantize

        bfl_repo = "black-forest-labs/FLUX.1-dev"
        dtype = torch.bfloat16

        # Carica e quantizza il transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            bfl_repo, subfolder="transformer", torch_dtype=dtype
        )
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        # Carica e quantizza il secondo encoder di testo
        text_encoder_2 = T5EncoderModel.from_pretrained(
            bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype
        )
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        # Carica la pipeline senza il transformer (lo aggiungiamo dopo)
        pipe = FluxPipeline.from_pretrained(
            bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype
        )
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2
        pipe.enable_model_cpu_offload()

        # Scelta location
        if costum_location.get('1.0', 'end').strip() != 'Luogo pers.':
            loc = costum_location.get('1.0', 'end').strip()
        else:
            loc = location.get()

        # Lista colori possibili
        colori_base = [
            "rosso", "blu", "verde", "giallo", "arancione",
            "viola", "bianco", "nero", "turchese", "rosa",
            "beige", "marrone", "grigio"
        ]
        randomcolor = random.choice(colori_base)

        prompt_it = ""

        # Prompt personalizzati e più fantasiosi
        if loc == 'Camera da letto' or 'Luogo':
            prompt_it = f"una camera da letto elegante con un grande letto matrimoniale, lenzuola {randomcolor}, cuscini morbidi, pareti color panna e una luce calda soffusa"
        elif loc == 'Divano soggiorno':
            prompt_it = f"un ampio soggiorno moderno con un divano {randomcolor} in velluto, pareti color avorio e una grande finestra che lascia entrare la luce naturale"
        elif loc == 'Doccia':
            prompt_it = f"interno doccia con mattonelle {randomcolor} lucide, ampia cabina in vetro, cornice in metallo cromato e getto d'acqua rilassante"
        elif loc == 'Cucina':
            prompt_it = f"una cucina moderna spaziosa di colore {randomcolor}, piano di lavoro in marmo e elettrodomestici in acciaio"
        elif loc == 'Salotto':
            prompt_it = f"un salotto accogliente con pareti {randomcolor} pastello, un tappeto soffice e una libreria piena di libri"
        elif loc == 'Spiaggia':
            # Ombrelloni e lettini con colori diversi o uguali
            tipo_spiaggia = random.choice(["spiaggia libera", "stabilimento balneare a pagamento"])
            colore_ombrelloni = random.choice([randomcolor, random.choice(colori_base)])
            colore_lettini = random.choice([randomcolor, random.choice(colori_base)])
            if colore_ombrelloni != colore_lettini:
                descr_colori = f"ombrelloni {colore_ombrelloni} e lettini {colore_lettini}"
            else:
                descr_colori = f"ombrelloni e lettini {colore_ombrelloni}"
            prompt_it = f"una {tipo_spiaggia} con sabbia dorata, {descr_colori}, acqua del mare azzurro cristallino e cielo limpido"

        # Traduzione in inglese
        prompt_eng = GoogleTranslator(source='it', target='en').translate(prompt_it)
        print(f"Prompt ita: {prompt_it}")
        print(f"Prompt en: {prompt_eng}")

        # Genera immagine
        image = pipe(
            prompt_eng,
            guidance_scale=3.5,
            output_type="pil",
            num_inference_steps=30,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        image.save("./frames/location.png")
    

    if Steps_making.get() in ['Step0', 'Step1', 'Step2', 'Step3', 'Step4', 'Step5']:
        from tqdm import tqdm
        bar = tqdm(total=100)
        bar.update(6)  # aggiorni di 1 passo
        import os
        from PIL import Image
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        path_location = "./frames/location.png"
        if os.path.exists(path_location):
            # Carica e croppa l'immagine location
            img = Image.open(path_location)
            offset = 36
            w, h = img.size
            box = (offset, offset, w - offset, h - offset)
            cropped_img = img.crop(box)
            cropped_img.save("./frames/loc_crop.png")
            print("Immagine croppata e salvata correttamente.")
            bar.update(1)  # aggiorni di 1 passo
            # Prepara le immagini con sfondo rimosso
            for k in range(1, 3):
                frame_path = f"./frames/frame0sog{k}.jpeg"
                if not os.path.exists(frame_path):
                    if os.path.exists(f"./frames/v{k}.png"):
                        os.rename(f"./frames/v{k}.png", frame_path)
                    elif os.path.exists(f"./frames/V{k}.png"):
                        os.rename(f"./frames/V{k}.png", frame_path)
                    else:
                        print(f"CREA IMMAGINE INIZIALE frame0sog{k}.jpeg")
                        continue  # passa al prossimo k
            

                if os.path.exists(frame_path):
                    output_path = f"./frames/alfa{k}.png"
                    from backgroundremover.bg import remove
                    def remove_bg(src_img_path, out_img_path):
                        model_choices = ["u2net", "u2net_human_seg", "u2netp"]
                        f = open(src_img_path, "rb")
                        data = f.read()
                        img = remove(data, model_name=model_choices[0],
                                    alpha_matting=True,
                                    alpha_matting_foreground_threshold=240,
                                    alpha_matting_background_threshold=10,
                                    alpha_matting_erode_structure_size=10,
                                    alpha_matting_base_size=1000)
                        f.close()
                        f = open(out_img_path, "wb")
                        f.write(img)
                        f.close()
                    remove_bg(frame_path,output_path)
                bar.update(6)  # aggiorni di 1 passo

            # Carica immagine di sfondo e soggetti
            imaginefinale1 = Image.open("./frames/loc_crop.png").convert('RGBA')
            imaginefinale1= imaginefinale1.resize((1024,1024),Image.BICUBIC)
            ws, hs = imaginefinale1.size

            alfa1 = Image.open("./frames/alfa1.png").convert('RGBA')
            wf1, hf1 = alfa1.size

            alfa2 = Image.open("./frames/alfa2.png").convert('RGBA')
            wf2, hf2 = alfa2.size

            # Margine dal bordo
            margine = 120

            # Posizione primo soggetto (a sinistra)
            pos_alf1 = (margine, (hs // 2) - (hf1 // 2))
            imaginefinale1.paste(alfa1, pos_alf1, alfa1)

            # Posizione secondo soggetto (a destra)
            pos_alf2 = (ws - wf2 - margine, (hs // 2) - (hf2 // 2))
            imaginefinale1.paste(alfa2, pos_alf2, alfa2)

            imaginefinale1.convert('RGB').save("./frames/frame0sog1sog2.jpeg")
            print("Immagine: frame0sog1sog2.jpeg creata con successo")
            bar.update(6)  # aggiorni di 1 passo

            import os
            from PIL import Image
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
            from backgroundremover.bg import remove

            # ---- Scontorno primi due soggetti ----
            for k in range(1, 3):
                frame_path = f"./frames/frame1sog{k}.png"
                if os.path.exists(frame_path):
                    output_path = f"./frames/alfa{(2+k)}.png"
                    def remove_bg(src_img_path, out_img_path):
                        model_choices = ["u2net", "u2net_human_seg", "u2netp"]
                        with open(src_img_path, "rb") as f:
                            data = f.read()
                        img = remove(
                            data,
                            model_name=model_choices[0],
                            alpha_matting=True,
                            alpha_matting_foreground_threshold=240,
                            alpha_matting_background_threshold=10,
                            alpha_matting_erode_structure_size=10,
                            alpha_matting_base_size=1000
                        )
                        with open(out_img_path, "wb") as f:
                            f.write(img)
                    remove_bg(frame_path, output_path)
            bar.update(6)  # aggiorni di 1 passo

            # ---- Collage iniziale con alfa3 e alfa4 ----
            sfondo = Image.open("./frames/loc_crop.png").convert('RGBA').resize((1024, 1024), Image.BICUBIC)
            ws, hs = sfondo.size

            alfa3 = Image.open("./frames/alfa3.png").convert('RGBA')
            alfa4 = Image.open("./frames/alfa4.png").convert('RGBA')
            wf1, hf1 = alfa3.size
            wf2, hf2 = alfa4.size

            margine = 120
            sfondo.paste(alfa3, (margine, (hs // 2) - (hf1 // 2)), alfa3)
            sfondo.paste(alfa4, (ws - wf2 - margine, (hs // 2) - (hf2 // 2)), alfa4)

            sfondo.convert('RGB').save("./frames/frame1sog1sog2.jpeg")
            print("Immagine: frame1sog1sog2.jpeg creata con successo")
            bar.update(1)  # aggiorni di 1 passo

            # ---- Funzioni utili ----
            def trova_file(base_path):
                estensioni = [".png", ".jpg", ".jpeg"]
                for est in estensioni:
                    path = base_path + est
                    if os.path.exists(path):
                        return path
                return None

            def remove_bg(src_img_path, out_img_path):
                model_choices = ["u2net", "u2net_human_seg", "u2netp"]
                with open(src_img_path, "rb") as f:
                    data = f.read()
                img = remove(
                    data,
                    model_name=model_choices[0],
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_structure_size=10,
                    alpha_matting_base_size=1000
                )
                with open(out_img_path, "wb") as f:
                    f.write(img)

            def riflette_e_sovrascrive(path_img):
                if os.path.exists(path_img):
                    img = Image.open(path_img).convert("RGBA")
                    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img_flip.save(path_img)
                    print(f"Immagine riflessa e salvata: {path_img}")

            # ---- Scontorna soggetti successivi (frame2 e frame3) ----
            j = 5
            for y in range(2, 4):  # y = 2, 3
                for k in range(1, 3):  # k = 1, 2
                    base_name = f"./frames/frame{y}sog{k}"
                    frame_path = trova_file(base_name)
                    if frame_path:
                        print(f"Scontorno {frame_path} -> alfa{j}.png")
                        output_path = f"./frames/alfa{j}.png"
                        remove_bg(frame_path, output_path)
                        j += 1
                    else:
                        print(f"File NON trovato: {base_name}.png/.jpg/.jpeg")
            bar.update(1)  # aggiorni di 1 passo

                
                
                

            # ---- Config collage ----
            sfondo_path = "./frames/loc_crop.png"
            dim_finale = (1024, 1024)

            def crea_collage(file_sinistra, file_destra, flip_left=False, flip_right=False):
                img_bg = Image.open(sfondo_path).convert('RGBA').resize(dim_finale, Image.BICUBIC)
                ws, hs = img_bg.size

                img_left = Image.open(file_sinistra).convert('RGBA')
                img_right = Image.open(file_destra).convert('RGBA')

                # Flip opzionale
                if flip_left:
                    img_left = img_left.transpose(Image.FLIP_LEFT_RIGHT)
                if flip_right:
                    img_right = img_right.transpose(Image.FLIP_LEFT_RIGHT)

                wf_left, hf_left = img_left.size
                wf_right, hf_right = img_right.size

                pos_left = (margine if not flip_left else ws - wf_left - margine, (hs // 2) - (hf_left // 2))
                pos_right = (ws - wf_right - margine if not flip_right else margine, (hs // 2) - (hf_right // 2))

                img_bg.paste(img_left, pos_left, img_left)
                img_bg.paste(img_right, pos_right, img_right)
                return img_bg

            # ---- File soggetti ----
            collage1_files = ("./frames/alfa5.png", "./frames/alfa8.png")
            collage2_files = ("./frames/alfa6.png", "./frames/alfa7.png")

            # ---- Immagini iniziali ----
            img1 = crea_collage(*collage1_files)
            img1.save("./frames/frame2sog1_frame3sog2.png")
            img2 = crea_collage(*collage2_files)
            img2.save("./frames/frame2sog2_frame3sog1.png")

            # ---- Figura e assi ----
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            plt.subplots_adjust(bottom=0.3)

            plot1 = ax1.imshow(img1)
            ax1.set_title("Collage 1")
            ax1.axis("off")

            plot2 = ax2.imshow(img2)
            ax2.set_title("Collage 2")
            ax2.axis("off")

            # ---- Riflessi interattivi ----
            collage_names = {
                1: "frame2sog1_frame3sog2.png",
                2: "frame2sog2_frame3sog1.png"
            }
            flipped1 = False
            flipped2 = False

            def riflettere1(event):
                global flipped1
                flipped1 = not flipped1
                new_img1 = crea_collage(*collage1_files, flip_left=flipped1, flip_right=flipped1)
                riflette_e_sovrascrive("./frames/alfa5.png")
                plot1.set_data(new_img1)
                fig.canvas.draw()
                save_path = os.path.join("./frames", collage_names[1])
                new_img1.save(save_path)
                print(f"Collage 1 salvato in: {save_path}")

            def riflettere2(event):
                global flipped2
                flipped2 = not flipped2
                new_img2 = crea_collage(*collage2_files, flip_left=flipped2, flip_right=flipped2)
                riflette_e_sovrascrive("./frames/alfa6.png")
                plot2.set_data(new_img2)
                fig.canvas.draw()
                save_path = os.path.join("./frames", collage_names[2])
                new_img2.save(save_path)
                print(f"Collage 2 salvato in: {save_path}")

            # ---- Bottoni ----
            ax_button1 = plt.axes([0.25, 0.1, 0.2, 0.1])
            btn1 = Button(ax_button1, "Rifletti Collage 1")
            btn1.on_clicked(riflettere1)

            ax_button2 = plt.axes([0.55, 0.1, 0.2, 0.1])
            btn2 = Button(ax_button2, "Rifletti Collage 2")
            btn2.on_clicked(riflettere2)

            plt.show()
            bar.update(1)  # aggiorni di 1 passo

            pathLocation= "./frames//location.png"
            import os
            from PIL import Image
            if os.path.exists(pathLocation):
                # Carica e croppa l'immagine location
                img = Image.open(pathLocation)
                offset = 241
                w, h = img.size
                box = (offset, offset, w - offset, h - offset)
                cropped_img = img.crop(box)
                cropped_img= cropped_img.resize((1024,1024),Image.BICUBIC)
                cropped_img.save("./frames/loc_crop2.png")
                print("Immagine croppata e salvata correttamente.")
                bar.update(6)  # aggiorni di 1 passo
            # COPPIA frame4sog1 , alfa6 m50
            # COPPIA frame4sog2 , alfa5 m80
            # COPPIA frame5sog1 , alfa6 m50
            # COPPIA frame5sog2 , alfa5 m80
            # COPPIA frame6sog1 , alfa6 m80
            # COPPIA frame6sog2 , alfa5 m100
            # ---- Config collage ----
            import os
            from PIL import Image
            from backgroundremover.bg import remove

            # ---- Funzioni utili ----
            def remove_bg(src_img_path, out_img_path):
                """Rimuove lo sfondo nero e salva PNG trasparente"""
                with open(src_img_path, "rb") as f:
                    data = f.read()
                img = remove(
                    data,
                    model_name="u2net_human_seg",  # modello ottimizzato per corpi umani
                    alpha_matting=False            # disattiva matting
                )
                with open(out_img_path, "wb") as f:
                    f.write(img)
                return out_img_path

            # ---- Config collage ----
            sfondo_path = "./frames/loc_crop2.png"   # sfondo croppato
            dim_finale = (1024, 1024)
            margine=15

            def crea_collage(frame_img, alfa_img, flip_frame=False, flip_alfa=False,m=margine):
                img_bg = Image.open(sfondo_path).convert('RGBA').resize(dim_finale, Image.BICUBIC)

                # rimuovo sfondo a frame e alfa
                base_frame, _ = os.path.splitext(frame_img)
                frame_img_cut = remove_bg(frame_img, base_frame + "_cut.png")
                img_frame = Image.open(frame_img_cut).convert('RGBA')

                base_alfa, _ = os.path.splitext(alfa_img)
                alfa_img_cut = remove_bg(alfa_img, base_alfa + "_cut.png")
                img_alfa = Image.open(alfa_img_cut).convert('RGBA')

                if flip_frame:
                    img_frame = img_frame.transpose(Image.FLIP_LEFT_RIGHT)
                if flip_alfa:
                    img_alfa = img_alfa.transpose(Image.FLIP_LEFT_RIGHT)

                wf, hf = img_frame.size
                wa, ha = img_alfa.size

                # frame sempre centrato verticalmente
                pos_frame = (0, (img_bg.height // 2) - (hf // 2))

                # calcolo centro sfondo
                center_x = img_bg.width // 2
                center_y = img_bg.height // 2

                # logica offset alfa
                if "alfa5" in alfa_img:
                    offset_x = +m   # centro +15 destra
                else:  # alfa6
                    offset_x = -m   # centro +15 sinistra

                pos_alfa = (center_x - (wa // 2) + offset_x, center_y - (ha // 2))

                img_bg.paste(img_frame, pos_frame, img_frame)
                img_bg.paste(img_alfa, pos_alfa, img_alfa)

                return img_bg

            coppie = [
                ("./frames/frame4sog1.jpg", "./frames/alfa6.png", "frame4sog1_alfa6.png", 50),   # m50
                ("./frames/frame4sog2.jpg", "./frames/alfa5.png", "frame4sog2_alfa5.png", 170),   # m80
                ("./frames/frame5sog1.jpg", "./frames/alfa6.png", "frame5sog1_alfa6.png", 50),   # m50
                ("./frames/frame5sog2.jpg", "./frames/alfa5.png", "frame5sog2_alfa5.png", 170),   # m80
                ("./frames/frame6sog1.jpg", "./frames/alfa6.png", "frame6sog1_alfa6.png", 360),   # m80
                ("./frames/frame6sog2.jpg", "./frames/alfa5.png", "frame6sog2_alfa5.png", 560),  # m100
            ]

            # ---- Genera collage ----
            for frame_img, alfa_img, out_name, marg_alfa in coppie:
                if not os.path.exists(frame_img):
                    print(f"ATTENZIONE: {frame_img} mancante, salto...")
                    continue
                if not os.path.exists(alfa_img):
                    print(f"ATTENZIONE: {alfa_img} mancante, salto...")
                    continue

                # flip solo per frame uniti con alfa6
                flip_frame = "alfa6" in alfa_img
                flip_alfa = False

                print(f"Creo collage: {out_name} (flip_frame={flip_frame}, margine={marg_alfa})")
                collage = crea_collage(
                    frame_img, alfa_img,
                    flip_frame=flip_frame,
                    flip_alfa=flip_alfa,
                    m=marg_alfa   # 👈 usa il margine definito nella lista
                )

                save_path = os.path.join("./frames", out_name)
                collage.save(save_path)
                print(f" --> Salvato in {save_path}")
                bar.update(6)  # aggiorni di 1 passo

            bar.n = bar.total
            bar.refresh()

            import os
            os.makedirs("./frames//frame_uniti",exist_ok=True)
            
            import shutil
             # Elenco file finali da spostare
            finali = [
                "frame0sog1sog2.jpeg",
                "frame1sog1sog2.jpeg",
                "frame2sog1_frame3sog2.png",
                "frame2sog2_frame3sog1.png",
                "frame4sog1_alfa6.png",
                "frame4sog2_alfa5.png",
                "frame5sog1_alfa6.png",
                "frame5sog2_alfa5.png",
                "frame6sog1_alfa6.png",
                "frame6sog2_alfa5.png",
            ]

            dest_dir = "./frames/frame_uniti"

            for file in finali:
                src = os.path.join("./frames", file)
                if os.path.exists(src):
                    dst = os.path.join(dest_dir, file)
                    shutil.move(src, dst)
                    print(f"Spostato: {file} -> frame_uniti/")
                else:
                    print(f"ATTENZIONE: {file} non trovato, salto...")


    else:
       print("GENERA IMMAGINE LOCATION, USA LO STEP 4")
    
    if Steps_making.get() in ['Step0', 'Step1', 'Step2', 'Step3', 'Step4', 'Step5','Step6']:
        print("ANIMA FOTOGRAMMI INIZIALI CON facefusion VIDEO ia NSFW, e registati con le email Temporaly: temp-mail.org per avere tentativi infiniti")
        print("\n Avvrei poduto automatizzare la cosa , ma non ero sicuro che funzionasse corettamente il tutto")
        print("\n una volta generate le clip dei fotogrammi iniziali , usa un qualsiasi edit video per montarle insieme e aggiungere altre clip a piacere, audio di preferenza")
        print("   -> PROMPTS: ")
        s1=''
        s1= soggetto1.get()  if "Soggetto 1 pers" in costum_soggetto1.get('1.0','end') else costum_soggetto2.get('1.0','end')
        from deep_translator import GoogleTranslator
        s1= GoogleTranslator(source='it',target='en').translate(s1)
        s2=''
        s2= soggetto2.get()  if "Soggetto 2 pers" in costum_soggetto2.get('1.0','end') else costum_soggetto2.get('1.0','end')
        s2= GoogleTranslator(source='it',target='en').translate(s2)

        prompt1,prompt2= "Two women hugging and French kissing each other with tongue, tongue movement","Two women hugging and French kissing each other with tongue, tongue movement"
        prompt3= f"A {s1} sucks the right nipple of another woman's breast cup"
        prompt4=f"A {s2} sucks the left nipple of another woman's breast cup"
        prompt5= f"A {s1} licks the pussy of a {s2} lying spread-legged in front of her, tongue inside her vulva"
        prompt6= f"A {s2} licks the pussy of a {s1} lying spread-legged in front of her, tongue inside her vulva"
        prompt7=f"a {s1} licks the anus of a {s2} lying with her legs raised high, the {s1} sticks her tongue inside the other {s2}'s asshole"
        prompt8=f"a {s2} licks the anus of a {s1} lying with her legs raised high, the {s2} sticks her tongue inside the other {s1}'s asshole"
        prompt9=f"""A {s1} leans forward with playful curiosity, her eyes twinkling as she slowly reaches out her tongue to gently lick
 the surface of a foot that is right in front of her. The moment her tongue touches the foot, she bites it with a slight crunch, The scene is intimate 
 and slightly surreal, lit by warm, golden light that highlights the texture of the foot and the subtle movement of her expression, a mixture of anticipation 
 and satisfaction. Cinematic atmosphere, soft, dreamlike blur, subtle slow motion."""
        prompt10=f"""A {s2} leans forward with playful curiosity, her eyes twinkling as she slowly reaches out her tongue to gently lick
 the surface of a foot that is right in front of her. The moment her tongue touches the foot, she bites it with a slight crunch, The scene is intimate 
 and slightly surreal, lit by warm, golden light that highlights the texture of the foot and the subtle movement of her expression, a mixture of anticipation 
 and satisfaction. Cinematic atmosphere, soft, dreamlike blur, subtle slow motion."""
        
        def print_prompts():
            frames_prompts = [
                ("frame0sog1sog2", prompt1),
                ("frame1sog1sog2", prompt2),
                ("frame2sog1_frame3sog2", prompt4),
                ("frame2sog2_frame3sog1", prompt3),
                ("frame4sog1_alfa6", prompt5),
                ("frame4sog2_alfa5", prompt6),
                ("frame5sog1_alfa6", prompt7),
                ("frame5sog2_alfa5", prompt8),
                ("frame6sog1_alfa6", prompt9),
                ("frame6sog2_alfa5", prompt10)
            ]

            print("\n===== ANIMA FOTOGRAMMI INIZIALI =====\n")
            for frame, prompt in frames_prompts:
                print(f"--- {frame} ---")
                print("PROMPT:")
                # indentiamo ogni riga lunga del prompt per renderla più leggibile
                for line in prompt.split("\n"):
                    print(f"    {line.strip()}")
                print("\n")  # linea vuota tra i frame

        # poi chiami
        print_prompts()
        import os
        import undetected_chromedriver as uc

        

        # Avvia Chrome "invisibile" come browser umano
        options = uc.ChromeOptions()
        options.add_argument("--no-first-run")
        options.add_argument("--no-service-autorun")
        options.add_argument("--password-store=basic")
        options.add_argument("--start-maximized")  # utile per evitare problemi di layout

        options2 = uc.ChromeOptions()
        options2.add_argument("--no-first-run")
        options2.add_argument("--no-service-autorun")
        options2.add_argument("--password-store=basic")
        options2.add_argument("--start-maximized")  # utile per evitare problemi di layout
        # options.add_argument("--headless=new")   # se vuoi invisibile, ma alcuni siti non caricano in headless
        

        driver1 = uc.Chrome(options=options)
        # prova a navigare
        driver1.get("https://www.facefusion.co/it/ai-video-generator")
        import time 
        time.sleep(1)
        driver2 = uc.Chrome(options=options2)
        driver2.get("https://temp-mail.org/it/")
        time.sleep(1)

        # apri directory Frames (Windows)
        os.startfile(r".\frames\frame_uniti")  # su Linux/macOS usa subprocess

        input("Premi INVIO per chiudere il browser...")

        driver1.quit()
        driver2.quit()
                


        

        
     

     



                





Make_Video = ttk.Button(window, text="Make Video", command=make_video)
Make_Video.grid(row=6, column=0, padx=5, pady=10)

window.mainloop()
