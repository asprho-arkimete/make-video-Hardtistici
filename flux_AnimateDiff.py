import os
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import (
    Tk, Canvas, Frame, Text, Scale, Button, messagebox, HORIZONTAL, Label
)
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
from deep_translator import GoogleTranslator

# ----------------------------
# Costanti e Variabili
# ----------------------------

CANVAS_SIZE = 720

GENERATED_IMAGE_PATH = "image_flux.png"
GENERATED_IMAGE_PATH2="image_sd15.png"

os.makedirs("Lora", exist_ok=True)

image_path = ""
ref_image_path = ""

# ----------------------------
# Funzioni Utili
# ----------------------------

def aggiorna_lora():
    lora_files = [f for f in os.listdir("Lora") if os.path.isfile(os.path.join("Lora", f))]
    LoraCombo['values'] = ["No_Lora"] + sorted(set(lora_files))

import random  # Assicurati che sia importato in cima al file
def genera_immagine():
    import os
    import random
    import subprocess
    from tkinter import messagebox
    from deep_translator import GoogleTranslator

    global steps_scale, cfg_scale, resolution_combobox
    global scale_IP_adapter, ref_image_path, seed, modalita_textImage
    global LoraCombo, scale_lora, text_prompt, GENERATED_IMAGE_PATH, GENERATED_IMAGE_PATH2
    global canvas_img, CANVAS_SIZE, model, image_path,scale_modifica

    prompt = text_prompt.get("1.0", "end").strip().replace('"', "'")
    if not prompt:
        messagebox.showwarning("Attenzione", "Inserisci un prompt valido.")
        return

    try:
        translated = GoogleTranslator(source='it', target='en').translate(prompt)
    except Exception as e:
        print(f"Errore traduzione: {e}")
        translated = prompt

    try:
        steps = int(steps_scale.get())
        cfg = float(cfg_scale.get())
        resolution = resolution_combobox.get()
        width, height = map(int, resolution.split(','))
    except Exception:
        messagebox.showerror("Errore", "Controlla i parametri numerici o la risoluzione.")
        return

    try:
        seed_input = seed.get("1.0", "end").strip()
        seed_val = int(seed_input) if seed_input != "-1" else random.randint(1, 100000000)
    except Exception:
        seed_val = random.randint(1, 100000000)

    lora_name = LoraCombo.get()
    lora_scale_val = float(scale_lora.get())
    model_filename = model.get().strip()
    image_path = None

    if modalita_textImage.get() == 'TextImage_Flux':
        cmd = (
            f'python fluxtext.py --prompt "{translated}" '
            f'--steps {steps} --guidance_scale {cfg} '
            f'--width {width} --height {height} '
            f'--output "{GENERATED_IMAGE_PATH}" '
            f'--seed {seed_val} '
        )

        if lora_name.lower() != "no_lora":
            cmd += f'--lora "{lora_name}" --scale {lora_scale_val} '

        if ref_image_path and os.path.exists(ref_image_path):
            try:
                scale_IP = float(scale_IP_adapter.get())
                cmd += (
                    f'--use_ip_adapter '
                    f'--scale_ip_adapter {scale_IP} '
                    f'--ip_adapter_image "{ref_image_path}" '
                )
            except Exception:
                messagebox.showwarning("Attenzione", "Scala IP Adapter non valida. Ignorato.")

        print("Eseguendo:", cmd)
        os.system(cmd)
        image_path = GENERATED_IMAGE_PATH

    elif modalita_textImage.get() == 'TxtImage_SD1.5':
        cmd = (
            f'python textsd.py --prompt "{translated}" '
            f'--risoluzione "{width},{height}" '
            f'--steps {steps} --cfg {cfg} '
            f'--output "{GENERATED_IMAGE_PATH2}" '
            f'--seed {seed_val} '
            f'--model "{model_filename}" '
        )

        if image_path and os.path.exists(image_path):
            cmd += f'--pose "{image_path}" '

        if lora_name.lower() != "no_lora":
            cmd += f'--lora "{lora_name}" --lorascale {lora_scale_val} '

        if ref_image_path and os.path.exists(ref_image_path):
            try:
                scale_IP = float(scale_IP_adapter.get())
                cmd += f'--ipadapter "{ref_image_path}" --scaleIP {scale_IP} '
            except Exception:
                messagebox.showwarning("Attenzione", "Scala IP Adapter non valida. Ignorato.")

        print("Eseguendo:", cmd)
        os.system(cmd)
        # Caricamento immagine generata
        image_path = GENERATED_IMAGE_PATH2
    elif modalita_textImage.get() == "sd_img2img":
        output_path = "./sd_img2img.png"
        resolution = f"{width},{height}"
        modifica_val = float(scale_modifica.get())
        
        image_path = "./image.png"  # ControlNet input

        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Errore", "Devi fornire un'immagine valida per Img2Img (ControlNet).")
            return

        cmd = (
            f'python sd_img2img.py '
            f'--path_image "{image_path}" '
            f'--prompt "{translated}" '
            f'--risoluzione "{resolution}" '
            f'--steps {steps} --cfg {cfg} '
            f'--seed {seed_val} '
            f'--model "{model_filename}" '
            f'--modifica {modifica_val} '
        )

        if lora_name.lower() != "no_lora":
            cmd += f'--lora "{lora_name}" --scale_lora {lora_scale_val} '

        scale_IP = 0.5  # default
        ref_image_path= "image_riferimento.png"
        if ref_image_path and os.path.exists(ref_image_path):
            try:
                scale_IP = float(scale_IP_adapter.get())
                cmd += f'--ip_adapter "{ref_image_path}" --scale_ip_adapter {scale_IP} '
            except Exception:
                messagebox.showwarning("Attenzione", "Scala IP Adapter non valida. Ignorato.")

        print("Eseguendo:", cmd)
        subprocess.run(cmd, shell=True)
        image_path = output_path


    
    if os.path.exists(image_path):
        carica_immagine_su_canvas(image_path, canvas_img, CANVAS_SIZE)
    else:
        messagebox.showerror("Errore", "Immagine non trovata. Qualcosa è andato storto nella generazione.")


def F_Inpainting():
    import subprocess
    from deep_translator import GoogleTranslator
    global image_path, steps_scale, cfg_scale, resolution_combobox, ref_image_path, scale_modifica, text_prompt, scale_IP_adapter, modalita_inpaint, LoraCombo, scale_lora

    # Traduzione prompt da italiano a inglese
    prompt = text_prompt.get("1.0", "end").strip()
    translated = GoogleTranslator(source='it', target='en').translate(prompt)

    if not image_path:
        image_path = "image_flux.png"

    resolution = resolution_combobox.get()

    if modalita_inpaint.get() == '' or modalita_inpaint.get() == "Inpant_stable_D_1.5":
        print("<-- INPAINTING CON STABLE DIFFUSION 1.5 -->")
        base_cmd = (
            f'python inpainting_sd.py '
            f'--prompt "{translated}" '
            f'--sorce_image "image.png" '
            f'--risoluzione {resolution} '
            f'--steps {int(steps_scale.get())} '
            f'--cfg {float(cfg_scale.get())} '
            f'--modifica {float(scale_modifica.get())} '
            f'--scale_ipadapter {float(scale_IP_adapter.get())}'
        )

        if ref_image_path and os.path.exists(ref_image_path):
            base_cmd += f' --ip_use --reference "{ref_image_path}"'

        print("📤 Comando in esecuzione:", base_cmd)
        subprocess.run(base_cmd, shell=True)

    elif modalita_inpaint.get() == "Flux_Inpant":
        print("<-- INPAINTING CON FLUX -->")
        flux_cmd = (
            f'python Flux_inpainting.py '
            f'--prompt "{translated}" '
            f'--source "image.png" '  # <-- Spazio aggiunto qui
            f'--risoluzione {resolution} '
            f'--steps {int(steps_scale.get())} '
            f'--cfg {float(cfg_scale.get())} '
            f'--modifica {float(scale_modifica.get())} '
        )

        if ref_image_path and os.path.exists(ref_image_path):
            flux_cmd += f'--ip_adapter --scale_ip_adapter {scale_IP_adapter.get()} '

        # Aggiunta opzionale LoRA
        lora_file_selected = LoraCombo.get()
        if lora_file_selected and lora_file_selected != "No_Lora":
            lora_path = os.path.join(".", "Lora", lora_file_selected)
            if os.path.exists(lora_path):
                flux_cmd += f'--lorafile "{lora_file_selected}" --lorascale {float(scale_lora.get())} '
            else:
                print(f"⚠️ File LoRA '{lora_file_selected}' non trovato nella cartella ./Lora")

        print("📤 Comando in esecuzione:", flux_cmd)
        subprocess.run(flux_cmd, shell=True)
# Variabili globali
image_path = None
ref_image_path = None
tk_image = None  # Necessario per mantenere il riferimento ed evitare garbage collection
tk_ref_image = None  # Anche per l'immagine di riferimento
img_lavoro = None  # immagine in memoria per gomma e modifiche

# Funzione riutilizzabile per caricare immagine su un canvas centrata
def carica_immagine_su_canvas(path, canvas, canvas_size):
    global tk_image, tk_ref_image, img_lavoro

    img = Image.open(path).convert("RGB")
    original_w, original_h = img.size

    if original_w >= original_h:
        new_w = canvas_size
        new_h = (canvas_size * original_h) // original_w
    else:
        new_h = canvas_size
        new_w = (canvas_size * original_w) // original_h

    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Salva immagine ridimensionata come immagine di lavoro (usata dalla gomma)
    img_lavoro = img.copy()

    tk_img = ImageTk.PhotoImage(img)

    # Salva immagine ridimensionata se è quella principale
    if path == "image.png":
        img.save(path)
        global tk_image
        tk_image = tk_img
    else:
        global tk_ref_image
        tk_ref_image = tk_img

    canvas.delete("all")
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2
    canvas.create_image(offset_x, offset_y, anchor="nw", image=tk_img)

# --- Gestione drop immagine principale ---
def handle_image_drop(event):
    global image_path, img_lavoro
    file_path = event.data.strip('{}')
    if not os.path.isfile(file_path) or not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        messagebox.showwarning("Attenzione", "Trascina un file immagine valido.")
        return

    image_path = "image.png"
    Image.open(file_path).convert("RGB").save(image_path)
    carica_immagine_su_canvas(image_path, canvas_img, CANVAS_SIZE)

from PIL import Image, ImageTk, ImageDraw




# Variabili globali per disegno
lazo = False
cancellino = False
points = []
points_clear = []
line_ids = []

def attiva_lazo(event=None):
    global lazo, points, cancellino
    lazo = True
    cancellino = False
    points = []  # Reset punti

def disegna(event):
    global lazo, points, line_ids
    if lazo:
        points.append((event.x, event.y))
        if len(points) > 1:
            line_id = canvas_img.create_line(points[-2], points[-1], fill="white", width=2)
            line_ids.append(line_id)

def disattiva_lazo(event=None):
    global lazo
    lazo = False
    if len(points) > 2:
        # Chiude e riempie il poligono
        canvas_img.create_line(points[-1], points[0], fill="white", dash=(3, 2))
        canvas_img.create_polygon(points, outline="white", fill="white", width=1)
        crea_mask()

def cancella(event=None):
    global cancellino, lazo
    cancellino = True
    lazo = False
    print(f"cancellino {cancellino}")

def fine_cancellazione(event=None):
    global cancellino, points_clear
    cancellino = False
    crea_mask()
    points_clear.clear()



def crea_mask():
    global points, image_path

    if not points or not os.path.exists(image_path):
        return

    img = Image.open(image_path)
    img_w, img_h = img.size

    offset_x = (CANVAS_SIZE - img_w) // 2
    offset_y = (CANVAS_SIZE - img_h) // 2

    # Carica o crea la maschera
    if os.path.exists("mask.png"):
        mask = Image.open("mask.png").convert('L')
        mask = mask.copy()  # Per poter modificare la copia
    else:
        mask = Image.new("L", (img_w, img_h), color=0)

    draw = ImageDraw.Draw(mask)

    translated_points = [
        (x - offset_x, y - offset_y)
        for (x, y) in points
        if 0 <= x - offset_x < img_w and 0 <= y - offset_y < img_h
    ]

    if len(translated_points) >= 3:
        draw.polygon(translated_points, fill=255)
        mask.save("mask.png")
        print("[INFO] Maschera salvata come 'mask.png'")
    else:
        print("[WARN] Non abbastanza punti validi nel poligono")

    # Pulisci i punti dopo il salvataggio
    points.clear()

def disegna_cancellino(event):
    global cancellino, points, points_clear, tk_image, img_lavoro

    if not cancellino:
        return

    try:
        if (event.x, event.y) not in points_clear:
            points_clear.append((event.x, event.y))

        def is_near(p1, p2, tol=5):
            return abs(p1[0] - p2[0]) <= tol and abs(p1[1] - p2[1]) <= tol

        # Rimuove punti vicini alla penna di cancellazione
        points[:] = [p for p in points if all(not is_near(p, pc) for pc in points_clear)]

        # Ricarica l'immagine base sul canvas
        if "image_path" in globals() and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGBA")
            img_w, img_h = img.size
            canvas_w = canvas_img.winfo_width()
            canvas_h = canvas_img.winfo_height()
            offset_x = (canvas_w - img_w) // 2
            offset_y = (canvas_h - img_h) // 2
            tk_image = ImageTk.PhotoImage(img)
            canvas_img.delete("all")
            canvas_img.create_image(offset_x, offset_y, anchor="nw", image=tk_image, tags="image")
            canvas_img.tag_lower("image")

        # Ridisegna linee
        if len(points) >= 2:
            for i in range(len(points) - 1):
                canvas_img.create_line(points[i], points[i+1], fill="white", width=2)
        if len(points) >= 3:
            canvas_img.create_line(points[-1], points[0], fill="white", dash=(3, 2))
            canvas_img.create_polygon(points, outline="white", fill="white", width=1)

        # Modifica la maschera: imposta a 0 (nero) i punti vicini al cancellino
        if os.path.exists("mask.png"):
            mask = Image.open("mask.png").convert("L")
            mask = mask.copy()
            draw = ImageDraw.Draw(mask)

            # Coordinate immagine
            img_mask = Image.open(image_path)
            img_w, img_h = img_mask.size
            offset_x = (CANVAS_SIZE - img_w) // 2
            offset_y = (CANVAS_SIZE - img_h) // 2

            translated_clear_points = [
                (x - offset_x, y - offset_y)
                for (x, y) in points_clear
                if 0 <= x - offset_x < img_w and 0 <= y - offset_y < img_h
            ]

            for (x, y) in translated_clear_points:
                draw.ellipse((x-10, y-10, x+10, y+10), fill=0)  # Crea un cerchio nero

            mask.save("mask.png")
            print("[INFO] Maschera aggiornata con cancellatura.")

    except Exception as error:
        print(f"[ERROR] disegna_cancellino: {error}")
    

    



def F_kon_text():
    import os
    import sys
    import subprocess
    from deep_translator import GoogleTranslator

    global image_path, steps_scale, cfg_scale, resolution_combobox
    global ref_image_path, scale_modifica, text_prompt, scale_IP_adapter
    global modalita_inpaint, LoraCombo, scale_lora

    # Traduzione del prompt da italiano a inglese
    prompt = text_prompt.get("1.0", "end").strip()
    translated = GoogleTranslator(source='it', target='en').translate(prompt)

    if not image_path:
        image_path = "image_flux.png"

    resolution = resolution_combobox.get()

    print("<-- FLUX KON TEXT -->")

    # Comando costruito in stile F_Inpainting
    base_cmd = (
        f'{sys.executable} flux_kon_text.py '
        f'--input_image "{image_path}" '
        f'--prompt "{translated}" '
        f'--cfg {float(cfg_scale.get())} '
        f'--steps {int(steps_scale.get())} '
        f'--risoluzione {resolution} '
        f'--lora "{LoraCombo.get()}" '
        f'--scale_lora {float(scale_lora.get())}'
    )

    print("📤 Comando in esecuzione:", base_cmd)
    subprocess.run(base_cmd, shell=True)

ref_image_path1 = None
ref_image_path2 = None


# ----------------------------
# Interfaccia Grafica
# ----------------------------

window = TkinterDnD.Tk()
window.title("Flux Text-to-Image Generator")
window.geometry("1600x1000")

# ----------------------------
# Frame principale
# ----------------------------
main_frame = Frame(window)
main_frame.pack(padx=10, pady=10, fill="both", expand=True)

# ----------------------------
# Frame per canvas immagine principale e riferimento
# ----------------------------
canvas_container = Frame(main_frame)
canvas_container.grid(row=0, column=0, sticky="nw")  # Allinea in alto a sinistra

# --- Canvas immagine principale ---
canvas_img = Canvas(canvas_container, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="#8B0000")
canvas_img.create_text(CANVAS_SIZE // 2, CANVAS_SIZE // 2,
                        text="Trascina immagine qui", fill="white", font=("Arial", 20))
canvas_img.pack(side="left", padx=10)

canvas_img.drop_target_register(DND_FILES)
canvas_img.dnd_bind("<<Drop>>", handle_image_drop)

# Eventi mouse per disegno
canvas_img.bind("<Button-1>", attiva_lazo)
canvas_img.bind("<B1-Motion>", disegna)
canvas_img.bind("<ButtonRelease-1>", disattiva_lazo)
canvas_img.bind("<Button-3>", cancella)
canvas_img.bind("<B3-Motion>", disegna_cancellino)
canvas_img.bind("<ButtonRelease-3>", fine_cancellazione)

# Forza l'aggiornamento per poter leggere le dimensioni
window.update()

# --- Gestione drop immagine di riferimento ---
def handle_ref_drop(event):
    global ref_image_path1, ref_image_path2

    widget = event.widget
    file_path = event.data.strip('{}')
    if not os.path.isfile(file_path) or not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        messagebox.showwarning("Attenzione", "Trascina un file immagine valido.")
        return

    if widget == canvas_ref:
        ref_image_path1 = "image_riferimento.png"
        Image.open(file_path).convert("RGB").save(ref_image_path1)
        carica_immagine_su_canvas(ref_image_path1, canvas_ref, 228)
    elif widget == canvas_ref2:
        ref_image_path2 = "immagine_riferimento_2.png"
        Image.open(file_path).convert("RGB").save(ref_image_path2)
        carica_immagine_su_canvas(ref_image_path2, canvas_ref2, 228)

# --- Canvas immagine di riferimento 1 ---
canvas_ref_x = canvas_img.winfo_x() + canvas_img.winfo_width() + 10
canvas_ref_y = canvas_img.winfo_y()

canvas_ref = Canvas(window, width=228, height=228, bg="#00008B")
canvas_ref.create_text(114, 114,
                        text="Trascina\nimmagine\nriferimento", fill="white", font=("Arial", 12), justify="center")
canvas_ref.place(x=canvas_ref_x, y=canvas_ref_y)

canvas_ref.drop_target_register(DND_FILES)
canvas_ref.dnd_bind("<<Drop>>", handle_ref_drop)

# --- Canvas immagine di riferimento 2 (sotto la prima) ---
canvas_ref2 = Canvas(window, width=228, height=228, bg="#00008B")
canvas_ref2.create_text(114, 114,
                        text="Trascina\nimmagine\nriferimento", fill="white", font=("Arial", 12), justify="center")
canvas_ref2.place(x=canvas_ref_x, y=canvas_ref_y + 238)

canvas_ref2.drop_target_register(DND_FILES)
canvas_ref2.dnd_bind("<<Drop>>", handle_ref_drop)


# --- Bottoni: Genera e Inpainting - in alto a destra della canvas_ref ---
bottoni_frame = Frame(window)
bottoni_frame.place(
    x=int(CANVAS_SIZE+228+630),  # appena a destra del bordo destro
    y=canvas_ref_y - 10,                              # appena sopra il bordo superiore
    anchor='ne'                                       # ancoraggio in alto a destra
)

generate_button = Button(bottoni_frame, text="Genera immagine", font=("Arial", 12), width=20, height=2,
                         command=genera_immagine, bg="orange", fg="black")
generate_button.grid(row=0, column=0, padx=5, pady=5)

inpaint_button = Button(bottoni_frame, text="Inpainting", font=("Arial", 12), width=20, height=2,
                        command=F_Inpainting, bg="green", fg="black")
inpaint_button.grid(row=0, column=1, padx=5, pady=5)


Flux_KonText = Button(bottoni_frame, text="Flux_Kon_Text", font=("Arial", 12), width=20, height=2,
                      command=F_kon_text, bg="#0073ff", fg="black")
Flux_KonText.grid(row=0, column=2, padx=5, pady=5)

import undetected_chromedriver as uc
from tkinter import Button
import tkinter.ttk as ttk
import os
import webbrowser

# Variabili globali
brosw_driver = False
emai_temp = False
driver1 = None

def generavideo_web():
    global brosw_driver, emai_temp, driver1

    if not emai_temp:
        webbrowser.open("https://temp-mail.org/it/")
        emai_temp = True

    if not brosw_driver:
        # Avvia Chrome "invisibile" come browser umano
        options = uc.ChromeOptions()
        # options.add_argument("--headless=new")   # se vuoi modalità nascosta
        options.add_argument("--no-first-run --no-service-autorun --password-store=basic")
        driver1 = uc.Chrome(options=options)
        brosw_driver = True

    scelta = select_video.get()
    if scelta == 'Video NSFW':
        driver1.get("https://www.facefusion.co/it/ai-video-generator")
    elif scelta == 'LTX Studio':
        driver1.get("https://ltx.studio/")
    elif scelta == 'makefilm.ai':
        driver1.get("https://makefilm.ai/tools/nsfw-ai-video-generator")
    elif scelta == 'Estrai Frame':
        driver1.quit()
        brosw_driver = False
        os.system('python estraiframe.py')
        return

    input("Premi INVIO per chiudere il browser...")
    driver1.quit()
    brosw_driver = False

# Assicurati che `bottoni_frame` sia già definito altrove
video_generator = Button(
    bottoni_frame,
    text="Video Generator",
    font=("Arial", 12),
    width=20,
    height=2,
    command=generavideo_web,
    bg="#ac5fe5",
    fg="black"
)
video_generator.grid(row=1, column=2, padx=5, pady=5)

select_video = ttk.Combobox(
    bottoni_frame,
    values=['Video NSFW', 'LTX Studio', 'makefilm.ai', 'Estrai Frame']
)
select_video.grid(row=2, column=2, padx=5, pady=5)

import os
import psutil
import threading  # modulo corretto per i thread

def is_control_panel_running():
    """Controlla se il processo 'Control Panel' è in esecuzione"""
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            # Controlla il nome del processo (come appare nel Task Manager)
            if proc.info['name'] == 'Control Panel':
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False

def launch_control_panel():
    if not is_control_panel_running():
        # Funzione da eseguire nel thread
        def run():
            os.system("python control.py")
        
        # Crea e avvia un thread in background
        thread = threading.Thread(target=run, daemon=False)
        thread.start()
        print("Avviato Control Panel in un nuovo thread")
    else:
        print("Control Panel è già in esecuzione!")

# Usa questo nella tua GUI
Make_Scanes = Button(
    bottoni_frame,
    text="Panel Make Scenes",
    bg='red',
    fg='white',
    font=('Helvetica', 16, 'bold'),
    relief='raised',
    bd=5,
    padx=10,
    pady=10,
    activebackground='darkred',
    activeforeground='white',
    command=launch_control_panel
)

Make_Scanes.grid(row=3, column=2, padx=5, pady=5)



modalita_textImage= ttk.Combobox(bottoni_frame, values=['TxtImage_SD1.5','TextImage_Flux','sd_img2img'])
modalita_textImage.set('TxtImage_SD1.5')
modalita_textImage.grid(row=1,column=0,padx=5,pady=5)

modalita_inpaint= ttk.Combobox(bottoni_frame, values=['Inpant_stable_D_1.5','Flux_Inpant'])
modalita_inpaint.set('Flux_Inpant')
modalita_inpaint.grid(row=1,column=1,padx=5,pady=5)

def F_azzera_mask():
    global points, points_clear,image_path  
    # Reset maschera
    if os.path.exists("mask.png"):
        os.remove("mask.png")
        points=[]
        points_clear=[]
        canvas_img.delete("all")
        # Se esiste, ricarica l'immagine originale
        if  not image_path=='' and os.path.exists(image_path):
            canvas_img.delete('All')
            img = Image.open(image_path)
            img_w, img_h = img.size
            canvas_w = canvas_img.winfo_width()
            canvas_h = canvas_img.winfo_height()
            offset_x = (canvas_w - img_w) // 2
            offset_y = (canvas_h - img_h) // 2
            tk_image = ImageTk.PhotoImage(img)
            canvas_img.create_image(offset_x, offset_y, anchor="nw", image=tk_image)
            canvas_img.update_idletasks()
        print("[INFO] Maschera azzerata.")

# Bottone "Azzera Mask" - CORRETTO
Azzerra_mask = Button(
    bottoni_frame,
    text='Azzera Mask',
    bg='black',
    fg='white',  # Colore del testo
    font=('Arial', 12),  # Usa una tupla senza keyword per il font

command=F_azzera_mask)
Azzerra_mask.grid(row=2, column=1, padx=5, pady=5)

# ----------------------------
# Frame strumenti sotto alle canvas
# ----------------------------
strumenti = Frame(main_frame)
strumenti.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

# --- Prompt text area ---
text_prompt = Text(strumenti, width=70, height=4)
text_prompt.insert("1.0", "una ragazza bionda ,capelli mossi, indossa una lingerie , seduta ad una sedia")
text_prompt.grid(row=0, column=0, columnspan=7, sticky="ew", padx=5, pady=5)

# --- Model ---
def aggiorna_modelli(event=None):
    global modellis
    modellis = [os.path.basename(m) for m in os.listdir('modelli') if m.endswith('.safetensors')]
    model['values'] = modellis  # aggiorna dinamicamente la combobox

# Inizializzazione della Combobox (inizialmente vuota)
model = ttk.Combobox(strumenti, values=[], state='readonly')  # state='readonly' impedisce input manuale
model.grid(row=2, column=0, padx=5, pady=5)
model.bind('<Button-1>', aggiorna_modelli)  # aggiorna elenco quando si clicca
model.set('hyperRealism_30.safetensors')

# --- Steps ---
steps_scale = Scale(strumenti, from_=1, to=100, resolution=1, label="Steps", orient=HORIZONTAL)
steps_scale.set(50)
steps_scale.grid(row=2, column=1, padx=5)

# --- CFG ---
cfg_scale = Scale(strumenti, from_=0, to=30, resolution=0.1, label="CFG", orient=HORIZONTAL)
cfg_scale.set(7.5)
cfg_scale.grid(row=2, column=2, padx=5)

# --- Risoluzione Label + Combobox ---
Label(strumenti, text="Risoluzione").grid(row=1, column=3)
resolution_combobox = ttk.Combobox(strumenti, values=[
    '512,512', '512,720', '720,512', '768,768', '1024,768', '768,1024',
    '960,512', '512,960', '960,720', '720,960', '960,960', '1280,720',
    '720,1280', '1024,1024', '1920,1080', '1080,1920'
])
resolution_combobox.set('1024,1024')
resolution_combobox.grid(row=2, column=3, padx=5)

# --- LoRA Label + Combobox ---
Label(strumenti, text="LoRA").grid(row=1, column=4)
LoraCombo = ttk.Combobox(strumenti)
aggiorna_lora()
LoraCombo.set("No_Lora")
LoraCombo.grid(row=2, column=4, padx=5)

# --- LoRA Scale ---
scale_lora = Scale(strumenti, from_=0.0, to=1.0, resolution=0.1, orient=HORIZONTAL, label="LoRA Scale")
scale_lora.set(0.7)
scale_lora.grid(row=2, column=5, padx=5)

# --- Modifica Scale ---
scale_modifica = Scale(strumenti, from_=0.0, to=1.0, resolution=0.01, orient=HORIZONTAL, label="Modifica")
scale_modifica.set(0.85)
scale_modifica.grid(row=2, column=6, padx=5)

# --- IP Adapter Scale ---
scale_IP_adapter = Scale(strumenti, from_=0.0, to=1.0, resolution=0.01, orient=HORIZONTAL, label="Scala IP_adapter")
scale_IP_adapter.set(0.50)
scale_IP_adapter.grid(row=2, column=7, padx=5)

seed = Text(strumenti, width=10, height=1)
seed.insert('1.0', '-1')
seed.grid(row=2, column=8, padx=5)



# ----------------------------
# Avvio dell'applicazione
# ----------------------------
window.mainloop()