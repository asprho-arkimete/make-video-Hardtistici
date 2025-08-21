from tkinter import Tk, Button, Canvas, Frame, Scale, filedialog
from moviepy import VideoFileClip
from PIL import Image, ImageTk
import os

# Finestra principale
root = Tk()
root.title("Seleziona Frame da Video")

window = Frame(root)
window.grid()

canvas = Canvas(window, width=512, height=512, bg='red')
canvas.grid(row=0, column=0)

fotogrammi = []
file = None
img_tk = None  # Per mantenere il riferimento all'immagine su canvas

def selezionaframe():
    global img_tk
    if not fotogrammi:
        return
    idx = scale.get()
    f = fotogrammi[idx]
    
    w, h = f.size
    wr, hr = 512, 512
    if w >= h:
        hr = (512 * h) // w
    else:
        wr = (512 * w) // h
    
    f_resize = f.resize((wr, hr), Image.BICUBIC)
    img_tk = ImageTk.PhotoImage(f_resize)
    canvas.delete("all")
    canvas.create_image(256, 256, image=img_tk)  # centro canvas
    canvas.update_idletasks()

def salva_frame():
    if not fotogrammi:
        print("Nessun fotogramma disponibile")
        return
    os.makedirs("frames", exist_ok=True)
    idx = scale.get()
    fname = os.path.basename(file)
    name, _ = os.path.splitext(fname)
    fotogrammi[idx].save(f'frames/{name}_frame_{idx}.jpg')
    print("Frame salvato!")

def seleziona():
    global fotogrammi, file
    file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not file:
        return
    clip = VideoFileClip(file)
    fotogrammi.clear()
    for frame in clip.iter_frames():
        img = Image.fromarray(frame)
        fotogrammi.append(img)
    clip.close()
    scale.config(to=len(fotogrammi) - 1)
    selezionaframe()  # mostra subito il primo frame

scale = Scale(window, from_=0, to=0, resolution=1, orient='horizontal', command=lambda x: selezionaframe())
scale.grid(row=1, column=0)

salva = Button(window, text='Salva Frame', command=salva_frame)
salva.grid(row=1, column=1)

scegligifle = Button(window, text="Seleziona video", command=seleziona)
scegligifle.grid(row=0, column=1)

root.mainloop()
