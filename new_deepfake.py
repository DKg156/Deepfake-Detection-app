checkimport = 0
from tkinter import * 
from tkinter import filedialog
from tkinter.ttk import *
import tkinter.messagebox as tkmb
from pathlib import Path
from PIL import ImageTk,Image 
import numpy as np
import pandas as pd
import requests
import os, glob
import textwrap
import re
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()



def imgimport() :
   global inputs,img,checkimport
   img_path = filedialog.askopenfilename(title="Select deepfake image", filetypes=(("JPG File", "*.jpg"),("PNG File", "*.png"), ("All Files", "*.*")))
   if img_path.endswith('.jpg') or img_path.endswith('.JPG') or img_path.endswith('.PNG') or img_path.endswith('.png') or img_path.endswith('.jpeg'):
      
      img = Image.open(img_path)
      img= img.resize((200,200))
     # inputs = ImageLoader.load_image(img)
      canvas1.create_text(240,290,fill="blue",font="Helvetica 12 bold",text="Image imported successfully from System!",tag = "read")
      if(checkimport == 1):
          canvas1.delete("selected")
          canvas1.create_text(260,163,fill="black",font="Helvetica 9 ",text=Path(img_path).name,tag = "selected")
      else:
          canvas1.create_text(260,163,fill="black",font="Helvetica 9 ",text=Path(img_path).name,tag = "selected")
          checkimport = 1
      b_next = Button(f2, text='Next',style = 'W.TButton',command=lambda :next_to_predict())
      b_next.place(relx = 0.8, rely = 0.8, anchor = 'se')
      canvas1.delete("no_url")
   else: 
      tkmb.showerror("Warning","Error: Sorry, file format not supported or no file selected!") 
def backf1():
    f1.tkraise()
    e_url.delete(0, END)
    canvas1.delete("no_url")

def btn_url(url):
    global img,inputs
    canvas1.delete("no_url")
    try:
        img = Image.open(requests.get(url, stream=True).raw)
        img= img.resize((200,200))
        #inputs = ImageLoader.load_image(img)
        canvas1.delete("selected")
        canvas1.delete("read")
        canvas1.create_text(240,290,fill="blue",font="Arial 12 bold",text="Image downloaded successfully from URL!",tag = "read")
        b_next = Button(f2, text='Next',style = 'W.TButton',command=lambda :next_to_predict())
        b_next.place(relx = 0.8, rely = 0.8, anchor = 'se')
    except Exception as e:
        print(e)
        canvas1.create_text(250,240,fill="red",font="Calibri 11 bold",text="Error. Please try again!", tag = "no_url")
        tkmb.showerror(" Error",e) 



def predict():    
    global inputs,b4
    b4['state'] = DISABLED
    face = mtcnn(img)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0) # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    # convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers=[model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        print(prediction)
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        
    print(real_prediction)
        
    global b5
    if (prediction == "real"):
        canvas3.create_text(250,260,fill="green",font="Times 13",text="✅ Real!")
        canvas3.create_text(250,310,fill="black",font="Arial 12",text="Confidence: "+str(real_prediction))
    else:   
        canvas3.create_text(250,250,fill="red",font="Times 13 ",text="⚠️ Fake!")
        canvas3.create_text(250,300,fill="black",font="Arial 12",text="Confidence: "+str(fake_prediction))
        #canvas3.create_text(250,330,fill="green",font="Times 14 ",text="Check prevention strategies below")
        #b5 = Button(f3, text='Symptoms',style = 'W.TButton',command= lambda :symptom(pred_result)).place(relx = 0.75, rely = 0.8, anchor = 'se')
        #b6 = Button(f3, text='Management',style = 'W.TButton',command= lambda :manage(pred_result)).place(relx = 0.5, rely = 0.8, anchor = 'se')
        



def next_to_predict():
    global img,canvas3,b4,bimage
    canvas3 = Canvas(f3,width=500,height=500)
    #bimage = ImageTk.PhotoImage(Image.open("Maiden1.jpg"))
    #canvas3.create_image(0,0,anchor=NW,image=bimage)
    canvas3.place(relx = 0.5, rely = 0.5, anchor = 'center')
    f3.tkraise()
    img.thumbnail((200,200))
    img1 = ImageTk.PhotoImage(img)
    lbl = Label(f3)
    lbl.place(relx = 0.07, rely = 0.03, anchor = 'nw')
    lbl.configure(image=img1)
    lbl.image = img1
    b4 = Button(f3, text='Detect',style = 'W.TButton',state = NORMAL,command=lambda :predict())
    b4.place(relx = 0.8, rely = 0.12, anchor = 'ne')
    bback = Button(f3, text='Back',style = 'W.TButton',command= lambda : backf2()).place(relx = 0.3, rely = 0.9, anchor = 'se')
    
def backf2():
    canvas1.delete("no_url")
    f2.tkraise()
    for widget in f3.winfo_children():
       widget.destroy()
    canvas1.delete("read")






root = Tk() 
#img = PhotoImage(file='doctor.ico')
#root.tk.call('wm', 'iconphoto', root._w, img)
root.title('Deepfake Detection') 
root.geometry("500x500")
root.resizable(0, 0) 
style = Style() 
style.configure('W.TButton', font = ('Calibri', 11),foreground = 'black',highlightbackground = 'blue')
f1 = Frame(root, width=500, height=500)
f2 = Frame(root, width=500, height=500)
f3 = Frame(root, width=500, height=500)

f1.grid(row=0, column=0, sticky = 'news')
f2.grid(row=0, column=0, sticky = 'news')
f3.grid(row=0, column=0, sticky = 'news')

canvas=Canvas(f1,width=500,height=500)
image=ImageTk.PhotoImage(Image.open("face.jpg"))
canvas.create_image(0,0,anchor=NW,image=image)
canvas.place(relx = 0.5, rely = 0.5, anchor = 'center')
canvas.create_text(230,20,fill="black",font="Times 13",text="Welcome")
canvas.create_text(230,170,fill="red",font="Arial 17 bold ",text="DEEPFAKE DETECTION ")
#canvas.create_text(240,200,fill="black",font="Calibri 14 bold ",text="")
b1 = Button(f1, text='Start', style = 'W.TButton',command=lambda :f2.tkraise()).place(relx = 0.75, rely = 0.75, anchor = 'se')
canvas1 = Canvas(f2,width=500,height=500)
#image1 = ImageTk.PhotoImage(Image.open("Plant wallpaper2.jpg"))
#canvas1.create_image(0,0,anchor=NW,image=image)
canvas1.place(relx = 0.5, rely = 0.5, anchor = 'center')
b2 = Button(f2, text='Back',style = 'W.TButton',command= lambda : backf1()).place(relx = 0.4, rely = 0.8, anchor = 'se')
canvas1.create_text(240,70,fill="darkblue",font="Calibri 14 bold",text="Import your image from System or Web")
canvas1.create_text(250,450,fill="blue",font="Arial 9",text="Images of all sizes are automatically resized. Formats accepted: jpg, png")
b = Button(f2, text='Browse image',style = 'W.TButton',command= lambda :imgimport()).place(relx = 0.55, rely = 0.25, anchor = 'ne')
canvas1.create_text(70,220,fill="black",font="Calibri 11 bold",text="URL")
e_url = Entry(f2)
e_url.place(x = 90,y = 205,width=300,height=25)
b3 = Button(f2, text='Download',style = 'W.TButton',command= lambda :btn_url(e_url.get())).place(relx = 0.96, rely = 0.41, anchor = 'ne')

f1.tkraise()
root.mainloop() 
