import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import random
import time
import playsound as ps
import tkinter as tk
from tkinter import messagebox







fname = str(random.randint(0, 99999))


np.set_printoptions(suppress=True)


model = load_model("C:/Users/pytho/Desktop/faceApp/keras_model.h5", compile=False)


class_names = open("C:/Users/pytho/Desktop/faceApp/labels.txt", "r").readlines()


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


def start_app1():

    while True:
        cap = cv2.VideoCapture(0)  

        if not cap.isOpened():
            print("Kamera açılamadı.")
        else:

            ret, frame = cap.read()


            if ret:
            
                image_path = f"images/{fname}.jpg"
                cv2.imwrite(image_path, frame)


                cap.release()

                image = Image.open(image_path).convert("RGB")


                    

                size = (224, 224)
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data[0] = normalized_image_array

                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]

                print("Class:", class_name[2:], end="")
                print("Confidence Score:", confidence_score)

                if 'alper' in class_name.lower():
                    print('sahip yüz algılandı!')
                    tk.messagebox.showinfo(title='SAHİP YÜZ', message='SAHİP YÜZ ALGILANDI')


                        
                elif 'arkaplan' in class_name.lower():
                    print('oda boş.')
                    tk.messagebox.showerror(title='SAHİP YÜZ', message='SAHİP YÜZ ALGILANAMADI')


                else:
                    print('odada yabancı varlık algılandı!')




            else:
                print("Kare yakalanamadı.")



            



app = tk.Tk()
app.resizable(False, False)
app.geometry('400x75')
app.title('YÜZ TANIMA YETKİLENDİRİCİSİ')

by = tk.Label(text='by Alper Tuna\nKILIÇ')
by.place(x=10, y=10)

start_app = tk.Button(text='UYGULAMAYI BAŞLAT', font='Helvetica 16', command=start_app1)
start_app.place(x = 100, y= 15)



app.mainloop()





    