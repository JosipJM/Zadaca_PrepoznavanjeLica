from retinaface import RetinaFace
from tkinter import *
import customtkinter
from deepface import DeepFace
import cv2
import numpy as np
from keras.utils.image_utils import img_to_array
import tkinter.filedialog
import tkinter as tk


global Adresa_Slike
Adresa_Slike=""
global beskorisna_variabla

beskorisna_variabla=0


def button_event():
    global beskorisna_variabla
    global Adresa_Slike
    if Adresa_Slike=="":
        beskorisna_variabla=1
    else:
        beskorisna_variabla=2
    root.destroy()
def GET_PATH():
    global Adresa_Slike
    global filepathforinfo
    filetypes = [('JPG files', '*.jpg'), ('JPEG files', '*.jpeg')]
    filepathforinfo = tk.filedialog.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
    Adresa_Slike=filepathforinfo

root = customtkinter.CTk()
root.eval('tk::PlaceWindow . center')
root.geometry("800x400")
button = customtkinter.CTkButton(master = root, text="Gumb",command=button_event)
button.place(relx=0.5, rely=0.5, anchor=CENTER)
entry = customtkinter.CTkButton(master=root,text="Browse",width=500,height=25,border_width=2,corner_radius=10,command=GET_PATH)
entry.place(relx=0.5, rely=0.3, anchor=CENTER)
texts=customtkinter.CTkLabel(master=root, text="Pronađite sliku i pritisnite gumb ili pritiskom samo na plavi gumb za kameru")
texts.place(relx=0.5, rely=0.23, anchor=CENTER)

root.mainloop()

if beskorisna_variabla==1:


    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    emotion_model = DeepFace.build_model("Emotion")
    age_model = DeepFace.build_model("Age")
    gender_model = DeepFace.build_model("Gender")

    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    gender_labels = ['Male', 'Female']

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        labels = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Get image ready for prediction
            roi = roi_gray.astype('float') / 255.0  # Scale
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

            preds = emotion_model.predict(roi)[0]  # Yields one hot encoded result for 7 classes
            label = class_labels[preds.argmax()]  # Find the label
            label_position = (x + w, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Gender
            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (224, 224), interpolation=cv2.INTER_AREA)
            gender_predict = gender_model.predict(np.array(roi_color).reshape(-1, 224, 224, 3))
            gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
            gender_label = gender_labels[gender_predict[0]]
            gender_label_position = (x + w, y + 50)  # 50 pixels below to move the label outside the face
            cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Age
            age_predict = age_model.predict(np.array(roi_color).reshape(-1, 224, 224, 3))
            age = round(age_predict[0, 0])
            age_label_position = (x + h, y +100)
            cv2.putText(frame, "Age=" + str(age), age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not cv2.getWindowProperty("Emotion Detector", cv2.WND_PROP_VISIBLE):
            break
    cap.release()
    cv2.destroyAllWindows()

from PIL import Image,ImageDraw

if beskorisna_variabla==2:
    img_path = Adresa_Slike
    faces = RetinaFace.extract_faces(img_path)
    feces_position_later = RetinaFace.detect_faces(img_path)
    im = Image.open(img_path)
    obj = [0 for i in range(len(faces))]


    def add_rec(im, topleft, bottomright, color):
        draw = ImageDraw.Draw(im)
        draw.rectangle((topleft, bottomright), outline=color)
        return im


    def add_image(im, text, topleft):

        draw = ImageDraw.Draw(im)
        draw.text(topleft, text, fill="green")
        return im


    for i in range(len(faces)):
        koje_lice = "face_" + str(1 + i)
        obj = DeepFace.analyze(faces[i], ["age", "gender", "race", "emotion"], detector_backend='skip')
        print(obj)
        x1 = feces_position_later[koje_lice]["facial_area"][0]
        y1 = feces_position_later[koje_lice]["facial_area"][1]
        x2 = feces_position_later[koje_lice]["facial_area"][2]
        y2 = feces_position_later[koje_lice]["facial_area"][3]
        dob = str(obj[0]['age'])
        rasa = str(obj[0]['dominant_race'])
        spol = str(obj[0]['dominant_gender'])
        osjecaji = str(obj[0]['dominant_emotion'])
        text = "Age: " + dob + "\n" + "Gender: " + spol + "\n" + "Race: " + rasa + "\n" + "Emotions: " + osjecaji
        im_new = add_rec(im, (x1, y1), (x2, y2), (0, 255, 0))
        im_new = add_image(im, text, (x2 + 10, y1))

    im_new.save('Nova_Slika.jpg')
    im_new.show()


