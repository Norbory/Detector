import numpy as np
import cv2
import pickle
import datetime
from tkinter import *

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels ={"person_name":1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
#Inicializa variables a usar
cesar=0
pedro=0
i = 0
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_,conf = recognizer.predict(roi_gray)
        
        if conf >= 85:  #and conf <= 85:
            print(id_)
            print(labels[id_])
        #Inicia cuenta de tomas de personas
            i+=1
        #Aumentar las variables para cada persona que puede soltar un valor
            if id_==1:
                cesar+=1
            elif id_==2:
                pedro+=1
        #Numero de tomas para encontrar a la persona que acaba de llegar y soltar la alerta en forma de Boton porque no tengo excel aun
            if i>=30:
                now =datetime.datetime.now()
                current_time = now.strftime('%H:%M:%S')
                root = Tk()             
                root.geometry('200x200')
        #Comparacion de valores para hallar la mayor coincidencia
                if cesar > pedro:
                    btn = Button(root, text = "Cesar llego a las " +  current_time, bd = '5',command = root.destroy)
                    btn.pack(side = 'top')
                    root.mainloop()
                else:
                    btn = Button(root, text = "Alguien llego a las " +  current_time, bd = '5',command = root.destroy)
                    btn.pack(side = 'top')
                    root.mainloop()
        #Reset a los valores para las siguientes lecturas            
                i=0
                cesar=0
                pedro=0

            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame , name ,(x,y), font, 1, color, stroke, cv2.LINE_AA)

        item = "6.png"
        cv2.imwrite(item, roi_color)

        color = (255,0,0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)   

    cv2.imshow('Reconocedor', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()