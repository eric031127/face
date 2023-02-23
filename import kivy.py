import kivy
import cv2
import requests
import numpy as np
import os
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout


r = requests.get('https://raw.githubusercontent.com/kipr/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml')
s = r.text
with open(f'C:/Users/cheri/Desktop/homework/APP/text.xml','w') as f:
    f.write(s)
    

class button(App):
    def build(self):
        btu = Button(text = "train",
                 font_size ="20sp",
                   background_color =(1, 1, 1, 1),
                   color =(1, 1, 1, 1),
                   size =(32, 20),
                   size_hint =(.2, .2),
                   pos =(100, 400))
        btu.bind(on_press = self.callback)
        btu1 = Button(text = "test",
                 font_size ="20sp",
                   background_color =(1, 1, 1, 1),
                   color =(1, 1, 1, 1),
                   size =(32, 20),
                   size_hint =(.2, .2),
                   pos =(100, 100))
        btu1.bind(on_press = self.callback1)
        boxlayout = BoxLayout()
        boxlayout.add_widget(btu)
        boxlayout.add_widget(btu1)
        return boxlayout
    def callback(self,event):
        face_detector = cv2.CascadeClassifier('C:/Users/cheri/Desktop/homework/APP/text.xml')  
        samples = []
        labels = []                             
        cap = cv2.VideoCapture(0)                    
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, frame = cap.read()                     
            if not ret:
                print("Cannot receive frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (200, 200))
                samples.append(roi_gray)
                labels.append(0)                             
            cv2.imshow('0', frame)  
            
            if  cv2.waitKey(1) & 0xFF == 27:           
                break 
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(samples, np.array(labels))
        model_path = 'model/trained_model.xml'
        if not os.path.exists('model'):
            os.mkdir('model')
        recognizer.write(model_path)                           
        print('ok!')
        cv2.destroyAllWindows()
    def callback1(self,event):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('model/trained_model.xml')
        face_cascade = cv2.CascadeClassifier('C:/Users/cheri/Desktop/homework/APP/text.xml')        
        cap = cv2.VideoCapture(0)                                 
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (200,200))

                
                id_, confidence = recognizer.predict(roi_gray)

                
                if confidence < 50:
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = 'Unknown'
                    if id_ == 0:
                        name = 'User1'
                    elif id_ == 1:
                        name = 'User2'
                    elif id_ == 2:
                        name = 'User3'
                    elif id_ == 3:
                        name = 'User4'
                    elif id_ == 4:
                        name = 'User5'
                    elif id_ == 5:
                        name = 'User6'
                    elif id_ == 6:
                        name = 'User7'
                    elif id_ == 7:
                        name = 'User8'
                    elif id_ == 8:
                        name = 'User9'
                    elif id_ == 9:
                        name = 'User10'
                    elif id_ == 10:
                        name = 'User11'
                    cv2.putText(frame, name, (x+5, y-5), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            
            cv2.imshow('Face Recognition', frame)

            
            if cv2.waitKey(1) & 0xFF == 27:
                break  
        cap.release()
        cv2.destroyAllWindows()
        

        
if __name__ == "__main__":
    button().run()
