import kivy
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import requests

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
        detector = cv2.CascadeClassifier('C:/Users/cheri/Desktop/homework/APP/text.xml') 
        recognition = cv2.face.LBPHFaceRecognizer_create()    
        faces = []   
        ids = []
        print('camera...')                              
        cap = cv2.VideoCapture(0)                    
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, img = cap.read()                     
            if not ret:
                print("Cannot receive frame")
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            img_np = np.array(gray,'uint8')               
            face = detector.detectMultiScale(gray)        
            for(x,y,w,h) in face:
                faces.append(img_np[y:y+h,x:x+w])         
                ids.append(3)                             
            cv2.imshow('oxxostudio', img)  
            
            if  cv2.waitKey(100) == 27:           
                break
 
        print('training...')  
        i=int()
        recognition.train(faces,np.array(ids))                            
        recognition.save(f'C:/Users/cheri/Desktop/homework/APP/face{i}.yml')                           
        print('ok!')
        i=i+1
        if i>30:
            i=0
        cv2.destroyAllWindows()
    def callback1(self,event):
        recognition = cv2.face.LBPHFaceRecognizer_create()
        
        cascade_path = 'C:/Users/cheri/Desktop/homework/APP/text.xml'  
        face_cascade = cv2.CascadeClassifier(cascade_path)        
        print('wait...')
        cap = cv2.VideoCapture(0)                                 
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
        while True:
            i=0
            recognition.read(f'C:/Users/cheri/Desktop/homework/APP/face{i}.yml')
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            img = cv2.resize(img,(540,300))             
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
            faces = face_cascade.detectMultiScale(gray) 
            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            
                idnum,confidence = recognition.predict(gray[y:y+h,x:x+w])  
                if confidence < 60:
                    text = "yes"
                                         
                else:
                    text = '???'                                          
                cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow('finish', img)
            i=i+1
            if i>30:
                i=0
            
            if cv2.waitKey(1) == 27:
                break    
        cap.release()
        cv2.destroyAllWindows()
        

        
if __name__ == "__main__":
    button().run()
