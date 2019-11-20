# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:03:19 2019

@author: luiss
"""

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
from PyQt5.QtCore import QThread

from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import joblib


def ExtraerRostro(bbox,imgsml):
    x1, y1 = abs(bbox[0]), abs(bbox[1])
    x2, y2 = x1 + bbox[2] ,y1 + bbox[3]
    cara=imgsml[y1:y2 , x1:x2]
    #Image preprocessing
    rostromono=Image.fromarray(cara)
    rostrores=rostromono.resize((160,160))
    rostro_array=np.asarray(rostrores)
    return rostro_array

def SacarVector(model,cara):
    #Preparar imagen para el modelo
    #Escalar pixeles
    cara = cara.astype('float32')
    #Estandarizar los valores en los canales
    m, std = cara.mean(),cara.std()
    cara=(cara - m )/std
    #Convetir a una muestra
    muestra = np.expand_dims(cara, axis=0)
    #Vector de características
    res= model.predict(muestra)
    #Normalizar vector
    res=Normalizer(norm='l2').transform(res)
    return res[0]
    
    
class FaceDetect(QWidget):
     def __init__(self,wf):
        super().__init__()
        self.wf = wf
        self.saveF = 0
        self.an = 0
        self.iden = 0
        self.texto1 = QLabel()
        self.texto2 = QLabel()
        self.texto3 = QLabel()
        self.textbox = QLineEdit()
        self.bandera = 0
        
                
     def charge(self):
        self.bandera=1
    
     def saveFace(self):
        self.texto1.setText("Saving Face...")
        self.wf.layout().addWidget(self.texto1)
        btnSaveFace.setEnabled(False)
        self.saveF = 1
    
     def Analizar(self):
        self.wf.layout().addWidget(self.texto1)
        #btnAnalizar.setEnabled(False)
        self.an = 1
        
     def Identify(self):

        
        self.wf.layout().addWidget(self.texto1)
        #btnIdentify.setEnabled(False)
        self.iden = 1
            
        
     def run(self):
        modelos2={}
        model = load_model('facenet_keras.h5')
        w=640
        h=480
        color=(0,0,0)
        analizando=False
        identificando=False
        nombre=''
        texto=''
        cam = cv2.VideoCapture(0)
        mirror = True
        detector = MTCNN()
        #imgVideo = QLabel()
        self.texto3.setText("Escribe el nombre de la persona a procesar:")
        self.wf.layout().addWidget(self.texto3)
        self.wf.layout().addWidget(self.textbox)
        while True:
            texto1 = QLabel()
            #imgVideo = QLabel()
            ret_val, img = cam.read()
            if mirror:
                img = cv2.flip(img, 1)
            imgsml = cv2.resize(img, (w, h))
            #TODO: Hacer que solo detecte una cara
            faces = detector.detect_faces(imgsml) 
            if len(faces) > 0:
                for k in range(len(faces)): 
                    bbox = faces[k]['box']
                    keypoints = faces[k]['keypoints']
                    imgsml = cv2.rectangle(imgsml, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0))
                    cv2.circle(imgsml,(keypoints['left_eye']), 2, (0,155,255), 2)
                    cv2.circle(imgsml,(keypoints['right_eye']), 2, (0,155,255), 2)
                    cv2.circle(imgsml,(keypoints['nose']), 2, (0,155,255), 2)
                    cv2.circle(imgsml,(keypoints['mouth_left']), 2, (0,155,255), 2)
                    cv2.circle(imgsml,(keypoints['mouth_right']), 2, (0,155,255), 2)
                    cv2.putText(imgsml, texto, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    
            cv2.circle(imgsml,(w-70,h-70), 10, color, -1)     
            #TODO: Checar si no hay un bug (valores negativos en el bounding box)
            
                                      
           # cv2.imshow('Webcam View', imgsml)
            #texto1.setText("prueba")
            img = imgsml#Image.fromarray(imgsml)
            img.save('my.png')
            im = QPixmap("my.png")
                
            imgVideo.setPixmap(im)
            self.wf.layout().addWidget(imgVideo)
            
            
            
            k=cv2.waitKey(1)
            
            if k == 27:
                break  # esc to exit
     
    # =============================================================================
    #         #Guardar un rostro
    # =============================================================================
            if self.saveF == 1: #Espacio
                rostro=list()
                modelo=list()
                if len(faces)==1: #Solo un rostro puede ser extraído a la vez
                    print('Guardando rostro')
#                    if len(nombre)==0:
                    nombre= self.textbox.text()
                            
                    #------------Extraer rostro            
                    rostro_out=ExtraerRostro(bbox,imgsml)
                    #------------Agregar
                    rostro.append(rostro_out)
                    np.asarray(rostro)
                    #-----------Vector
                    modelo=SacarVector(model,rostro[0])
                    #modelo.append(res)
                    
                    if nombre in modelos2:
                        modelos2[nombre].append(modelo)
                        
                    else:
                        modelos2[nombre]=[modelo]
                            
                    print('Rostro guardado de ', nombre) 
                    #agregar =input('Quieres agregar más imágenes? s para si \t')
                    #if agregar != 's':
                     #   nombre=''
                rostro.clear()
                #modelo.clear()
                self.texto1.setText("Saved " + self.textbox.text() + " Face.")
                self.wf.layout().addWidget(self.texto1)
                btnSaveFace.setEnabled(True)
                self.saveF = 0
                k=-1
                continue
    # =============================================================================
    #         #Analizar rostro
    # =============================================================================
            if self.an == 1:#Tab
                if not analizando:
                    analizando=True
                    self.texto1.setText("Analyzing Face...")
                else:
                    analizando=False
                    color=(0,0,0)
                    self.texto1.setText("")
                    self.texto2.setText(" ")
                self.an = 0
                k=-1
            
            if analizando:
                if len(modelos2)>0:  
                    if len(faces)==1: #Solo un rostro puede ser extraído a la vez
                        #------------Extraer rostro
                        rostro_out=ExtraerRostro(bbox,imgsml)
                        #-----------Vector 
                        res=SacarVector(model,rostro_out)
                        for nom in modelos2:
                            for modelo in modelos2[nom]:
                                dist = np.linalg.norm(res-modelo)
                                print(dist)
                                if dist<=0.5:
                                    print('Rostro de ',nom)
                                    self.texto2.setText("Rostro de : " + nom)
                                    self.wf.layout().addWidget(self.texto2)   
                                    color=(0,255,0)
                                else:
                                    color=(0,0,0)
                                    self.texto2.setText(" ")
                #self.texto1.setText("Analyzed " + self.textbox.text() + " Face.")
                self.wf.layout().addWidget(self.texto1)
                #btnAnalizar.setEnabled(True)
                #self.an
    # =============================================================================
    # Identificar un rostro                  SVC
    # =============================================================================
            if (self.iden == 1 or self.bandera == 1): #+
                if not identificando:
                    self.texto1.setText("Identifying Face...")
                    trainX=np.empty((len(modelos2),128))
                    trainY=np.empty((len(modelos2),))   
                    if len(modelos2)>1: #Debe haber mínimo 2 clases 
                        if len(faces)==1: #Solo un rostro puede ser extraído a la vez
                            identificando=True
                            if self.bandera == 1:
                                model2 = joblib.load('model2.pkl')
                            else:
                                for nom in modelos2:
                                        for modelo in modelos2[nom]:
                                            temp=np.append(trainX,[modelo.astype('float32')],axis=0)
                                            trainX=np.copy(temp)
                                            temp2=np.append(trainY,[nom],axis=0)
                                            trainY=np.copy(temp2)  
                                model2 = SVC(kernel='linear', probability=True)
                                model2.fit(trainX, trainY)
                                joblib.dump(model2,'model2.pkl',compress = 9)
                else:
                    self.texto1.setText("")
                    identificando=False
                    texto=''
                self.iden = 0
                self.bandera=0
                k=-1
            
            if identificando:
                rostro_out=ExtraerRostro(bbox,imgsml)
                res=SacarVector(model,rostro_out)
                pred=model2.predict([res])
                print(pred)
                #self.texto2.setText("Creo que eres: " + pred)
                self.texto1.setText("Identified " + self.textbox.text() + " Face.")
                self.wf.layout().addWidget(self.texto1)
                texto= ''.join(pred)
                #btnIdentify.setEnabled(True)
                #self.iden = 1
                        
              
        
        cv2.destroyAllWindows()

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    wf = QWidget()
    wf.resize(400,300)
    wf.setWindowTitle("Reconocimiento Facial")
    
    
    txtDetection = QLabel()
    imgVideo = QLabel()
    
    aFaceDetect = FaceDetect(wf)
    
    
    btnSaveFace = QPushButton("Save Face")
    btnAnalizar = QPushButton("Measure euclidean distance")
    btnIdentify = QPushButton("Analize using SVC")
    btnOpenVideoCamera = QPushButton("Open Face Video Camera")
    btnCharge = QPushButton("Charge Pretrained Model")
    
    #im = QPixmap("my.png")
    #imgVideo.setPixmap(im)

    wf.setLayout(QVBoxLayout())
    wf.layout().addWidget(btnOpenVideoCamera)
    wf.layout().addWidget(btnSaveFace)
    wf.layout().addWidget(btnAnalizar)
    wf.layout().addWidget(btnIdentify)
    wf.layout().addWidget(btnCharge)
    wf.layout().addWidget(aFaceDetect)
    wf.layout().addWidget(imgVideo)

    

    #aFaceDetect.run()    
    btnOpenVideoCamera.clicked.connect(lambda: aFaceDetect.run())
    btnAnalizar.clicked.connect(lambda: aFaceDetect.Analizar())
    btnIdentify.clicked.connect(lambda: aFaceDetect.Identify())
    btnSaveFace.clicked.connect(lambda: aFaceDetect.saveFace())
    btnCharge.clicked.connect(lambda: aFaceDetect.charge())
    
    wf.show()
    sys.exit(app.exec_())