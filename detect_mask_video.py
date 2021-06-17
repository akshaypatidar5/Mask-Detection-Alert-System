#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from pygame import mixer
import numpy as np
import imutils
import cv2
import os
import time


# In[9]:


prototxtPath = r"face_detector/deploy.prototxt"
weightsPath  = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath,weightsPath)


# In[10]:


maskNet = load_model("mask_detector.model")


# In[11]:


mixer.init()
sound = mixer.Sound('alarm.wav')


# In[12]:


def detect_and_predict_mask(frame,facenet,maskNet):
    h= frame.shape[0]
   
    w= frame.shape[1]
   
    blob = cv2.dnn.blobFromImage(frame,1.0,(244,244),(104.0,177.0,123.0))
    facenet.setInput(blob)
    detections = facenet.forward()
    print(detections.shape[2])
    print(detections.shape)
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")
            
            (startX,startY) = (max(0,startX),max(0,startY))
            (endX,endY) = (min(w-1,endX),min(h-1,endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX,startY,endX,endY))
    if len(faces) > 0:

        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


# In[13]:


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = cv2.VideoCapture(0)
#address = "https://192.168.43.1:8080/video"
#vs.open(address)
time.sleep(2.0)
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY) = box
        (mask,withoutMask) = pred
        
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0,255,0) if label == "Mask" else (0,0,255)
        
        label = "{}: {:.2f}%".format(label,max(mask,withoutMask) * 100)
        print(label)
        
        cv2.putText(frame,label,(startX,startY-10),
                   cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        
        l=label.split(':')
        if(l[0] =='Mask'):
            print("No Beep")
        elif(l[0] =='No Mask'):
            sound.play()
            print("Beep") 
        
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
        


# In[ ]:





# In[ ]:




