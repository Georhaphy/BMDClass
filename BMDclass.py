# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:12:28 2024

@author: user
"""

import streamlit as st 
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model


model = load_model('best_model3.h5' ,compile = False)

def Bone(a):
    if a > 0.5:
        return 'Abnormal'
    else:
        return 'Normal'

st.title("Predict Osteoporosis")
img_file = st.file_uploader("เปิดไฟล์ภาพ")


col1, col2 = st.columns([1,1]) 

if img_file is not None:
   
    im = Image.open(img_file)

    st.image(img_file,channels="BGR")
    
    img= np.asarray(im).astype(np.float32) /255.0 
    image= cv2.resize(img,(128, 128))
    X_submission = np.array(image)
    y = np.expand_dims(X_submission, 0)
    
    result = model.predict(y)
    
    
    
    
    with col1:
        st.write("Predict Osteoporosis" )
    with col2:
        st.code(f"""{Bone(result[0][0])} """) 



