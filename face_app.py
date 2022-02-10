
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image 
import requests
from skimage.transform import resize
from io import BytesIO
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('**Face Mask Recognition**')
st.text('***Upload your image here***')

@st.cache(allow_output_mutation=True)
def load_model():
  model= tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/model/best_model.h5')
  return model

model=load_model()

classes=['with_face_mask', 'without_face_mask']

uploaded_file=st.file_uploader("choose an image....", type=["jpg","png"])
if uploaded_file is not None:
  img= Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image') 

  if st.button('PREDICT'):
    classes=['with_face_mask', 'without_face_mask']
    img=img.resize((256,256))

    i= img_to_array(img)

    i=preprocess_input(i)

    input_arr=np.array([i])
    
    
    y_out=np.argmax(model.predict(input_arr))
    y_out=classes[y_out]
    
    st.write(f' I can confirm that this is ',y_out)