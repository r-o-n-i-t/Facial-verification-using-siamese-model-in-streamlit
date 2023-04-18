import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from verify import verify, check_path
import os


#Input image path
APP_INP = './data/app_data/input_image'
check_path(APP_INP)

#Loading our pre build model
siamese_model = load_model('siamesemodel.h5')

def main():

    st.set_page_config(layout="centered") #Setting the page configuration
    st.title(" Welcome to the Facial Verification App!!!")

    st.markdown("<h1 style='text-align: center;'>Verify your self!!!</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col3:
        st.image("""https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Face_ID_logo.svg/1200px-Face_ID_logo.svg.png""", width = 50)
   
    
    # Capture input image from our webcam as an input
    img_file_buffer = st.camera_input("")

    # Saving the input image to the required path
    SAVE_PATH = os.path.join(APP_INP, 'input_image.jpg')

    if st.button('Verify'):
        if img_file_buffer:
            with open (SAVE_PATH,'wb') as file:
                file.write(img_file_buffer.getbuffer())

        result = verify(siamese_model, APP_INP)     # Our input image is passed through the model we just loaded.
            
        if result:
            st.success("Verified!!!" + "\U00002705")    # Verified!!!✅
        else:
            st.error("Unverified!!!" + "\U0001F6A8")    # Unverified!!!🚨


if __name__ == "__main__":
    main()
