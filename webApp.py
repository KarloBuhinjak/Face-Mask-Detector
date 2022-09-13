import numpy as np
import tensorflow as tf
from astropy.samp import hub
from matplotlib import pyplot as plt
import streamlit as st
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.utils import img_to_array
from keras import preprocessing
from keras.preprocessing import image
from PIL import Image



st.header("Mask detector")

def main():
    file_uploaded=st.file_uploader("Upload Image", type=['jpg'])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        figure=plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result=predict_class(image)

        if result:
            st.write("A person is not wearing a mask")
        else:
            st.write("A person wears a mask")


        st.pyplot(figure)

def predict_class(image):
    mymodel = load_model(r'C:\Users\bu\OneDrive\Radna površina\test2\mymodel.h5')
    test_image=tf.image.resize(image,[150,150])
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = mymodel.predict(test_image)

    return prediction[0][0]

main();