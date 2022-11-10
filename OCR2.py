import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from PIL import Image


st.title('Welcome')

st.header('Upload Your Image')

a = st.file_uploader('Upload Here')


batch_size = 32

epochs = 50

IMG_HEIGHT = 28

IMG_WIDTH = 28

p1 = r'C:\Users\sujee\Downloads\data2/training_data'

p2 = r'C:\Users\sujee\Downloads\data2/testing_data'



augmented_image_gen = ImageDataGenerator(
        rescale = 1/255.0,
    rotation_range=2,
    width_shift_range=.1,
    height_shift_range=.1,
    zoom_range=0.1,
    shear_range=2,
    brightness_range=[0.9, 1.1],
    validation_split=0.2,
   
   )




normal_image_gen = ImageDataGenerator(
    rescale = 1/255.0,
    validation_split=0.2,
  
   )



train_data_gen = augmented_image_gen.flow_from_directory(batch_size=batch_size,
                                                     directory=p1,
                                                     color_mode="grayscale",
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode="categorical",
                                                     seed=65657867,
                                                     subset='training')




val_data_gen = normal_image_gen.flow_from_directory(batch_size=batch_size,
                                                     directory=p2,
                                                     color_mode="grayscale",
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode="categorical",
                                                     seed=65657867,
                                                     subset='validation')


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(36, activation='softmax'))




model.compile(optimizer=SGD(lr=0.01, momentum=0.9),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])



model.fit_generator(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=32,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // batch_size)






def image_to_text(a):

    img = a
    
    img = img.convert('L')
    
    img = img.resize([28,28])
    
    img_arr = image.img_to_array(img)
    
    img_arr = img_arr.reshape((1,28,28))
    
    prediction = model.predict(np.array(img_arr))
    
    prediction = np.argmax(prediction,axis = 1)
    
    print(prediction)


st.subheader("Here's Your Text ")
b = image_to_text(a)
st.code(b)