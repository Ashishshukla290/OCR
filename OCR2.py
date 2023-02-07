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

    dic = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:"C",13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'} 
    img = Image.open(a)        
    img = img.convert('L')    
    img = img.resize([28,28])    
    img_arr = image.img_to_array(img)    
    img_arr = img_arr.reshape((1,28,28,1))    
    prediction = model.predict(img_arr)    
    prediction = np.argmax(prediction,axis = 1)
    a = int(prediction) 
    print(dic[a])


b = r'C:\Users\sujee\Downloads\data\testing_data\A/28320.png'
st.code(b)