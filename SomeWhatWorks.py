import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
    #from google.colab import files
from keras.preprocessing import image
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

(train_images,train_labels),(test_images,test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape(len(train_images),28,28,1) # (60,000,784)
test_images = test_images.reshape(len(test_images),28,28,1)

model = keras.models.Sequential([
    keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(28,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
    ])

print(model.summary())

model.compile(optimizer='adam',metrics=['acc'],loss='sparse_categorical_crossentropy')

model.fit(train_images,train_labels,epochs=10,batch_size=32, validation_split=0.1)

model.evaluate(test_images,test_labels)

def predict_img(path):
    img = image.load_img(path)
    x = image.img_to_array(img)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)  #converting from rbg image to grayscale
    plt.imshow(x,cmap='gray')
    plt.show()
    x = cv2.resize(np.array(x), (28, 28))    #resizing it to 28x28
    x = x.reshape(28,28,1)                   #Reshaping it to fit in our model
    x = np.expand_dims(x, axis=0)
    class_label = model.predict(x)          #predicting
    print('Predicted Value is:',np.where(class_label[0]==max(class_label[0]))[0])

predict_img(r'C:\\Users\\Yash\\Desktop\\KTH\\Sem1-p2-AI\\ProjectNn\\Digits\\2.png')

