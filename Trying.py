import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing import image


(train_x, train_y) , (test_x, test_y) = mnist.load_data()

train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)

train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)

model = Sequential()
model.add(Dense(units=128,activation='relu',input_shape=(784,)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer=SGD(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights('mnist-model.h5')
#model.fit(train_x,train_y,batch_size=32,epochs=10,verbose=1)

img = image.load_img(path=r'C:\\Users\\Yash\\Desktop\\KTH\\Sem1-p2-AI\\ProjectNn\\inverted\\9.jpg',color_mode = 'grayscale',target_size=(28,28,1))
img = image.img_to_array(img)
test_img = img.reshape((1,784))

img_class = model.predict_classes(test_img)
prediction = img_class[0]
classname = img_class[0]

print('Class: ',classname)
img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()

