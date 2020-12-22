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

def first_UD (img):
    for x in range(len(img)):
        for y in range(len(img[x])):
            if 112 <img[x][y]:
                return x

def last_UD(img, x):
    for a in range(len(img)-x):
        c=0
        for b in range(len(img[x])):
            if 112 < img[x+a][b]:
                a=a+1
                c=1
        if c == 0:
            return a+x

    return a+x

def move_UD(img ,x ,a):
    from_bottom_edge= 28-a
    from_uper_edge=x

    if from_bottom_edge==from_uper_edge:
        return img

    for c in range(x,a):
        for d in range(len(img)):
            temp_img[c-x][d]=img[c][d]
            img[c][d]=0

    if from_bottom_edge<from_uper_edge:
        differance = from_uper_edge-from_bottom_edge
        for c in range(x-differance,a-differance):
            for d in range(len(img)):
                img[c][d]=temp_img[c-x][d]

    else:
        differance = from_bottom_edge-from_uper_edge
        for c in range(x+differance,a+differance):
            for d in range(len(img)):
                img[c][d]=temp_img[c-x][d]
    return img


def find_L (img):
    for y in range(len(img)):
        for x in range(len(img[x])):
            if 112 <img[x][y]:
                return x

def find_R(img, x):
    for b in range(len(img)):
        c=0
        for a in range(len(img[b])-x):
            if 112 < img[x+a][b]:
                b=b+1
                c=1
        if c == 0:
            return a+x

    return a+x




test_img = img.reshape((1,784))

img_class = model.predict_classes(test_img)
prediction = img_class[0]
classname = img_class[0]

print('Class: ',classname)
img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()

