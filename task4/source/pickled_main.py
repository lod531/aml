from get_data import get_videos_from_folder,get_target_from_csv
import os
import numpy as np
from utils import save_solution
import pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"../train/")
test_folder = os.path.join(dir_path,"../test/")

x_train = pickle.load(open('x_train.pickle', 'rb'))
y_train = pickle.load(open('y_train.pickle', 'rb'))
x_test = pickle.load(open('x_test.pickle', 'rb'))

x_train_images = x_train[0]
x_test_images = x_test[0]
for i in range(1, len(x_train)):
    x_train_images = np.concatenate([x_train_images, x_train[i]])
for i in range(1, len(x_test)):
    x_test_images = np.concatenate([x_test_images, x_test[i]])
y_train_images = []
print(len(x_train[0]))
for i in range(0, len(y_train)):
    for j in range(0, len(x_train[i])):
        y_train_images.append(y_train[i])

print(x_train_images.shape)
x_train_images = np.expand_dims(x_train_images, 4)
print(x_train_images.shape)

#one-hot encode target column
print(y_train.shape)
y_train_images = to_categorical(y_train_images)
print(y_train[0])

#create model
model = Sequential()

#add model layers
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(100,100, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dense(1028, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.summary()

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train_images, y_train_images, epochs=5)


model.save('conv.model')
