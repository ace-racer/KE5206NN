from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load the training and testing data
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
# Display purpose:
X_train_orig = X_train
X_test_orig = X_test

img_rows, img_cols = 28, 28

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
nb_classes = 10
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# X_train = X_train.reshape(-1, img_rows, img_cols)

# Number of hidden units to use:
nb_units = 50

model = Sequential()

# Recurrent layers supported: SimpleRNN, LSTM, GRU:
# using img_row as time_steps, img_cols as inputs
model.add(SimpleRNN(nb_units,
                    input_shape=(img_rows, img_cols)))

# To stack multiple RNN layers, all RNN layers except the last one need
# to have "return_sequences=True".  An example of using two RNN layers:
# model.add(SimpleRNN(16,
#                    input_shape=(img_rows, img_cols),
#                    return_sequences=True))
# model.add(SimpleRNN(32))

model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

epochs = 20

history = model.fit(X_train,
                    Y_train,
                    epochs=epochs,
                    batch_size=128,
                    verbose=2)

plt.figure(figsize=(5, 3))
plt.plot(history.epoch, history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5, 3))
plt.plot(history.epoch, history.history['acc'])
plt.title('accuracy')

plt.show()
