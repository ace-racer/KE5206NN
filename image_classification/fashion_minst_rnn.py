
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time, os

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

# TODO explore GRU
# TODO try multiple layers

# Recurrent layers supported: SimpleRNN, LSTM, GRU:
# using img_row as time_steps, img_cols as inputs
# model.add(SimpleRNN(nb_units,
#                     input_shape=(img_rows, img_cols)))
# model.add(LSTM(nb_units,
#                input_shape=(img_rows, img_cols))) # acc 88.5

# Stack multiple RNN layers
model.add(LSTM(nb_units, input_shape=(img_rows, img_cols), return_sequences=True))
model.add(LSTM(32))

model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

# checkpoint
outputFolder = './output'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/output_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1,
                             save_best_only=True, save_weights_only=True,
                             mode='auto')
callbacks_list = [checkpoint]

earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=5,
                          verbose=1, mode='auto')
callbacks_list.append(earlystop)

start = time.time()
history = model.fit(X_train,
                    Y_train,
                    epochs=1000, callbacks=callbacks_list,
                    batch_size=128,
                    verbose=2)
end = time.time()

plt.figure(figsize=(5, 3))
plt.plot(history.epoch, history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5, 3))
plt.plot(history.epoch, history.history['acc'])
plt.title('accuracy')

plt.show()

# Evaluating the model on the test data
score, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score)
print('Test accuracy:', accuracy)
