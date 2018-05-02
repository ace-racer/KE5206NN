from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD
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

# Number of hidden units to use:
nb_units = 50

model = Sequential()

# Recurrent layers supported: SimpleRNN, LSTM, GRU:
# using img_row as time_steps, img_cols as inputs
# model.add(SimpleRNN(nb_units,
#                     input_shape=(img_rows, img_cols)))
# model.add(LSTM(nb_units,
#                input_shape=(img_rows, img_cols))) # acc 88.5

# Stack multiple RNN layers

model.add(LSTM(nb_units, input_shape=(img_rows, img_cols), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print(model.summary())

# checkpoint
outputFolder = './output'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath = outputFolder + "/output_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=True,
                             mode='auto')
callbacks_list = [checkpoint]

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,
                          verbose=1, mode='auto')
callbacks_list.append(earlystop)

start = time.time()
history = model.fit(X_train,
                    Y_train,
                    epochs=1000, callbacks=callbacks_list,
                    batch_size=128,
                    verbose=2, validation_data=(X_test, Y_test))
end = time.time()

# plt.figure(figsize=(5, 3))
# plt.plot(history.epoch, history.history['loss'])
# plt.title('loss')
#
# plt.figure(figsize=(5, 3))
# plt.plot(history.epoch, history.history['acc'])
# plt.title('accuracy')

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.show()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Training', 'Validation'], loc='lower right')

plt.show()

# Evaluating the model on the test data
score, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score)
print('Test accuracy:', accuracy)

# Stacked LSTM
# Test score: 0.49652242830842735
# Test accuracy: 0.8836

# GRU
# Test score: 0.48214196188151837
# Test accuracy: 0.8882

# GRU with dropout 0.2 after 1st GRU layer
# Test score: 0.3564792712450027
# Test accuracy: 0.8943

# GRU with dropout 0.2 after 2nd GRU layer
# Test score: 0.3871491623699665
# Test accuracy: 0.894

# LSTM with dropout 0.2 after 1st GRU layer
# Test score: 0.3853205562978983
# Test accuracy: 0.8906
