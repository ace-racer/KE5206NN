import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM, Bidirectional, Conv1D, concatenate, Permute, Dropout

batch_size = 32
num_classes = 10
epochs = 100

row_hidden = 128
col_hidden = 128

row, col = X_train.shape[1:]

input = Input(shape=(row, col))

def lstm_pipe(in_layer):
    x = Conv1D(row_hidden, kernel_size=3, padding = 'same')(in_layer)
    x = Conv1D(row_hidden, kernel_size=3, padding = 'same')(x)
    encoded_rows = Bidirectional(LSTM(row_hidden, return_sequences = True))(x)
    return LSTM(col_hidden)(encoded_rows)
# read it by rows
first_read = lstm_pipe(input)
# read it by columns
trans_read = lstm_pipe(Permute(dims = (1,2))(input))
encoded_columns = concatenate([first_read, trans_read])
encoded_columns = Dropout(0.2)(encoded_columns)
prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(input, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# checkpoint
outputFolder = './output'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath = outputFolder + "/output_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1,
                             save_best_only=True, save_weights_only=True,
                             mode='auto')
callbacks_list = [checkpoint]

earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=5,
                          verbose=1, mode='auto')
callbacks_list.append(earlystop)

history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs, callbacks=callbacks_list,
              verbose=2,
              validation_data=(X_test, Y_test))


scores = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Test loss: 0.3130993789434433
# Test accuracy: 0.8823