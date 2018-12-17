import os
import argparse
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from simple_logger import SimpleLogger

MFCC_LENGTH = 10
TEST_TRAIN_RATIO = 0.1

class_enum = {
    'IT': 0,
    'CH': 1
}

def get_class_from_path(path):
    base_filename = os.path.basename(path)
    class_name = base_filename.split('.')[0]
    return class_enum[class_name]

def trim_mfcc(mfcc):
    return mfcc[:,:MFCC_LENGTH]

def normalize(mfcc):
    trimmed_data = trim_mfcc(mfcc)
    mms = MinMaxScaler()
    return(mms.fit_transform(np.abs(trimmed_data)))

def train_model(train_data, train_data_labels, validation_data, validation_data_labels, batch_size=128):
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''

    sample_n, mfcc_n, frame_n = train_data.shape
    train_data = train_data.reshape(sample_n, mfcc_n, frame_n, 1)
    validation_data = validation_data.reshape(validation_data.shape[0], mfcc_n, frame_n, 1)
    input_shape = (mfcc_n, frame_n, 1)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(len(class_enum), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.05)

    # Fit model using ImageDataGenerator
    model.fit_generator(datagen.flow(train_data, train_data_labels, batch_size=batch_size),
                        steps_per_epoch=len(train_data) / 32
                        , epochs=10,
                        callbacks=[es,tb], validation_data=(validation_data,validation_data_labels))

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', help='Paths to pickle files with MFCC data. Pickle file name should be like EN.ext or CH.ext or any other class', nargs='*')
    args = parser.parse_args()

    logger = SimpleLogger(should_print=True)

    input_data = []
    for data_path in args.data_paths:
        data_class_int = get_class_from_path(data_path)
        with open(data_path, 'rb') as f:
            try:
                while True:
                    input_data.append((pickle.load(f), data_class_int))
            except EOFError:
                pass

    mfccs = []
    labels = []
    for data in input_data:
        if data[0].shape[1] < MFCC_LENGTH:
            continue
        mfccs.append(normalize(data[0]))
        labels.append(data[1])

    train_data = np.array(mfccs)
    label_data = np.array(to_categorical(labels, len(class_enum)))

    train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, label_data, test_size=TEST_TRAIN_RATIO)

    trained_model = train_model(np.array(train_data), np.array(train_labels), np.array(validation_data), np.array(validation_labels))
    trained_model.save('out/my_first_model.h5')