import matplotlib.pyplot as plt
import pandas as pd

from utils.work_images import read_path

import keras
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


class Model:

    def __init__(self, shape=(32, 32, 3), num_classes=4, is_plot=True,
                 path_database='database/{}', epochs=8):
        self.opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-5)
        self.shape = shape
        self.num_classes = num_classes
        self.model = None
        self.is_plot = is_plot
        self.path_database = path_database
        self.epochs = epochs
        self.history = None

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.opt,
                           metrics=['accuracy'])

    def create_model(self):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        self.model = model

    def plot(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('utils/acc_epoch.jpg')

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('utils/loss_epoch.jpg')

    def save_model(self):
        model_json = self.model.to_json()
        with open('utils/model.json', 'w') as json_file:
            json_file.write(model_json)

        self.model.save_weights('utils/model.h5')

    def load_model(self):
        json_file = open('utils/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('utils/model.h5')

    def datagen_image(self, type):
        label_train = pd.read_csv(self.path_database.format('train.truth.csv'))
        datagen = ImageDataGenerator(validation_split=0.2)
        generator = datagen.flow_from_dataframe(dataframe=label_train,
                                                directory=self.
                                                path_database.
                                                format('train'),
                                                x_col='fn',
                                                y_col='label',
                                                class_mode='categorical',
                                                target_size=self.shape[:2],
                                                batch_size=32,
                                                subset=type,)
        return generator

    def train(self):

        train_gen = self.datagen_image('training')
        valid_gen = self.datagen_image('validation')
        STEP_SZ_TRAIN = train_gen.n//train_gen.batch_size
        STEP_SZ_VALID = valid_gen.n//valid_gen.batch_size

        self.history = self.model.fit_generator(generator=train_gen,
                                                steps_per_epoch=STEP_SZ_TRAIN,
                                                validation_data=valid_gen,
                                                validation_steps=STEP_SZ_VALID,
                                                epochs=self.epochs)
        if self.is_plot:
            self.plot()

    @property
    def predictions(self):
        x_test = read_path(self.path_database.format('test'), self.shape[:2])
        return self.model.predict_classes(x_test)
