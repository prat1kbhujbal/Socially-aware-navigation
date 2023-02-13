from json import load
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize
from tqdm.auto import trange
from tensorflow import keras
from keras import layers


class Model:
    def __init__(self):

        self.data_dir = Path(
            __file__).parent.parent / "pose_data/images/single"
        self.height = 180
        self.width = 320

    def load_images(self):

        x = np.zeros((len(y), self.height, self.width, 3), dtype=np.float32)

        for i in trange(0, len(y)):
            img = cv2.imread(f"{self.data_dir}/{i}.jpg")
            height, width, depth = img.shape
            img = resize(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                (self.height, self.width))
            x[i] = img
            y[i] = [[xy[0] / width, xy[1] / height] for xy in y[i]]

        y = np.array([np.array(a).flatten() for a in y])
        return x, y

    def show_image(self, img, points, prediction=None):
        fig, ax = plt.subplots()
        ax.imshow(img)

        x_coords = []
        y_coords = []
        for idx, p in enumerate(points):
            if idx % 2 == 0:
                x_coords.append(p * self.width)
            else:
                y_coords.append(p * self.width)
        plt.plot(x_coords, y_coords, "go")

        if prediction is not None:
            x_coords = []
            y_coords = []
            for idx in range(prediction.shape[0]):
                if idx % 2 == 0:
                    x_coords.append(prediction[idx] * self.width)
                else:
                    y_coords.append(prediction[idx] * self.height)
            plt.plot(x_coords, y_coords, "ro")

        plt.show()

    def output(self, x, y):
        inputs = layers.Input(shape=(self.height, self.width, 3))

        root = layers.Conv2D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="swish")(inputs)
        root = layers.BatchNormalization()(root)
        root = layers.Conv2D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="swish")(root)
        root = layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(root)

        root = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="swish")(root)
        root = layers.BatchNormalization()(root)
        root = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="swish")(root)
        root = layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(root)

        root = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="swish")(root)
        root = layers.BatchNormalization()(root)
        root = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="swish")(root)
        root = layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(root)

        root = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="swish")(root)
        root = layers.BatchNormalization()(root)
        root = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="swish")(root)
        root = layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(root)

        root = layers.Flatten()(root)

        knees = layers.Dense(2048, activation="swish")(root)
        knees = layers.Dense(1024, activation="swish")(knees)
        knees = layers.Dense(4, activation="sigmoid")(knees)

        head = layers.Dense(1024, activation="swish")(root)
        head = layers.Dense(512, activation="swish")(head)
        head = layers.Dense(2, activation="sigmoid")(head)

        shoulders = layers.Dense(2048, activation="swish")(root)
        shoulders = layers.Dense(1024, activation="swish")(shoulders)
        shoulders = layers.Dense(4, activation="sigmoid")(shoulders)

        outputs = layers.Concatenate()([knees, head, shoulders])

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.summary()
        model.save('./Model/Model.h5')

        test_split = 0.25
        idx = int(np.floor(test_split * len(x)))

        X_test = x[:idx]
        X_train = x[idx:]
        y_test = y[:idx]
        y_train = y[idx:]

        print(X_train.shape)
        print(y_train.shape)

        print(X_test.shape)
        print(y_test.shape)

        model.compile(
            optimizer="adam",
            loss=keras.losses.MeanSquaredError(),
            metrics=["acc"])
        model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=2)

        # Test
        test_scores = model.evaluate(X_test, y_test)
        print(f"Test loss: {test_scores[0]}")
        print(f"Test accuracy: {test_scores[1]}")


        # Visualize a test point
        idx = np.random.randint(0, len(X_test))
        img = X_test[idx]
        prediction = model(np.expand_dims(img, axis=0))
        self.show_image(img, y_test[idx], prediction.numpy().flatten())
