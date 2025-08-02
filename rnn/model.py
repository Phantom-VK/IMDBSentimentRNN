from keras.src.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential


class IMDBRNNMODEL:

    def __init__(self, optimizer: str, loss: str):
        self.model = Sequential()
        self.model.add(Embedding(10000, 128))
        self.model.add(SimpleRNN(128, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def get_model(self):
        return self.model()

    def train_model(self, X_train, y_train, epochs: int, batch_size: int, validation_split: float):
        """
        Compiles and trains model. Saves a copy to local.
        :param X_train:
        :param y_train:
        :param epochs:
        :param batch_size:
        :param validation_split:
        :return: Trained model instance
        """
        earlystopping = EarlyStopping(patience=7, restore_best_weights=True)
        history = self.model.fit(
            X_train,
            y_train,
            epochs=10,
            validation_split=0.2,
            callbacks=[earlystopping]
        )
        self.model.save('trained_model/simple_rnn_imdb.keras')
        return self.model
