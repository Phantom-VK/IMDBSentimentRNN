from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

class ETL:

    def __init__(self):
        self.voc_size = 10000
        self.padding_len = 500

    def getr_train_test_split(self):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=self.voc_size)

        X_train = sequence.pad_sequences(sequences=X_train, maxlen=self.padding_len)
        X_test = sequence.pad_sequences(sequences=X_test, maxlen=self.padding_len)


        return X_train, y_train, X_test, y_test

