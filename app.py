from rnn.etl.etl import ETL
from rnn.model import IMDBRNNMODEL
import tensorflow as tf

if __name__ == "__main__":
    # model = IMDBRNNMODEL(optimizer='adam',loss='binary_crossentropy')
    # X_train, y_train, X_test, y_test = ETL().getr_train_test_split()
    # trained_model = model.train_model(X_train,y_train)
    trained_model = tf.keras.models.load_model('trained_model/simple_rnn_imdb.keras')
    print(trained_model.summary())