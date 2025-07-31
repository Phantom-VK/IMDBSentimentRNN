from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

from rnn.layers.embeddinglayer import EmbeddingLayer

if __name__ == "__main__":

    ### temp sentences
    sent = ['the glass of milk',
            'the glass of juice',
            'the cup of tea',
            'I am a good boy',
            'I am a good developer',
            'understand the meaning of words',
            'your videos are good', ]

    embedding_layer = EmbeddingLayer(sent=sent, voc_size=10000, max_sent_len=8, dim=10)
    embedded_docs = embedding_layer.get_embedded_data(padding='pre')
    print("Padded input sequences (token indices):\n", embedded_docs)

    # Add Embedding layer with input_length:
    embedding = embedding_layer.layer()

    model = Sequential()
    model.add(embedding)
    model.compile('adam', 'mse')
    model.summary()

    # Predicting embeddings for the input data
    prediction_arr = model.predict(embedded_docs)
    print("Embedding output shape:", prediction_arr.shape)

