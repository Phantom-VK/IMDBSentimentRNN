from typing import List


from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import pad_sequences



class EmbeddingLayer():

    def __init__(self, sent:List[str], voc_size:int, max_sent_len:int, dim:int):
        """
        Creates an embedding layer embedded on given sentence data
        :param sent: List of sentences or text data to embedd
        :param voc_size: Max vocabulary size
        :param max_sent_len: Max padding size for one hot encoded text words
        :param dim: Dimension of feature representation array
        """
        self.sent = sent
        self.voc_size = voc_size
        self.max_sent_len = max_sent_len
        self.dim = dim


    def _get_one_hot_repr(self):
        """
        Returns one hot encoded representation of text
        :return: List[List[int]]
        """
        return [one_hot(words, self.voc_size) for words in self.sent]

    def get_embedded_data(self, padding:str):
        """
        Returns padded one hot representation of text data provided
        :return:
        """
        return pad_sequences(self._get_one_hot_repr(), padding=padding, maxlen=self.max_sent_len)

    def layer(self):
        """
        Returns trained embedding layer
        :return:
        """
        return Embedding(input_dim=self.voc_size, output_dim=self.dim, input_length=self.max_sent_len)






