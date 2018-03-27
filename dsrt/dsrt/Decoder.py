"""
Represents the decoder part of an encoder-decoder network for dialogue.
"""

# Keras packages
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, GRU, Dense, Bidirectional

# nltk
from nltk import word_tokenize

# numpy
import numpy as np

# Python stdlib
import math
import re

# Our packages
from dsrt import Config
from dsrt import DialogueCorpus


class Decoder:
    def __init__(self, data, encoder, config=Config()):
        self.config = config
        self.data = data
        self.encoder = encoder
        
        self.build() # this will be a Keras model for now
        
        return
        
    def build(self):
        """
        The decoder computational graph consists of three components:
        (1) the input node                       `decoder_input`
        (2) the embedding node                   `decoder_embed`
        (3) the recurrent (RNN) part             `decoder_rnn`
        (4) the output of the decoder RNN        `decoder_output`
        (5) the classification output layer      `decoder_dense`
        """
        
        # Grab hyperparameters from self.config:
        hidden_dim = self.config['encoding-layer-width']
        recurrent_unit = self.config['recurrent-unit-type']
        bidirectional = False #self.config['encoding-layer-bidirectional']
        vocab_size = self.data.vocab_size
        embedding_dim = math.ceil(math.log(vocab_size, 2))    # self.config['embedding-dim']
        input_length = self.data.max_utterance_length + 1
        
        # Assemble the network components:
        decoder_input = Input(shape=(None,))
        decoder_embed = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_input) #, input_length=input_length)(decoder_input)
        
        if recurrent_unit == 'lstm':
            decoder_rnn = LSTM(hidden_dim, return_sequences=True, return_state=True)
            decoder_output, decoder_h, decoder_c = decoder_rnn(decoder_embed,
                                                initial_state=self.encoder.encoder_hidden_state)
        elif recurrent_unit == 'gru':
            decoder_rnn = GRU(hidden_dim, return_sequences=True, return_state=True)
            decoder_output, _ = decoder_rnn(decoder_embed, 
                                             initial_state=self.encoder.encoder_hidden_state)
        else:
            raise Exception('Invalid recurrent unit type: {}'.format(recurrent_unit))
        
        # make the RNN component bidirectional, if desired
        if bidirectional:
            decoder_rnn = Bidirectional(decoder_rnn, merge_mode='ave')
        
        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_output = decoder_dense(decoder_output)
        
        # save the four Decoder components as class state
        self.decoder_input = decoder_input
        self.decoder_embed = decoder_embed
        self.decoder_rnn = decoder_rnn
        self.decoder_dense = decoder_dense
        self.decoder_output = decoder_output
        
        return