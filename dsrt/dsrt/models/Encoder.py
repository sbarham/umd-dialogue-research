"""
Represents the encoder part of an encoder-decoder for dialogue modeling.
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


class Encoder:
    def __init__(self, data, config=Config()):
        self.config = config
        self.data = data
        
        self.build() # this will be a Keras model for now
        
        return
        
    def build(self):
        """
        The encoder computational graph consists of four components:
        (1) the input node                  `encoder_input`
        (2) the embedding node              `encoder_embed`
        (3) the recurrent (RNN) part        `encoder_rnn`
        (4) the hidden state output         `encoder_hidden_state`
        For convenience, we also construct the (un-compiled) Encoder training model:
        (5) uncompiled model                `encoder_training_model`
        """
        
        # Grab hyperparameters from self.config:
        hidden_dim = self.config['encoding-layer-width']
        recurrent_unit = self.config['recurrent-unit-type']
        bidirectional = False # self.config['encoding-layer-bidirectional']
        vocab_size = self.data.vocab_size
        embedding_dim = math.ceil(math.log(vocab_size, 2))    # self.config['embedding-dim']
        input_length = self.data.max_utterance_length + 1
        
        # Assemble the network components:
        encoder_input = Input(shape=(None,))
        encoder_embed = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_input) #, input_length=input_length)(encoder_input)
        # input of this Embedding() is  (None, input_length)
        # output of this Embedding() is (None, input_length, embedding_dim)
        encoder_rnn, encoder_hidden_state = None, None
        
        if recurrent_unit == 'lstm':
            encoder_rnn = LSTM(hidden_dim, return_state=True)
            encoder_output, encoder_state_h, encoder_state_c = encoder_rnn(encoder_embed)
            # discard the encoder output, keeping only the hidden state
            encoder_hidden_state = [encoder_state_h, encoder_state_c]
        if recurrent_unit == 'gru':
            encoder_rnn = GRU(hidden_dim, return_state=True)
            encoder_output, encoder_hidden_state = encoder_rnn(encoder_embed)
        else:
            raise Exception('Invalid recurrent unit type: {}'.format(recurrent_unit))
        
        # make the RNN component bidirectional, if desired
        if bidirectional:
            encoder_rnn = Bidirectional(encoder_rnn, merge_mode='ave')
        
        # save the three Enccoder components as class state
        self.encoder_input = encoder_input
        self.encoder_embed = encoder_embed
        self.encoder_rnn = encoder_rnn
        self.encoder_hidden_state = encoder_hidden_state
        
        # finally, build the training model
        self.encoder_training_model = Model(self.encoder_input, self.encoder_hidden_state)
        
        return