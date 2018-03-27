"""
Represents an encoder-decoder dialogue model
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


class EncoderDecoder:
    def __init__(self, encoder, decoder, config=Config()):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        
        # build the trainin and inference models; save them
        self.build_training_model()
        self.build_inference_model()
        
        self.save_models()
        
        return
        
    def build_training_model(self):
        self.encoder_input = self.encoder.encoder_input
        self.encoder_embed = self.encoder.encoder_embed
        self.decoder_input = self.decoder.decoder_input
        self.decoder_embed = self.decoder.decoder_embed
        self.decoder_output = self.decoder.decoder_output
        
        if self.config['hierarchical']:
            # do something
            pass
        else:
            self.training_model = Model([self.encoder_input, self.decoder_input], self.decoder_output)
    
    def build_inference_model(self):
        # grab some important hyperparameters
        hidden_dim = self.config['encoding-layer-width']
        recurrent_unit = self.config['recurrent-unit-type']
        
        # build the encoder model
        self.encoder_model = Model(self.encoder.encoder_input, self.encoder.encoder_hidden_state)
    
        decoder_hidden_state_input = None
        decoder_hidden_state_output = None
        decoder_output = None
        # build the decoder model
        if recurrent_unit == 'lstm':
            decoder_hidden_state_input_h = Input(shape=(hidden_dim,))
            decoder_hidden_state_input_c = Input(shape=(hidden_dim,))
            decoder_hidden_state_input = [decoder_hidden_state_input_h, decoder_hidden_state_input_c]
            # take in the regular inputs, condition on the hidden state
            _, decoder_state_h, decoder_state_c = self.decoder.decoder_rnn(self.decoder_embed,
                                                                           initial_state=decoder_hidden_state_input)
            decoder_hidden_state_output = [decoder_state_h, decoder_state_c]
        elif recurrent_unit == 'gru':
            decoder_hidden_state_input = [Input(shape=(hidden_dim,))]
            # take in the regular inputs, condition on the hidden state
            decoder_output, hidden_state = self.decoder.decoder_rnn(self.decoder_embed,
                                                                    initial_state=decoder_hidden_state_input)
            decoder_hidden_state_output = [hidden_state]
        else:
            raise Exception('Invalid recurrent unit type: {}'.format(recurrent_unit))
            
        decoder_output = self.decoder.decoder_dense(decoder_output)
        self.decoder_model = Model([self.decoder_input] + decoder_hidden_state_input,
                                   [decoder_output] + decoder_hidden_state_output)
    
    def fit(self, data):
        # grab some hyperparameters from our config
        optimizer = self.config['optimizer']
        loss = self.config['loss']
        batch_size = self.config['batch-size']
        num_epochs = self.config['num-epochs']
        validation_split = self.config['validation-split']
        
        # grab the training and validation data
        encoder_x = data.train.encoder_x
        decoder_x = data.train.decoder_x
        decoder_y = data.train.decoder_y_ohe
        
        self.training_model.compile(optimizer=optimizer, loss=loss)
        self.training_model.fit([encoder_x, decoder_x], decoder_y,
                                batch_size=batch_size,
                                epochs=num_epochs,
                                validation_split=validation_split)
        
        self.save_models()
        
        
    def predict(self, x):
        """
        Take in an integer-vectorized (i.e., index) vector, and predict the maximally
        likely response, returning it as an integer-vectorized (i.e., index) vector.
        """
        recurrent_unit = self.config['recurrent-unit-type']
        
        # encode the input seq into a context vector
        if recurrent_unit == 'lstm':
            context_state = self.encoder_model.predict(np.array(x))
        elif recurrent_unit == 'gru':
            hidden_state = self.encoder_model.predict(np.array(x))
            context_state = [hidden_state]
        else:
            raise Exception('Invalid recurrent unit type: {}'.format(recurrent_unit))
        
        # create an empty target sequence, seeded with the start character
        y = self.data.vectorize_utterance([self.data.start])
        response = []
        
        # i = 0
        while True:
            
            # decode the current sequence + current context into a
            # conditional distribution over next token:
            output_token_probs = None
            if recurrent_unit == 'lstm':
                output_token_probs, h, c = self.decoder_model.predict([y] + context_state)
                context_state = [h, c]
            elif recurrent_unit == 'gru':
                output_token_probs, hidden_state = self.decoder_model.predict([y] + context_state)
                context_state = [hidden_state]
            else:
                raise Exception('Invalid recurrent unit type: {}'.format(recurrent_unit))
            
            # sample a token from the output distribution (currently using maximum-likelihoo -- i.e., argmax)
            sampled_token = np.argmax(output_token_probs[0, -1, :])
            
            # add the sampled token to our output string
            response += [sampled_token]
            
            # exit condition: either we've
            # - hit the max length (self.data.output_max_len), or
            # - decoded a stop token ('\n')
            if (sampled_token == self.data.ie.transform([self.data.stop]) or 
                len(response) >= self.data.max_utterance_length):
                break
                
            # update the np array (target seq)
            y = np.array([sampled_token]) # np.concatenate((y, [sampled_token]))
            
        return response
    
    def save_models(self):
        name = self.config['model-name']
        if name == None:
            name = 'model'
        
        self.training_model.save('tmp/' + name + '_train')
        self.encoder_model.save('tmp/' + name + '_inference_encoder')
        self.decoder_model.save('tmp/' + name + '_inference_decoder')
        
        return