from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, GRU, Dense
import numpy as np

class EncoderDecoder:
    def __init__(self, data, model_path='s2s.h5', latent_dim=512, batch_size=64, 
                 epochs=10, validation_split=0.2, rnn_type='lstm', optimizer='rmsprop'):
        self.data = data
        self.model_path = model_path
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.rnn_type = rnn_type
        self.optimizer = optimizer
        
        # now build the training and inference networks
        self.build_training_model()
        self.build_inference_model()
        
    def build_training_model(self):
        # the encoder
        encoder_inputs = Input(shape=(None, self.data.num_input_chars))
        encoder_rnn, encoder_hidden_state = None, None
        
        if self.rnn_type == 'lstm':
            encoder_rnn = LSTM(self.latent_dim, return_state=True)
            encoder_outputs, encoder_state_h, encoder_state_c = encoder_rnn(encoder_inputs)
            # discard the encoder output, keeping only the hidden state
            encoder_hidden_state = [encoder_state_h, encoder_state_c]
        else:
            encoder_rnn = GRU(self.latent_dim, return_state=True)
            encoder_outputs, encoder_hidden_state = encoder_rnn(encoder_inputs)
        
        # the decoder
        decoder_inputs = Input(shape=(None, self.data.num_output_chars))
        
        if self.rnn_type == 'lstm':
            decoder_rnn = LSTM(self.latent_dim, return_sequences=True, return_state=True)
            decoder_outputs, _, _ = decoder_rnn(decoder_inputs,
                                                initial_state=encoder_hidden_state)
        else:
            decoder_rnn = GRU(self.latent_dim, return_sequences=True, return_state=True)
            decoder_outputs, _ = decoder_rnn(decoder_inputs, 
                                             initial_state=encoder_hidden_state)
                
        decoder_dense = Dense(self.data.num_output_chars, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # save the network as attributes and build the training model
        self.encoder_inputs = encoder_inputs
        self.encoder_rnn = encoder_rnn
        self.encoder_hidden_state = encoder_hidden_state
        self.decoder_inputs = decoder_inputs
        self.decoder_rnn = decoder_rnn
        self.decoder_dense = decoder_dense
        self.decoder_outputs = decoder_outputs
        
        self.training_model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)


    def build_inference_model(self):
        # the inference model actually consists of two discrete sub-models --
        # the encoder ...
        self.encoder_model = Model(self.encoder_inputs, self.encoder_hidden_state)
        
        # ... and the decoder
        decoder_hidden_state_input = None
        decoder_outputs = None
        decoder_state = None
        if self.rnn_type == 'lstm':
            decoder_hidden_state_input_h = Input(shape=(self.latent_dim,))
            decoder_hidden_state_input_c = Input(shape=(self.latent_dim,))
            decoder_hidden_state_input = [decoder_hidden_state_input_h, decoder_hidden_state_input_c]
            # take in the regular inputs, condition on the hidden state
            decoder_outputs, state_h, state_c = self.decoder_rnn(self.decoder_inputs,
                                                                initial_state=decoder_hidden_state_input)
            decoder_state = [state_h, state_c]
        else:
            decoder_hidden_state_input = Input(shape=(self.latent_dim,))
            # take in the regular inputs, condition on the hidden state
            decoder_outputs, hidden_state = self.decoder_rnn(self.decoder_inputs,
                                                            initial_state=decoder_hidden_state_input)
            decoder_state = hidden_state
            
        # run it through a dense softmax layer
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = Model([self.decoder_inputs] + decoder_hidden_state_input,
                                   [decoder_outputs] + decoder_state)


    def fit(self):
        self.training_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
        self.training_model.fit([self.data.encoder_x, self.data.decoder_x], self.data.decoder_y,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                validation_split=self.validation_split)

        
    def predict(self, input_seq):
        return self.decode_sequence(input_seq)


    def translate(self, input_str):
        # transform input_str into a numpy array
        input_seq = np.zeros((1, self.data.input_max_len, self.data.num_input_chars))
        for i, char in enumerate(input_str):
            input_seq[1, i, self.data.char2index(char, 'input')] = 1
        
        # predict the translation using the inference model
        return self.predict(input_seq)
    
    
    def decode_sequence(self, input_seq):
        # encode the input seq into a context vector
        context_state = self.encoder_model.predict(input_seq)
        
        # create an empty target sequence, seeded with the start character
        target_seq = np.zeros((1, 1, self.data.num_output_chars))
        target_seq[0, 0, char2index('\t')] = 1.
        
        output_str = ''
        while True:
            # decode the current sequence + current context into a
            # conditional distribution over next token:
            output_token_probs = None
            if self.rnn_type == 'lstm':
                output_token_probs, h, c = self.decoder_model.predict([target_seq] + context_state)
                context_state = [h, c]
            else:
                output_token_probs, context_state = \
                    self.decoder_model.predict([target_seq] + context_state)
            
            # sample a token from the output distribution
            sampled_token_index = np.argmax(output_token_probs[0, -1, :])
            sampled_char = self.data.index2char(sampled_token_index)
            
            # add the sampled token to our output string
            output_str += sampled_char
            
            # exit condition: either we've
            # - hit the max length (self.data.output_max_len), or
            # - decoded a stop token ('\n')
            if (sampled_char == '\n' or
                len(output_str) >= self.data.output_max_len):
                break
                
            # update the np array (target seq)
            target_seq = np.zeros((1, 1, self.data.num_output_chars))
            target_seq[0, 0, sampled_token_index] = 1.
            
        return output_str
            

    def save(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        
        self.training_model.save(model_path)