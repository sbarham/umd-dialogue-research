# nltk
from nltk import word_tokenize

# numpy
import numpy as np

# Python stdlib
import math
import re

# our imports
from dsrt import Config

class Conversation:
    def __init__(self, encoder, decoder, data, config=Config()):
        self.config = config
        
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.history = []
        
        
    def start(self, user_utterance=None):
        self.converse(user_utterance)
        
        return
    
    def converse(self, user_utterance=None):
        if user_utterance is None:
            user_utterance = input("> ")
        
        user_utterance = user_utterance.lower()
        
        self.history.append(user_utterance)
        
        while not re.search(u'\s*(#exit)', user_utterance, re.I):
            # get network's response
            response = self.get_response(user_utterance)
            
            # add it to the history and display to the user
            self.history.append("Machine > " + response)
            print(response)
            
            # get the user's next utterance
            user_utterance = input("> ").lower()
            self.history.append("User > " + user_utterance)
            
        return
    
    def get_response(self, utterance):
        # first tokenize the utterance
        utterance = word_tokenize(utterance)
        
        # then vectorize the utterance
        utterance = self.data.vectorize_utterance(utterance)
        
        # then invoke the model's predict function
        response = self.predict(utterance)
        
        # then devectorize the model's prediction and return it as a string
        return ' '.join(self.data.devectorize_utterance(response))
    
    def predict(self, x):
        """
        Take in an integer-vectorized (i.e., index) vector, and predict the maximally
        likely response, returning it as an integer-vectorized (i.e., index) vector.
        """
        recurrent_unit = self.config['recurrent-unit-type']
        
        # encode the input seq into a context vector
        if recurrent_unit == 'lstm':
            context_state = self.encoder.predict(np.array(x))
        elif recurrent_unit == 'gru':
            hidden_state = self.encoder.predict(np.array(x))
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
                output_token_probs, h, c = self.decoder.predict([y] + context_state)
                context_state = [h, c]
            elif recurrent_unit == 'gru':
                output_token_probs, hidden_state = self.decoder.predict([y] + context_state)
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