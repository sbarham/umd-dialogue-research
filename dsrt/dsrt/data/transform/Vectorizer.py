from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from numpy import argmax

from dsrt.config import DataConfig

class Vectorizer:
	def __init__(self, vocab_list, config=DataConfig()):
		self.config = config
		self.vocab_list = vocab_list
		
		# reserved vocabulary items
        self.pad_u = config['pad-u']
        self.pad_d = config['pad-d']
        self.start = config['start']
        self.stop = config['stop']
        self.unk = config['unk']
		
		# initialize the integer- and OHE-encoders
		self.init_encoders()
		
		return
    
    def init_encoders(self, config=DataConfig()):
        """
        Initialize the integer encoder and the one-hot encoder, fitting them to the vocabulary
        of the corpus.
        
        NB:
        From here on out,
            - 'ie' stands for 'integer encoded', and
            - 'ohe' stands for 'one-hot encoded'
        """
        self.log('info', 'Initializing the encoders ...')
        
        # create the integer encoder and fit it to our corpus' vocab
        self.ie = LabelEncoder()
        self.ie_vocab = self.ie.fit_transform(self.vocab_list)
        
        self.pad_u_index = self.ie.transform([self.pad_u])[0]
        
        # create the OHE encoder and fit it to our corpus' vocab
        self.ohe = OneHotEncoder(sparse=False)
        self.ohe_vocab = self.ohe.fit_transform(self.ie_vocab.reshape(len(self.ie_vocab), 1))
        
        return
        
    def transform(self, dialogues, ohe=False):
    	if not ohe:
    		return self.vectorize_dialogues(dialogues)
    	else:
    		return self.vectorize_dialogues_ohe(dialogues)
    
    
    #################################
    #     INTEGER VECTORIZATION     #
    #################################
        
    def vectorize_dialogues(self, dialogues):
        """
        Take in a list of dialogues and vectorize them all
        """
        return [self.vectorize_dialogue(d) for d in dialogues]
    
    def vectorize_dialogue(self, dialogue):
        """
        Take in a dialogue (a sequence of tokenized utterances) and transform it into a 
        sequence of sequences of indices
        """
        return [self.vectorize_utterance(u) for u in dialogue]
    
    def vectorize_utterance(self, utterance):
        """
        Take in a tokenized utterance and transform it into a sequence of indices
        """
        for i, word in enumerate(utterance):
            if not word in self.vocab_list:
                utterance[i] = '<unk>'
        
        return self.swap_pad_and_zero(self.ie.transform(utterance))
    
    def devectorize_dialogue(self, dialogue):
        """
        Take in a dialogue of ohe utterances and transform them into a tokenized dialogue
        """
        return [self.devectorize_utterance(u) for u in dialogue]
    
    def devectorize_utterance(self, utterance):
        """
        Take in a sequence of indices and transform it back into a tokenized utterance
        """
        utterance = self.swap_pad_and_zero(utterance)
        return self.ie.inverse_transform(utterance)
        
    def word_to_index(self, word):
    	return self.swap_pad_and_zero(self.ie.transform([word]))[0]
    
    
    #################################
    #       OHE VECTORIZATION       #
    #################################
    
    def vectorize_batch_ohe(self, batch):
        """
        One-hot vectorize a whole batch of dialogues
        """
        return np.array([self.vectorize_dialogue_ohe(dia) for dia in batch])
    
    def vectorize_dialogue_ohe(self, dia):
        """
        Take in a dialogue (a sequence of tokenized utterances) and transform it into a 
        sequence of sequences of one-hot vectors
        """
        # we squeeze it because it's coming out with an extra empty
        # dimension at the front of the shape: (1 x dia x utt x word)
        return np.array([[self.vectorize_utterance_ohe(utt) for utt in dia]]).squeeze()
    
    def vectorize_utterance_ohe(self, utterance):
        """
        Take in a tokenized utterance and transform it into a sequence of one-hot vectors
        """
        for i, word in enumerate(utterance):
            if not word in self.vocab_list:
                utterance[i] = '<unk>'
        
        ie_utterance = self.swap_pad_and_zero(self.ie.transform(utterance))
        ohe_utterance = np.array(self.ohe.transform(ie_utterance.reshape(len(ie_utterance), 1)))
        
        return ohe_utterance
    
    def devectorize_dialogue_ohe(self, ohe_dialogue):
        """
        Take in a dialogue of ohe utterances and transform them into a tokenized dialogue
        """
        return [self.devectorize_utterance_ohe(u) for u in ohe_dialogue]
    
    def devectorize_utterance_ohe(self, ohe_utterance):
        """
        Take in a sequence of one-hot vectors and transform it into a tokenized utterance
        """
        ie_utterance = [argmax(w) for w in ohe_utterance]
        utterance = self.ie.inverse_transform(self.swap_pad_and_zero(ie_utterance))
        
        return utterance
    
    
    ##############################
    #      IE-to-OHE Encoding    #
    ##############################
    
    def ie_to_ohe_dialogue(self, dialogue):
        return np.array([self.ie_to_ohe_utterance(u) for u in dialogue])
    
    def ie_to_ohe_utterances(self, dialogue):
        return np.array([self.ie_to_ohe_utterance(u) for u in dialogue])
    
    def ie_to_ohe_utterance(self, utterance):
        return self.ohe.transform(utterance.reshape(len(utterance), 1))
    
    
    ###################
    #     MASKING     #
    ###################
    
    def swap_pad_and_zero(self, utterance):
        for i, w in enumerate(utterance):
            if w == 0:
                utterance[i] = self.pad_u_index
            elif w == self.pad_u_index:
                utterance[i] = 0
        
        return utterance