"""
Ideas:
	(1) move data transformers to a sub-package
	(2) build corpus such that it conditionally builds and passes dialogues
	    through the transformers according to properties in DataConfig
"""

# external imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from numpy import argmax

import logging
import re
import copy

# our own imports
from dsrt import Config


class SampleSet:
    def __init__(self, dialogues, properties, config=DataConfig()):
        self.dialogues = dialogues
        self.length = len(dialogues)
        self.config = config
        self.properties = properties
        
        # initialize the logger
        self.init_logger()
        
        # initialize the one-hot encoder
        
        # these are set in encoder_decoder_split()
        self.encoder_x = []
        self.decoder_x = []
        self.decoder_y = []
        self.decoder_y_ohe = []
        
        # make the encoder/decoder split:
        self.encoder_decoder_split()
        
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
    
    def encoder_decoder_split(self):
        """
        For now, this assumes a flat (non-hierarchical) model, and therefore
        assumes that dialogues are simply adjacency pairs.
        """
        start = self.vectorizer.word_to_index(self.config['start'])
        stop = self.vectorizer.word_to_index(self.config['stop'])
        pad = 0
        
        self.encoder_x = np.copy(self.dialogues[:][0])
        self.decoder_x = np.zeros(self.decoder_x.shape)
        self.decoder_y = np.copy(self.dialogues[:][1])
        
        # prepare decoder_x (prefix the <start> symbol to every second-pair part)
        self.decoder_x[:][0] = start
        for i in range(len(self.decoder_y)):
			if self.decoder_y[i] == pad:
				self.decoder_y[i] = stop
        		break
        	
        	self.decoder_x[i + 1] = self.decoder_y[i]
        	
        # prepare decoder_y_ohe
        self.decoder_y_ohe = self.vectorizer.ie_to_ohe_utterances(self.decoder_y)
    
    def log(self, priority, msg):
        """
        Just a wrapper, for convenience.
        NB1: priority may be set to one of:
        - CRITICAL     [50]
        - ERROR        [40]
        - WARNING      [30]
        - INFO         [20]
        - DEBUG        [10]
        - NOTSET       [0]
        Anything else defaults to [20]
        NB2: the levelmap is a defaultdict stored in Config; it maps priority
             strings onto integers
        """
        # self.logger.log(self.config.levelmap[priority], msg)
        self.logger.log(logging.CRITICAL, msg)


    

class Corpus:
    def __init__(self, path=None, config=Config()):
        # load configuration and init path to corpus
        self.config = config
        self.path_to_corpus = path if not path is None else self.config['path-to-corpus']
        
        # init logger
        self.init_logger()
        
        self.log('info', 'Logger initialized')
        self.log('info', 'Configuration loaded')
        self.log('warn', 'Preparing to process the dialogue corpus ...')
        
        self.corpus_loaded = False
        
        # load and tokenize the dataset
        self.dialogues = self.load_corpus(self.path_to_corpus) # <-- tokenization happens here
        self.word_list, self.word_set = self.load_vocab(self.dialogues)
        
        # gather dataset properties
        self.properties = Properties(self.dialogues, self.config)
        
        # create the data transformers
        self.filter = Filter(self.properties, self.config)
        self.padder = Padder(self.properties, self.config)
        self.pairer = AdjacencyPairer(self.properties, self.config)
        self.vectorizer = Vectorizer(self.word_list, self.config)
        
        transformers = [self.filter, self.padder, self.pairer, self.vectorizer]
        
        # transform the data
        self.dialogues = self.transform(self.dialogues, transformers)
        
        # split the data
        self.train, self.test = self.split_corpus(self.dialogues)
        
        # report success!
        self.log('info', 'Corpus succesfully loaded! Ready for training.')
    
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
        
        return
    
    def load_corpus(self, path):
        self.log('info', 'Loading the dataset ...')
        
        dialogues = []
        
        if not self.corpus_loaded:
            with open(path, 'r') as f:
                dialogues = list(f)
                
        # if desired, retain only a subset of the dialogues:
        if self.config['restrict-sample-size']:
            dialogues = np.random.choice(dialogues, self.config['sample-size'])
        
        return dialogues
        
    def reload_corpus(self):
        """
        Reload the dialogue corpus, in case some changes have been made to it since we last
        loaded.
        """
        self.corpus_loaded = False
        self.load_corpus()
        
        return
    
    def load_vocab(self):
        self.log('info', 'Initializing vocabulary ...')
        
        reserved_words = set([self.pad_u, self.pad_d, self.start, self.stop, self.unk])
        corpus_words = set([w for d in self.dialogues for u in d for w in u])
        
        vocab_set = set.union(reserved_words, corpus_words)
        vocab_list = list(self.vocab_set)
        
        self.vocab_size = len(self.vocab_list)
        
        return vocab_list, vocab_set
    
    def transform(self, dialogues, transformers):
    	for t in transformers:
    		dialogues = t.transform(dialogues)
    	
    	return dialogues
    
    def train_test_split(self, dialogues):
        self.log('info', 'Splitting the corpus into train/test subsets ...')
        
        # grab some hyperparameters from our config
        split = self.config['train-test-split']
        rand_state = self.config['random-state']
        
        # split the corpus into train and test samples
        train, test = train_test_split(dialogues, train_size=split, random_state=rand_state)
        
        train = DialogueSampleSet(train, self.properties)
        test = DialogueSampleSet(test, self.properties)
        
        return train, test
    
    
    ###################
    #    UTILITIES    #
    ###################
    
    def log(self, priority, msg):
        """
        Just a wrapper, for convenience.
        NB1: priority may be set to one of:
        - CRITICAL     [50]
        - ERROR        [40]
        - WARNING      [30]
        - INFO         [20]
        - DEBUG        [10]
        - NOTSET       [0]
        Anything else defaults to [20]
        NB2: the levelmap is a defaultdict stored in Config; it maps priority
             strings onto integers
        """
        # self.logger.log(self.config.levelmap[priority], msg)
        self.logger.log(logging.CRITICAL, msg)
        
        return
    
    def pretty_print_dialogue(self, dia):
        for utt in dia:
            if utt[0] == self.pad_d:
                break
            print(self.stringify_utterance(utt))
                
        return
                      
    def stringify_utterance(self, utt):
        return ' '.join([w for w in utt if not w == self.pad_u])
    


class Properties(dict):
	def __init__(self, dialogues, config=DataConfig())
		self.config = config
		self.analyze(dialogues)
	
	def analyze(self, dialogues):
        dialogue_lengths = []
        utterance_lengths = []
        
        for dialogue in dialogues:
            # get dialogue length
            dialogue_lengths += [len(dialogue)]

            # get constituent utterances lengths
            utterance_lengths_flat += [len(u) for u in dialogue]
        
        self['max_dialogue_length'] = max(dialogue_lengths)
        self['max_utterance_length'] = max(utterance_lengths_flat)
        self['num-dialogues'] = len(dialogue_lengths)
        self['num-utterances'] = len(utterance_lengths)
        
        return


#############################
#    DATA TRANSFORMATIONS   #
#############################

class AdjacencyPairer:
	def __init__(self, properties, config=DataConfig()):
		self.properties = properties
		self.config = config
		
	def transform(self, dialogues):
		return self.dialogues_to_adjacency_pairs(dialogues)
		
	def dialogues_to_adjacency_pairs(self, dialogues):
        return [ap for d in dialogues for ap in self.dialogue_to_adjacency_pairs(d)]

    def dialogue_to_adjacency_pairs(self, dialogue):
        adjacency_pairs = []
        for i in range(len(dialogue)):
            if i + 1 < len(dialogue):
                adjacency_pairs += [[dialogue[i], dialogue[i + 1]]]
        
        return adjacency_pairs


class Filter:
	def __init__(self, properties, config=DataConfig()):
		self.properties = properties
		self.config = config
		
	def transform(self, dialogues):
		return self.filter_dialogues_by_length(dialogues)
	
    def filter_dialogues_by_length(self, dialogues):
        self.log('info', 'Filtering out long samples ...')
        
        # for filtering out long dialogues and utterances, we'll need these settings:
        max_dl = self.config['max-dialogue-length']
        use_max_dl = not (self.config['use-corpus-max-dialogue-length'])
        max_ul = self.config['max-utterance-length']
        use_max_ul = not (self.config['use-corpus-max-utterance-length'])
        
        filtered_dialogues = []
        
        # if we're putting a limit on dialogue length, 
        # iterate through the dialogues ...
        if use_max_dl:
            for dialogue in dialogues:
                # skip it if we're filtering out long dialogues and this one is too long
                if len(dialogue) > max_dl:
                    continue

                # if we're putting a limit on utterance length, 
                # iterate through utterances in this dialogue ...
                keep_dia = True
                if use_max_ul:
                    for utterance in dialogue:
                        # if an utterance is too long, mark this dialogue for exclusion
                        if len(utterance) > max_ul:
                            keep_dia = False
                            break
                            
                if keep_dia:
                    filtered_dialogues += [dialogue]
        
        return filtered_dialogues


class Tokenizer:
	def __init__(self, properties, config=DataConfig()):
		self.properties = properties
		self.config = config
	
	def transform(self, dialogues):
		return self.tokenize_dialogues(dialogues)
        
    def tokenize_dialogues(self, dialogues):
        return [self.tokenize_dialogue(d) for d in dialogues]

    def tokenize_dialogue(self, dialogue):
        utterances = dialogue.split('\t')[:-1]
        return [self.tokenize_utterance(u) for u in utterances]
    
    def tokenize_utterance(self, utterance):
        return utterance.split(' ')
        


class Padder:
	def __init__(self, properties, config=DataConfig()):
		self.properties = properties
		self.config = config
		
	def transform(self, dialogues):
		return self.pad_dialogues(dialogues)
		
	def pad_dialogues(self, dialogues):
        """
        Pad the entire dataset.
        This involves adding padding at the end of each sentence, and in the case of
        a hierarchical model, it also involves adding padding at the end of each dialogue,
        so that every training sample (dialogue) has the same dimension.
        """
        self.log('info', 'Padding the dialogues ...')
        
        empty_turn = [self.config['pad-d']] * (self.properties['max-utterance-length'] + 1)
        
        for i, d in enumerate(dialogues):
            for j, u in enumerate(d):
                dif = self.config['max-utterance-length'] - len(u) + 1
                dialogues[i][j] += [self.config['pad-u']] * dif
                
            # only pad the dialogue if we're training a hierarchical model
            if self.config['hierarchical']:
                dif = self.config['max-dialogue-length'] - len(d)
                dialogues[i] += [empty_turn] * dif
        
        return dialogues


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