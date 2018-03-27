"""
Ideas:
	(x) move data transformers to a sub-package
	(2) build corpus such that it conditionally builds and passes dialogues
	    through the transformers according to properties in DataConfig
"""

# Scikit-Learn imports
from sklearn.model_selection import train_test_split

# NumPy
import numpy as np
from numpy import argmax

# Python stdlib
import logging
import re
import copy

# our own imports
from dsrt.config import DataConfig

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

