# NumPy
import numpy as np
from numpy import argmax

# our own imports
from dsrt.config import DataConfig


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