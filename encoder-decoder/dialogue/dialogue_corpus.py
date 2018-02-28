# import keras.
from config import Config

class DialogueCorpus:
    def __init__(corpus, config=Config()):
        # read in the configuration for DialogueCorpus
        self.config = config['dialogue-corpus-config']

        self.corpus = corpus

    def import_word_embeddings(source='glove'):
        """
        Download and or prepare a pre-computed set of word embeddings, and
        place them in a 2D tensor shaped so as to fit in the Embedding layers
        of our UtteranceEncoder.

        source : str -- designates the word embedding set to use; can be one of:
            * glove :    for GloVe (Global Vectors for Word Representation), developed
                         at Stanford in 2014
            * word2vec : developed by Tomas Mikolov at Google in 2013
        """

        
