class Config(dict):
    def __init__(self, vocab_size=15000, path_to_corpus='../corpora/cornell-movie/dialogues.txt',
                 logging_level='high', max_dialogue_length=10, max_utterance_length=40,
                 shuffle=True, random_state=100):
        # these are more or less global parameters
        self['path-to-corpus'] = path_to_corpus
        self['logging-level'] = logging_level
        self['vocab-size'] = vocab_size
        self['random-state'] = random_state
        self['shuffle'] = shuffle
        # use the max lengths already present in the corpus in case the user doesn't provide values
        # (this is usually preferred)
        self['use-corpus-max-utterance-length'] = max_utterance_length is None
        self['use-corpus-max-dialogue-length'] = max_dialogue_length is None
        # with a default setting of 20x100 dialogues, a one-hot encoded
        # ialogue occupies about 160MB of space in memory
        self['max-utterance-length'] = max_utterance_length
        self['max-dialogue-length'] = max_dialogue_length

        # init parameters relevant to the four components of the network
        self.init_embedding_config()
        self.init_utterance_encoder_config()
        self.init_context_encoder_config()
        self.init_utterance_decoder_config()
        
        # init training and validation parameters
        self.init_training_and_validation_parms()

    def init_embedding_config(self):
        self['embedding-dim'] = 512
        
    def init_utterance_encoder_config(self):
        self['encoding-layer-width'] = 512
        self['encoding-layer-depth'] = 3
        self['encoding-layer-bidirectional'] = True

    def init_context_encoder_config(self):
        self['context-layer-width'] = 512
        self['context-layer-depth'] = 2
        self['context-layer-bidirectional'] = True

    def init_utterance_decoder_config(self):
        self['decoding-layer-width'] = 512
        self['decoding-layer-depth'] = 3
        self['decoding-layer-bidirectional'] = True
        
    def init_training_and_validation_parms(self):
        self['batch-size'] = 64
        self['train-test-split'] = .8
        self['num-epochs'] = 10000
        self['reporting-frequency'] = 100
