class Config(dict):
    def __init__(self):
        # these parameters are shared in some way by multiple components;
        # we control them centrally here
        self.LOGGING_LEVEL = 'high'
        self.VOCAB_SIZE = 15000
        self.UTTERANCE_LENGTH = 100

        # the system's configuration is divided into four parts, each a dict:
        # one for each component
        self['dialogue-corpus-config'] = self.init_dialogue_corpus_config()
        self['utterance-encoder-config'] = self.init_utterance_encoder_config()
        self['context-encoder-config'] = self.init_context_encoder_config()
        self['utterance-decoder-config'] = self.init_utterance_decoder_config()

    def init_dialogue_corpus_config(self):
        config = dict()

        config['logging-level'] = self.LOGGING_LEVEL
        config['vocab-size'] = self.VOCAB_SIZE
        config['utterance-length'] = self.UTTERANCE_LENGTH

        return config

    def init_utterance_encoder_config(self):
        config = dict()

        config['logging-level'] = self.LOGGING_LEVEL
        config['vocab-size'] = self.VOCAB_SIZE
        config['utterance-len'] = self.UTTERANCE_LENGTH
        config['embedding-dim'] = 512
        config['depth'] = 10
        config['rnn-dim'] = 512

        return config

    def init_context_encoder_config(self):
        config = dict()

        config['logging-level'] = self.LOGGING_LEVEL
        config['width'] = 10
        config['depth'] = 10

        return config

    def init_utterance_decoder_config(self):
        config = dict()

        config['logging-level'] = self.LOGGING_LEVEL
        config['vocab-size'] = self.VOCAB_SIZE
        config['utterance-len'] = self.UTTERANCE_LENGTH
        config['embedding-dim'] = 512
        config['depth'] = 10

        return config
