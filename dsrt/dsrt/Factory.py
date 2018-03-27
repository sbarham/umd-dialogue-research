"""
Maintains a dialogue research application context, which includes (minimally) 
(1) a Config configuration object (which < dict), and
(2) a DialogueCorpus object, which represents a corpus of dialogues, and understands how to prepare
    them for training encoder-decoder neural networks (i.e., the details of padding, masking,
    vectorization, etc.)
    
Using this application context, is able to construct any kind of neural dialog model desired.
"""

from keras.models import load_model
from dsrt import Conversation, Encoder, Decoder, EncoderDecoder

class Factory:
    def __init__(self, config, data=None):
        self.config = config
        self.data = data
        
        return
    
    def train(self, model):
        model.fit(self.data)
        
    def get_flat_encoder_decoder(self, config=None):
        if not config is None:
            config = dict(self.config, **config)
        else:
            config = self.config
        
        encoder = Encoder(data=self.data, config=config)
        decoder = Decoder(data=self.data, config=config, encoder=encoder)
        
        return EncoderDecoder(config=config, encoder=encoder, decoder=decoder)
    
    def get_hierarchical_encoder_decoder(self, config=None):
        if not config is None:
            config = dict(self.config, **config)
        else:
            config = self.config
        
        print("This method has not yet been implemented. Returning a flat dialogue model instead.")
        
        return self.get_flat_encoder_decoder(config)
    
    def get_conversation(self, model_name=None, config=None):
        if not config is None:
            config = dict(self.config, **config)
        else:
            config = self.config
            
        if model_name is None:
            model_name = config['model-name']
            
        encoder = load_model('models/' + model_name + '_inference_encoder')
        decoder = load_model('models/' + model_name + '_inference_decoder')
            
        return Conversation(encoder=encoder, decoder=decoder, data=self.data)