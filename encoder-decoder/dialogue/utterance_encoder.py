from config import Config

from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import GRU

class UtteranceEncoder:
    def __init__(self, config=Config(), logger=Logger()):
        # import the configuration for an UtteranceEncoder
        self.config = config["utterance-encoder-config"]

        # initialize an empty model
        self.model = Sequential()

    def create_model(self):
        # create the embedding layer
        # nb: the mask_zero option causes 0's in the input data to be treated
        #     as a privileged value indicating "no data" -- this is to support
        #     variable length utterances
        # nb: because we're allowing masks, recurrent layers down the line must
        #     be approrpiately configured to use them as well

        encoder_input = Input(shape=(None,))
        encoder_embedding = Embedding(self.config['vocab-size'],
                                      self.config['embedding-dim'],
                                      input_length = self.config['utterance-len'],
                                      mask_zero = True)(encoder_input)

        # create the recurrent layers, one by one
        # nb: the return_state option causes the network to output
        encoder_gru = encoder_embedding
        for i in range(self.config['depth'] - 1):
            encoder_gru = GRU(self.config['rnn-dim'], return_sequences=True)(encoder_gru)

        # create the final recurrent layer
        encoder_outputs, encoder_hidden_state = GRU(self.config['rnn-dim'], return_state=True)(encoder_gru)

        # add all the layers to our model
        self.model = Model(encoder_input, encoder_hidden_state)

        if(self.config['logging-level'] == 'high'):
            print("_________________________________________________________________")
            print("Constructed Encoder.\nSummary follows:\n\n")
            self.model.summary()

    def set_embeddings(self):
        return

    def fit(self):
        return

    def predict(self):
        return
