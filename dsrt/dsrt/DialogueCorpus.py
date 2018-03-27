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


class DialogueSampleSet:
    """
    Fields:
        #dialogues             the list of tokenized dialogues
        #length                the length of the sampleset (i.e., the number of dialogues)
        #max_dialogue_length   the length of the longest dialogue in all samplesets
        #max_utterance_length  the length of the longest utterance in all samplesets
        #encoder_x             the inputs to the encoder
        #decoder_x             the inputs to the decoder
        #decoder_y             the target output of the decoder
    """
    def __init__(self, dialogues=[], config=Config()):
        self.dialogues = dialogues
        self.length = len(dialogues)
        self.config = config
        self.init_logger()
        
        # reserved vocabulary items
        self.pad_u = '<pad_u>'
        self.pad_d = '<pad_d>'
        self.start = '<start>'
        self.stop = '<stop>'
        self.unk = '<unk>'
        
        # these are set in #dialogues_to_adjacency_pairs()
        self.encoder_x = []
        self.decoder_x = []
        self.decoder_y = []
        
        self.record_sequence_lengths()
        
        return
        
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
        
        return
    
    def pad_pair_and_prepare(self):
        # pad the dialogues, pair them up (for non-hierarchical models), and
        # prepare the encoder/decoder inputs/targets
        self.pad_dialogues()
        self.dialogues_to_adjacency_pairs()
        self.make_encoder_decoder_split()
        
        return
        
    def record_sequence_lengths(self):
        # record original sequence lengths at the utterance and dialogue level
        self.dialogue_lengths = []
#         self.utterance_lengths = []
        self.utterance_lengths_flat = []
        
        for dialogue in self.dialogues:
            # get dialogue length
            self.dialogue_lengths += [len(dialogue)]

            # get constituent utterances lengths
            lens = [len(u) for u in dialogue]
#             self.utterance_lengths += [lens]
            self.utterance_lengths_flat += lens
        
        self.max_dialogue_length = max(self.dialogue_lengths)
        self.max_utterance_length = max(self.utterance_lengths_flat)
        
        return
        
    def pad_dialogues(self):
        """
        Pad the entire dataset.
        This involves adding padding at the end of each sentence, and in the case of
        a hierarchical model, it also involves adding padding at the end of each dialogue,
        so that every training sample (dialogue) has the same dimension.
        The padded dialogues are then prepared for their role as encoder_x, decoder_x, and
        decoder_y (encoder and decoder inputs, and decoder targets).
        """
        self.log('info', 'Padding the dialogues ...')
        
        if self.config['hierarchical']:
            empty_turn = [self.pad_d] * (self.max_utterance_length + 1)
        
        for i, d in enumerate(self.dialogues):
            for j, u in enumerate(d):
                dif = self.max_utterance_length - len(u) + 1
                self.dialogues[i][j] += [self.pad_u] * dif
#                 self.utterance_lengths[i][j] = len(self.dialogues[i][j])
        
#         std = self.max_utterance_length + 1
#         for i, lens in enumerate(self.utterance_lengths):
#             for j, length in enumerate(lens):
#                 dif = self.max_utterance_length - length + 1
#                 self.dialogues[i][j] += [self.pad_u] * dif
#                 self.utterance_lengths[i][j] = len(self.dialogues[i][j])
                
            # only pad the dialogue if we're training a hierarchical model
            if self.config['hierarchical']:
                dif = self.max_dialogue_length - self.dialogue_lengths[i]
                self.dialogues[i] += [empty_turn] * dif
        
        return
    
    def dialogues_to_adjacency_pairs(self):
        if self.config['hierarchical']:
            pass
        self.dialogues = [ap for d in self.dialogues for ap in self.dialogue_to_adjacency_pairs(d)]
        
        return

    def dialogue_to_adjacency_pairs(self, dialogue):
        adjacency_pairs = []
        for i in range(len(dialogue)):
            if i + 1 < len(dialogue):
                adjacency_pairs += [[dialogue[i], dialogue[i + 1]]]
        
        return adjacency_pairs
    
    def make_encoder_decoder_split(self):
        """
        For now, this assumes a flat (non-hierarchical) model, and therefore
        assumes that dialogues are simply adjacency pairs.
        """
        # prepare the encoder_x
        self.encoder_x = copy.deepcopy([pair[0] for pair in self.dialogues])
        
        decoder = [pair[1] for pair in self.dialogues]
        
        # prepare decoder_x (prefix the <start> symbol to every second-pair part)
        self.decoder_x = copy.deepcopy([list([self.start] + u)[:-1] for u in decoder])
        
        # prepare decoder_y (postfix the <stop> symbol to every second-pair part)
        self.decoder_y = copy.deepcopy(decoder)
        for i in range(len(self.decoder_y)):
            self.decoder_y[i][-1] = self.stop
        
        return
    
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


    

class DialogueCorpus:
    def __init__(self, path=None, config=Config()):
        # load configuration
        self.config = config
        self.path_to_corpus = path
        
        if path is None:
            self.path_to_corpus = self.config['path-to-corpus']
        
        # reserved vocabulary items
        self.pad_u = '<pad_u>'
        self.pad_d = '<pad_d>'
        self.start = '<start>'
        self.stop = '<stop>'
        self.unk = '<unk>'
        
        # init logger
        self.init_logger()
        
        self.log('info', 'Logger initialized')
        self.log('info', 'Configuration loaded')
        self.log('warn', 'Preparing to process the dialogue corpus ...')
        
        self.corpus_loaded = False
        
        # initialize training bookkeeping parameters
#         self._epochs_completed = 0
#         self._index_in_epoch = 0
        
        # load and tokenize the dataset
        self.load_corpus() # <-- tokenization happens here (there's a good reason)
        self.load_vocab()
        
        # tokenize and split
        self.split_corpus()
        self.compute_max_sequence_lengths()
        
        # vectorize
        self.initialize_encoders()
        self.vectorize_corpus()
        
        # report success!
        self.log('warn', 'Corpus succesfully loaded! Ready for training.')
        
        return
    
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
        
        return
    
    ######################
    #       LOADING      #
    ######################
    
    def load_corpus(self):
        self.log('info', 'Loading the dataset ...')
        
        dialogues = []
        
        if not self.corpus_loaded:
            with open(self.path_to_corpus, 'r') as f:
                dialogues = list(f)
                
        # filter out long dialogues, or dialogues with long utterances
        dialogues = self.tokenize_dialogues(dialogues) # <-- we need to tokenize first so the filtering 
                                                       #     registers the correct utterance lengths
        dialogues = self.filter_dialogues_by_length(dialogues)
                
        # if desired, retain only a subset of the dialogues:
        if self.config['restrict-sample-size']:
            dialogues = np.random.choice(dialogues, self.config['sample-size'])
        
        self.dialogues = dialogues
        
        return
        
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
        
        self.vocab_set = set.union(reserved_words, corpus_words)
        self.vocab_list = list(self.vocab_set)
        
        self.vocab_size = len(self.vocab_list)
        
        return
    
    
    ######################
    #    TOKENIZATION    #
    ######################
    
    def tokenize_corpus(self):
        self.log('info', 'Tokenizing the dataset ...')
        self.dialogues = self.tokenize_dialogues(self.dialogues)
        
    def tokenize_dialogues(self, dialogues):
        return [self.tokenize_dialogue(d) for d in dialogues]

    def tokenize_dialogue(self, dialogue):
        utterances = dialogue.split('\t')[:-1]
        return [self.tokenize_utterance(u) for u in utterances]
    
    def tokenize_utterance(self, utterance):
        return utterance.split(' ')
    
    
    ###########################
    #   FILTERING BY LENGTH   #
    ###########################
    
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
    
    
    ##########################
    #    TRAIN/TEST SPLIT    #
    ##########################
    
    def split_corpus(self):
        self.log('info', 'Splitting the corpus into train/test subsets ...')
        
        # grab some hyperparameters from our config
        split = self.config['train-test-split']
        rand_state = self.config['random-state']
        
        # split the corpus into train and test samples
        train, test = train_test_split(self.dialogues, train_size=split, random_state=rand_state)
        
        self.train = DialogueSampleSet(train)
        self.test = DialogueSampleSet(test)
        
        return
    
    
    ###################
    #     PADDING     #
    ###################
    
    def compute_max_sequence_lengths(self):
        self.log('info', 'Recording sequence lengths ...')
        
        self.max_dialogue_length = max(self.train.max_dialogue_length, self.test.max_dialogue_length)
        self.max_utterance_length = max(self.train.max_utterance_length, self.test.max_utterance_length)
        
        self.train.max_dialogue_length = self.max_dialogue_length
        self.test.max_dialogue_length = self.max_dialogue_length
        self.train.max_utterance_length = self.max_utterance_length
        self.test.max_utterance_length = self.max_utterance_length
        
        return
    
    
    #################################
    #     INTEGER VECTORIZATION     #
    #################################
    
    """
    A NOTE:
    This should have been obvious to the thinking man, but any reasonable dialogue corpus will be
    *far* too big to one-hot encode all in one go -- think 10,000 word vocabulary x 4,000,000 words x
    4 bytes per ohe-vector entry: that's 10 * 4 * 4 = 160 GB of one-hot vectors. That *will* fit in
    our Azure supercomputer's memory (it has a memory of 240GB), but it makes testing impossible
    on any other machine (and the Azure machine is far too expensive to use for testing). Instead,
    we'll have to vectorize on demand, on the fly -- unless we encode sentences as integer (index)
    sequences, and feed these into a Keras Embedding layer
    """
    
    def initialize_encoders(self):
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
    
    def vectorize_corpus(self):
        """
        Vectorize the entire dataset using integer (index) encoding.
        """
        self.log('info', 'Vectorizing the dialogues (this may take a while) ...')
        
        # pad the dialogues, pair them up (for non-hierarchical models), and
        # prepare the encoder/decoder inputs/targets
        self.train.pad_pair_and_prepare()
        self.test.pad_pair_and_prepare()
        
        self.remember_ex = self.train.encoder_x
        self.remember_dx = self.train.decoder_x
        self.remember_dy = self.train.decoder_y
        
        # vectorize train samples
        self.train.encoder_x = np.array([self.vectorize_utterance(u) for u in self.train.encoder_x])
        self.train.decoder_x = np.array([self.vectorize_utterance(u) for u in self.train.decoder_x])
        self.train.decoder_y = np.array([self.vectorize_utterance(u) for u in self.train.decoder_y])
        self.train.decoder_y_ohe = np.array([self.ie_to_ohe_utterance(u) for u in self.train.decoder_y])
        
        # vectorize test samples
        self.test.encoder_x = np.array([self.vectorize_utterance(u) for u in self.test.encoder_x])
        self.test.decoder_x = np.array([self.vectorize_utterance(u) for u in self.test.decoder_x])
        self.test.decoder_y = np.array([self.vectorize_utterance(u) for u in self.test.decoder_y])
        self.test.decoder_y_ohe = np.array([self.ie_to_ohe_utterance(u) for u in self.test.decoder_y])
        
        return
        
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
    
    #################################
    #       OHE VECTORIZATION       #
    #################################
    
    
    def ie_to_ohe_utterance(self, utterance):
        return self.ohe.transform(utterance.reshape(len(utterance), 1))
    
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
    
    
    #################
    #   BATCHING    #
    #################
    
#     def next_batch(self):
#         start = self._index_in_epoch
        
#         # Shuffle for the first epoch
#         if self._epochs_completed == 0 and start == 0 and self.config['shuffle']:
#             perm = np.arange(self.num_train_samples)
#             np.random.shuffle(perm)
#             self._train = self.t_train_dia[perm]
#             self._seqlens = [self.train_utt_seqlens[i] for i in perm]
        
#         # If we're out of training samples ...
#         if start + self.config['batch-size'] > self.num_train_samples:
#             # ... then we've finished the epoch
#             self._epochs_completed += 1
            
#             # Gather the leftover dialogues from this epoch
#             num_leftover_samples = self.num_train_samples - start
#             leftover_dialogues = self._train[start:self.num_train_samples]
#             leftover_seqlens = self._seqlens[start:self.num_train_samples]
            
#             # Get a new permutation of the training dialogues
#             if self.config['shuffle']:
#                 perm = numpy.arange(self.num_train_samples)
#                 np.random.shuffle(perm)
#                 self._train = self.t_train_dia[perm]
#                 self._seqlens = [self.train_utt_seqlens[i] for i in perm]
                
#             # Start next epoch
#             start = 0
#             self._index_in_epoch = batch_size - rest_num_examples
#             end = self._index_in_epoch
            
#             # Put together our batch from leftover and new dialogues
#             new_dialogues = self._train[start:end]
#             new_seqlens = self._seqlens[start:end]
#             batch = np.concatenate((leftover_dialogues, new_dialogues), axis=0)
#             seqlens = np.concatenate((leftover_seqlens, new_seqlens), axis=0)
            
#             # prepare the decoder input/output
#             #TODO
            
#             # release the processed batch
#             return (self.vectorize_batch_ohe(batch), seqlens)
#         else:
#             # update the current index in the training data
#             end = self._index_in_epoch + self.config['batch-size']
#             self._index_in_epoch = end
            
#             # get the next batch
#             batch = self._train[start:end]
#             seqlens = self._seqlens[start:end]
            
#             # prepare the decoder input/output
#             #TODO
            
#             # release the processed batch
#             return (self.vectorize_batch_ohe(batch), seqlens)