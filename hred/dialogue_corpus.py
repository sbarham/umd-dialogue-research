# external imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from numpy import argmax

import logging
import re

# our own imports
from config import Config

"""
DONE: add the <pad> character to the vocab so that it gets included in the ie/ohe encoders
DONE: go through the t_train_dia/t_train_utt/t_test_dia/t_test_utt and pad all seqs
DONE: implement a vectorize/devectorize_dialogue() function
DONE/*: implement the next_batch function, which should also return original sequence lengths:
          so four outputs -- batch_enc_in, batch_dec_in, batch_dec_out, seqlens
DONE: implement optional sequence trimming (rather discarding very long sequences) --
        this is crucial, because the longest dialogue is 89 turns long, and the longest utterance
        is basically a 684-word essay; that means that, given a modest 10000-word vocabulary,
        a single dialogue, one-hot encoded, takes ~ 100 * 600 * 10,000 * 4 bytes to represent,
        i.e., 2.4GB. Given a batch size of, say 64 (i.e., ~50, which is trendy these days), well,
        that's impracticable to say the least. That's ~150GB for a single batch of ohe dialogues.
        Thus, it's crucial to limit the size of the representation by discarding outliers
        in terms of sequence length (whether dialogue or utterance length).
        
TODO: finish next_batch(), such that it returns also the decoder inputs/outputs
"""

class DialogueCorpus:
    def __init__(self, config=Config()):
        # load configuration
        self.config = config
        
        # init logger
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
        
        self.log('info', 'Configuration loaded')
        self.log('info', 'Logger initialized')
        self.log('warn', 'Preparing to process the dialogue corpus ...')
        
        # initialize corpus-related parameters
        self.corpus_loaded = False
        self.pad_u = '<pad_u>'
        self.pad_d = '<pad_d>'
        self.start = '<start>'
        self.stop = '<stop>'
        
        # initialize training bookkeeping parameters
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        # load the dataset
        self.log('info', 'Loading the dataset ...')
        self.load_corpus()
        self.load_vocab()
        
        # tokenize and split
        self.log('info', 'Tokenizing the dataset ...')
        self.tokenize_corpus() # NB: also converts to numpy array
        self.log('info', 'Filtering out long samples ...')
        self.filter_dialogues_by_length() # filter out dialogues/utterances that are too long
        self.log('info', 'Splitting the corpus into train/test subsets ...')
        self.split_corpus()
        self.log('info', 'Recording sequence lengths ...')
        self.record_sequence_lengths()
        
        # pad and vectorize
        self.log('info', 'Padding the dialogues ...')
        self.pad_corpus()
        self.log('info', 'Initializing the encoders ...')
        self.initialize_encoders()
        self.log('info', 'Vectorizing the dialogues (this may take a while) ...')
        self.vectorize_corpus()
        
        # flatten the dialogues into adjacency pairs if we're training a non-hierarchical model
        self.log('warn', 'Converting dialogues to adjacency pairs ...')
        self.corpus_to_adjacency_pairs()
        
        # arraify the datasets ...
        self.t_train_dia_vec = np.array(self.t_train_dia_vec)
        self.t_test_dia_vec = np.array(self.t_test_dia_vec)
        
        # report success!
        self.log('warn', 'Corpus succesfully loaded! Ready for training.')
        
        return
    
    
    ######################
    #       LOADING      #
    ######################
    
    def load_corpus(self):
        corpus = self.config['path-to-corpus']
        
        if not self.corpus_loaded:
            with open(corpus, 'r') as f:
                self.dialogues = list(f)
                
        # if desired, retain only a subset of the dialogues:
        if self.config['restrict-sample-size']:
            self.dialogues = np.random.choice(self.dialogues, self.config['sample-size'])
        
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
        reserved_words = set([self.pad_u, self.pad_d, self.start, self.stop])
        corpus_words = set([w for d in self.dialogues for w in re.split('\s', d)])
        
        self.vocab_set = set.union(reserved_words, corpus_words)
        self.vocab_list = list(self.vocab_set)
        
    def split_corpus(self):
        # grab some hyperparameters from our config
        split = self.config['train-test-split']
        rand_state = self.config['random-state']
        
        # split the corpus into train and test samples
        self.t_train_dia, self.t_test_dia = train_test_split(self.t_dialogues, train_size=split, random_state=rand_state)
        
        # record num samples
        self.num_train_samples = len(self.t_train_dia)
        self.num_test_samples = len(self.t_test_dia)
        
        return
    
    
    ######################
    #    TOKENIZATION    #
    ######################
    
    def tokenize_corpus(self):
        self.t_dialogues = self.tokenize_dialogues(self.dialogues)
        
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
    
    def filter_dialogues_by_length(self):
        # for filtering out long dialogues and utterances, we'll need these settings:
        max_dl = self.config['max-dialogue-length']
        use_max_dl = not (self.config['use-corpus-max-dialogue-length'])
        max_ul = self.config['max-utterance-length']
        use_max_ul = not (self.config['use-corpus-max-utterance-length'])
        
        filtered_dialogues = []
        
        # if we're putting a limit on dialogue length, 
        # iterate through the dialogues ...
        if use_max_dl:
            for dialogue in self.t_dialogues:
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
                    filtered_dialogues += dialogue
    
    
    ###################
    #     PADDING     #
    ###################
    
    def record_sequence_lengths(self):
        """
        Make a record of the lengths of every dialogue- and utterance-level sequence (i.e., the
        lengths of all dialogues, and the lengths of all utterances in each dialogue). These lengths
        are used as a convenience in padding the sequences.
        
        NB:
        The lengths of dialogue- and utterance-level sequences are
        recorded as:
            self.train_dia_seqlens (lengths of training dialogues)
            self.train_utt_seqlens (lengths of training utterances)
            self.test_dia_seqlens  (lengths of test dialogues)
            self.test_utt_seqlens  (lengths of test utterances)
        """
        sample_sets = ['train', 'test']
        
        for sample_set in sample_sets:
            t_dialogues = getattr(self, 't_' + sample_set + '_dia')
            # record the original seqlens at the utterance and dialogue level
            seqlens_dia = []
            seqlens_utt = []
            seqlens_utt_flat = []
            
            for t_dialogue in t_dialogues:
                # get dialogue length
                seqlens_dia += [len(t_dialogue)]

                # get constituent utterances lengths
                lens = [len(u) for u in t_dialogue]
                seqlens_utt += [lens]
                seqlens_utt_flat += lens

            setattr(self, sample_set + '_dia_seqlens', seqlens_dia)
            setattr(self, sample_set + '_utt_seqlens', seqlens_utt)
            setattr(self, sample_set + '_dia_maxlen', max(seqlens_dia))
            setattr(self, sample_set + '_utt_maxlen', max(seqlens_utt_flat))
            
        self.max_dialogue_length = max(self.train_dia_maxlen, self.test_dia_maxlen)
        self.max_utterance_length = max(self.train_utt_maxlen, self.test_utt_maxlen)
        
        return
        
    def pad_corpus(self):
        """
        Pad the entire dataset.
        This involves adding padding at the end of each sentence, but it
        also involves adding padding at the end of each dialogue, so that every
        training sample (dialogue) has the same dimension.
        The padded samples are stored as `t_train_dia` and `t_test_dia`.
        """
        if self.config['hierarchical']:
            empty_turn = [self.pad_d] * self.max_utterance_length
        
        sample_sets = ['train', 'test']
        for sample_set in sample_sets:
            dia_seqlens = getattr(self, sample_set + '_dia_seqlens')
            utt_seqlens = getattr(self, sample_set + '_utt_seqlens')
            samples = getattr(self, 't_' + sample_set + '_dia')
            
            for i, lens in enumerate(utt_seqlens):
                # pad the utterances ...
                for j, length in enumerate(lens):
                    utt_diff = self.max_utterance_length - length
                    samples[i][j] += [self.pad_u] * utt_diff
                
                # only pad the dialogue if we're training a hierarchical model
                if self.config['hierarchical']:
                    dia_diff = self.max_dialogue_length - dia_seqlens[i]
                    samples[i] += [empty_turn] * dia_diff
        
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
        # create the integer encoder and fit it to our corpus' vocab
        self.ie = LabelEncoder()
        self.ie_vocab = self.ie.fit_transform(self.vocab_list)
        
        # only create the OHE encoder if we have to:
        if self.config['one-hot-encode']:
            self.ohe = OneHotEncoder(sparse=False)
            self.ohe_vocab = self.ohe.fit_transform(self.ie_vocab.reshape(len(self.ie_vocab), 1))
        
        return
    
    def vectorize_corpus(self):
        """
        Vectorize the entire dataset using integer (index) encoding. The resulting
        training and testing sets will be saved as class fields `t_train_dia_vec` and
        `t_test_dia_vec`, each of which will be a 3D tensor, whose dimensions are
        [num_dialogues x dialogue_len x utterance_len].
        """
        sample_sets = ['train', 'test']
        for sample_set in sample_sets:
            t_dialogues_vec = []
            t_dialogues = getattr(self, 't_' + sample_set + '_dia')
            t_dialogues_vec += [self.vectorize_dialogue(dia) for dia in t_dialogues]
            setattr(self, 't_' + sample_set + '_dia_vec', t_dialogues_vec)

    
    def vectorize_dialogue(self, dia):
        """
        Take in a dialogue (a sequence of tokenized utterances) and transform it into a 
        sequence of sequences of indices
        """
        return np.array([self.vectorize_utterance(utt) for utt in dia])
    
    def vectorize_utterance(self, utterance):
        """
        Take in a tokenized utterance and transform it into a sequence of indices
        """
        return self.ie.transform(utterance)
    
    def devectorize_dialogue(self, dialogue):
        """
        Take in a dialogue of ohe utterances and transform them into a tokenized dialogue
        """
        return [self.devectorize_utterance(u) for u in dialogue]
    
    def devectorize_utterance(self, utterance):
        """
        Take in a sequence of indices and transform it back into a tokenized utterance
        """
        return self.ie.inverse_transform(utterance)
    
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
        ie_utterance = self.ie.transform(utterance)
        ohe_utterance = self.ohe.transform(ie_utterance.reshape(len(ie_utterance), 1))
        
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
        utterance = self.ie.inverse_transform(ie_utterance)
        
        return utterance
    
    
    ###############################################
    #     FLATTEN dialogues to ADJACENCY PAIRS    #
    ###############################################
    
    def corpus_to_adjacency_pairs(self):
        sample_sets = ['train', 'test']
        for sample_set in sample_sets:
            dialogues = getattr(self, 't_' + sample_set + '_dia_vec')
            
            adjacency_pairs = []
            
            for i, dialogue in enumerate(dialogues):
                # go through each dialogue and accumulate adjacency pairs
                adjacency_pairs += self.dialogue_to_adjacency_pairs(dialogue)
            
            # now overwrite the sample set with these
            setattr(self, 't_' + sample_set + '_dia_vec', adjacency_pairs)
            
    def dialogue_to_adjacency_pairs(self, dialogue):
        adjacency_pairs = []
        for i in range(len(dialogue)):
            if i + 1 < len(dialogue):
                adjacency_pairs += [[dialogue[i], dialogue[i + 1]]]
        
        return adjacency_pairs
                
            
        
    #######################################
    #    ENCODER/DECODER preprocessing    #
    #######################################
    
    def corpus_to_decoder_inputs(self):
        sample_sets = ['train', 'test']
        for sample_set in sample_sets:
            pass
            
    def dialogue_to_decoder_inputs(self, dialogue):
        pass
        
    def utterance_to_decoder_inputs(self, utterance):
        pass
    
    def corpus_to_decoder_outputs(self):
        sample_sets = ['train', 'test']
        for sample_set in sample_sets:
            pass
        
    def dialogue_to_decoder_outputs(self, dialogue):
        pass
        
    def utterance_to_decoder_outputs(self, utterance):
        pass
    
    
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
    
    def next_batch(self):
        start = self._index_in_epoch
        
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and self.config['shuffle']:
            perm = np.arange(self.num_train_samples)
            np.random.shuffle(perm)
            self._train = self.t_train_dia[perm]
            self._seqlens = [self.train_utt_seqlens[i] for i in perm]
        
        # If we're out of training samples ...
        if start + self.config['batch-size'] > self.num_train_samples:
            # ... then we've finished the epoch
            self._epochs_completed += 1
            
            # Gather the leftover dialogues from this epoch
            num_leftover_samples = self.num_train_samples - start
            leftover_dialogues = self._train[start:self.num_train_samples]
            leftover_seqlens = self._seqlens[start:self.num_train_samples]
            
            # Get a new permutation of the training dialogues
            if self.config['shuffle']:
                perm = numpy.arange(self.num_train_samples)
                np.random.shuffle(perm)
                self._train = self.t_train_dia[perm]
                self._seqlens = [self.train_utt_seqlens[i] for i in perm]
                
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            
            # Put together our batch from leftover and new dialogues
            new_dialogues = self._train[start:end]
            new_seqlens = self._seqlens[start:end]
            batch = np.concatenate((leftover_dialogues, new_dialogues), axis=0)
            seqlens = np.concatenate((leftover_seqlens, new_seqlens), axis=0)
            
            # prepare the decoder input/output
            #TODO
            
            # release the processed batch
            return (self.vectorize_batch_ohe(batch), seqlens)
        else:
            # update the current index in the training data
            end = self._index_in_epoch + self.config['batch-size']
            self._index_in_epoch = end
            
            # get the next batch
            batch = self._train[start:end]
            seqlens = self._seqlens[start:end]
            
            # prepare the decoder input/output
            #TODO
            
            # release the processed batch
            return (self.vectorize_batch_ohe(batch), seqlens)