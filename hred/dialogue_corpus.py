from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from numpy import argmax

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
        
        # initialize corpus-related parameters
        self.corpus_loaded = False
        self.pad_u = '<pad_u>'
        self.pad_d = '<pad_d>'
        
        # initialize training bookkeeping parameters
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        # load the corpus and make the training/test split
        self.load_corpus()
        
        # pad and vectorize the dialogue samples, remembering their original length
        self.process_corpus()
        
        return
        
    def load_corpus(self):
        if self.corpus_loaded:
            return
        
        # load the corpus into memory
        self.read_input_data(self.config['path-to-corpus'])
        
        # split the corpus into train and test samples
        self.t_train_dia, self.t_test_dia = train_test_split(self.t_dialogues,
                                                     train_size=self.config['train-test-split'],
                                                     random_state=self.config['random-state'])
        
        # record num samples
        self.num_train_samples = len(self.t_train_dia)
        self.num_test_samples = len(self.t_test_dia)
        
        # get the flattened utterance lists for the train and test sets
        self.t_train_utt = [utt for dia in self.t_train_dia for utt in dia]
        self.t_test_utt = [utt for dia in self.t_test_dia for utt in dia]
        
        return
        
    def reload_corpus(self):
        self.corpus_loaded = False
        self.load_corpus()
        
        return
        
    def process_corpus(self):
        # record the original sequence lengths for all dialogues and utterances
        # in both the train and test sets
        self.record_seq_lengths()
        
        # initialize the integer and one-hot encoders
        self.init_ohe()
        
        # pad the sequences
        self.pad_sequences()
        
        # now pad and vectorize the sequences
        # self.vectorize_sequences()
        # ^ we no longer do this en masse b/c it takes too much memory;
        # rather, we vectorize individual batches on the fly when requested
        # via next_batch
        
        # finally, mark that the corpus was successfully loaded
        self.corpus_loaded = True
        
        return
                
    ######################################
    #      READING IN THE DIALOGUE
    ######################################
    
    def read_input_data(self, path_to_corpus):
        with open(path_to_corpus, 'r') as f:
            lines = list(f)

        # each line is a discrete dialogue, with utterances
        # tab-delimited
        # dialogues = []
        t_dialogues = []
        # utterances = []
        # t_utterances = []
        words = [self.pad_u, self.pad_d]
        
        # for filtering out long dialogues and utterances, we'll need these settings:
        max_dl = self.config['max-dialogue-length']
        use_max_dl = not (self.config['use-corpus-max-dialogue-length'])
        max_ul = self.config['max-utterance-length']
        use_max_ul = not (self.config['use-corpus-max-utterance-length'])
        
        for i, line in enumerate(lines):
            dialogue = line.split('\t')[:-1]
            
            # filter out really really long dialogues
            if use_max_dl and len(dialogue) > max_dl:
                continue

            t_dialogue = []
            keep_dia = True
            for utt in dialogue:
                t_utt = utt.split(' ')
                
                # filter out dialogues with really really long utterances
                if use_max_ul and len(t_utt) > max_ul:
                    keep_dia = False
                    break
                
                t_dialogue += [t_utt]
                words += t_utt

            # If the dialogue had utterances that were too large, skip it ...
            if not keep_dia:
                continue
            
            # ... otherwise, keep it:
            # dialogues += [dialogue] # add the dialogue (i.e., list of utterances)
            # utterances += dialogue # add the utterances to the flattened list of all utterances
            t_dialogues += [t_dialogue]
            # t_utterances += t_dialogue
        
        # Record the accumulated dialogues, utterances, tokenized dialogues,
        # and tokenized utterances
        self.t_dialogues = np.array(t_dialogues)
        # self.t_utterances = np.array(t_utterances)
        
        # Record the list of tokens and the list of vocab items
        self.words = words
        self.vocab = list(set(words))
        
        return
    
    ######################################
    # RECORDING SEQLEN and (DE)VECTORIZING
    ######################################
    
    def record_seq_lengths(self):
        """
        The lengths of dialogue- and utterance-level sequences are
        recorded as:
            self.train_dia_seqlens (lengths of training dialogues)
            self.train_utt_seqlens (lengths of training utterances)
            self.test_dia_seqlens  (lengths of test dialogues)
            self.test_utt_seqlens  (lengths of test dialogues)
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
            # setattr(self, sample_set + '_utt_flat_seqlens', seqlens_utt_flat)
            setattr(self, sample_set + '_dia_maxlen', max(seqlens_dia))
            setattr(self, sample_set + '_utt_maxlen', max(seqlens_utt_flat))
            
        self.max_dialogue_length = max(self.train_dia_maxlen, self.test_dia_maxlen)
        self.max_utterance_length = max(self.train_utt_maxlen, self.test_utt_maxlen)
        
        return
    
    def init_ohe(self):
        """
        From here on out,
            - 'ie' stands for 'integer encoded', and
            - 'ohe' stands for 'one-hot encoded'
        """
        # create the encoders
        self.ie = LabelEncoder()
        self.ohe = OneHotEncoder(sparse=False)
        
        # fit the encoders to the corpus vocabulary
        self.ie_vocab = self.ie.fit_transform(self.vocab)
        self.ohe_vocab = self.ohe.fit_transform(self.ie_vocab.reshape(len(self.ie_vocab), 1))
        
        return
        
    def pad_sequences(self):
        pad_d = '<pad_d>' # used to fill empty dialogue turn
        pad_u = '<pad_u>' # used to pad an utterance
        
        empty_turn = [pad_d] * self.max_utterance_length
        
        sample_sets = ['train', 'test']
        for sample_set in sample_sets:
            dia_seqlens = getattr(self, sample_set + '_dia_seqlens')
            utt_seqlens = getattr(self, sample_set + '_utt_seqlens')
            
            for i, lens in enumerate(utt_seqlens):
                for j, length in enumerate(lens):
                    utt_diff = self.max_utterance_length - length
                    getattr(self, 't_' + sample_set + '_dia')[i][j] += [pad_u] * utt_diff
                dia_diff = self.max_dialogue_length - dia_seqlens[i]
                getattr(self, 't_' + sample_set + '_dia')[i] += [empty_turn] * dia_diff
        
        return
        
    def pretty_print_dialogue(self, dia):
        for utt in dia:
            if utt[0] == self.pad_d:
                break
            print(self.prettify_utterance(utt))
                
        return
                      
    def prettify_utterance(self, utt):
        return ' '.join([w for w in utt if not w == self.pad_u])
        
    """
    This should have been obvious to the thinking man, but any reasonable dialogue corpus will be
    *far* too big to one-hot encode all in one go -- think 10,000 word vocabulary x 4,000,000 words x
    4 bytes per ohe-vector entry: that's 10 * 4 * 4 160 GB of one-hot vectors. That *will* fit in
    our Azure supercomputer's memory (it has a memory of 240GB), but it makes testing impossible
    on any other machine (and the Azure machine is far too expensive to use for testing). Instead,
    we'll have to vectorize o demand, on the fly.
    """
#     def vectorize_sequences(self):
#         sample_sets = ['train', 'test']
#         for sample_set in sample_sets:
#             t_dialogues_vec = []
#             t_dialogues = getattr(self, 't_' + sample_set + '_dia')
            
#             for i, dia in enumerate(t_dialogues):
#                 t_dialogues_vec += [[self.vectorize_utterance(utt) for utt in dia]]
            
#             setattr(self, 't_' + sample_set + '_dia_vec', t_dialogues_vec)

    def vectorize_batch(self, batch):
        return np.array([self.vectorize_dialogue(dia) for dia in batch])
    
    def vectorize_dialogue(self, dia):
        # we squeeze it because it's coming out with an extra empty
        # dimension at the front of the shape: (1 x dia x utt x word)
        return np.array([[self.vectorize_utterance(utt) for utt in dia]]).squeeze()
        
    def vectorize_utterance(self, utterance):
        """
        Take in a tokenized utterance and transform it into a sequence of one-hot vectors
        """
        ie_utterance = self.ie.transform(utterance)
        ohe_utterance = self.ohe.transform(ie_utterance.reshape(len(ie_utterance), 1))
        
        return ohe_utterance

    def devectorize_utterance(self, ohe_utterance):
        """
        Take in a sequence of one-hot vectors and transform it into a tokenized utterance
        """
        ie_utterance = [argmax(w) for w in ohe_utterance]
        utterance = self.le.inverse_transform(ie_utterance)
        
        return utterance
    
    ######################################
    # GETTING THE NEXT BATCH IN TRAINING
    ######################################
    
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
            return (self.vectorize_batch(batch), seqlens)
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
            return (self.vectorize_batch(batch), seqlens)