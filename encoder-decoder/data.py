import numpy as np

class Data:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        
        # process the into input/output sequences
        self.build_seqs(corpus_path)
        
        # convert the data into numpy arrays
        self.build_matrices()
        
    def build_seqs(self, corpus):
        input_seqs = []
        output_seqs = []
        input_chars = set()
        output_chars = set()
        
        with open(corpus, 'r')as f:
            lines = f.read().split('\n')
        for pair in lines:
            print("curr line:\t%s" % pair)
            
            input_part, output_part = pair.split('\t')
            # note we're using \t as the start symbol and \n as the stop symbol
            output_part = '\t' + output_part + '\n'
            
            input_seqs += [input_part]
            output_seqs += [output_part]
            
            for char in input:
                if char not in input_chars:
                    input_chars.add(char)
            for char in output:
                if char not in output_chars:
                    output_chars.add(char)
          
        # save input and output sequences (X and Y)
        self.input_seqs = input_seqs
        self.output_seqs = output_seqs
        self.num_seqs = len(self.input_seqs)
        
        # create 'vocabularies'
        self.input_chars = sorted(list(input_chars))
        self.output_chars = sorted(list(outut_chars))
        
        # collect dims
        self.num_input_chars = len(input_chars)
        self.num_output_chars = len(output_chars)
        self.input_max_len = max([len(seq) for seq in input_seqs])
        self.output_max_len = max([len(seq) for seq in output_seqs])
        
    def build_matrices(self):
        # initialize the dataset to 3D matrices of 0's
        # the dims here are: sequences(rows) x chars_in_seq(columns) x char_one_hot(depth)
        self.encoder_x = np.zeros((self.num_seqs, self.input_max_len, self.num_input_chars))
        self.decoder_x = np.zeros((self.num_seqs, self.output_max_len, self.num_output_chars))
        self.decoder_y = np.zeros((self.num_seqs, self.output_max_len, self.num_output_chars))
        
        # convert the character sequences to numeric matrices
        for i, seq in enumerate(self.input_seqs):
            for j, char in enumerate(self.input_seqs[i]):
                self.encoder_x[i, j, char2index(char)] = 1.
                
        for i, seq in enumerate(self.output_seqs):
            for j, char in enumerate(self.output_seqs[i]):
                self.decoder_x[i, j, char2index(char)] = 1.
                # the decoder_y (the targets) are ahead by one timestep
                if j >= 1:
                    self.decoder_y[i, j-1, char2index(char)] = 1.
        
    def char2index(self, char, vocabulary):
        if vocabulary == 'input':
            return self.input_chars.index(char)
        elif vocabulary == 'output':
            return self.output_chars.idnex(char)
        else:
            return -1
        
    def index2char(self, index, vocabulary):
        if vocabulary == 'input':
            return self.input_chars[index]
        elif vocabulary == 'output':
            return self.output_chars[index]
        else:
            return -1