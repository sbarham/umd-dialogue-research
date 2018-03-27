"""
Typically, this script will be invoked with options and arguments.
When it is not, it becomes an interactive command-line tool.
"""

import argparse
import dsrt
import os
import sys

class App:
    description = '''some description'''
    usage = '''some usage information'''
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=self.description, usage=self.usage)
        self.config = dsrt.Config()
        
        # initialize primary command's arguments
        self.init_global_args(self.parser)
        
        # dispatch the subcommand (or, if it's bad, issue a usage message)
        self.dispatch_subcommand(self.parser)
        
        return
    
    
    ##############################
    #   Application Subcommands  #
    ##############################
    
    def dispatch_subcommand(self, parser):
        args = parser.parse_args(sys.argv[1:2])
        
        if not hasattr(self, args.subcommand):
            print('Unrecognized command.')
            parser.print_help()
            exit(1)
            
        # use dispatch pattern to invoke method with same name
        getattr(self, args.subcommand)()
        
        return

    def train(self):
        '''The 'train' subcommand'''
        parser = argparse.ArgumentParser(description='Train a dialogue model on a corpus of dialogues')
        self.init_train_args(parser)
        
        args = parser.parse_args(sys.argv[2:])
        
        # This logic will eventually need to be moved out to its own submodule, Trainer ...
        config = self.load_config(args.configuration)
        data = self.load_corpus(args.dataset, config)
        
        factory = dsrt.Factory(config=config, data=data)
        
        model = factory.get_flat_encoder_decoder(config=config)
        factory.train(model)
        
        model_name = 'test'
        #config = self.load_config(args.configuration)
        #factory = dsrt.Factory(config=config)
        conversation = factory.get_conversation(model_name)
        
        conversation.start()
        
        return
    
    def converse(self):
        '''The 'converse' subcommand'''
        parser = argparse.ArgumentParser(description='Initiate a conversation with a trained dialogue model')
        self.init_converse_args(parser)
        
        args = parser.parse_args(sys.argv[2:])
        
        # This logic will eventually need to be moved out to its own submodule, Converse ...
        
        model_name = 'test'
        config = self.load_config(args.configuration)
        factory = dsrt.Factory(config=config)
        conversation = factory.get_conversation(model_name)
        
        conversation.start()
        
        return
    
    def interactive(self):
        '''The 'interactive' default subcommand, which launches an interactive CLI application'''
        
        print("This submodule has not yet been written.")
        exit(0)
        
        return
    
    
    #############################
    #   Application Utilities   #
    #############################
    
    def load_config(self, path):
        '''Load a configuration file; eventually, support dicts, .yaml, .csv, etc.'''
        if path == None:
            print("Path to config was null; using defaults.")
            return self.config
        
        with open(path) as f:
            config = f.read()
        
        extension = os.path.splitext(path)
        if extension == 'yaml':
            config = yaml.load(config)
        else:
            raise Error('Configuration file type "{}" not supported'.format(extension))
        
        self.config.update(config)
        
        return self.config
    
    def load_corpus(self, path, config):
        '''Load a dialogue corpus; eventually, support pickles and potentially other formats'''
        
        if path == '':
            path = self.default_path_to_corpus
        
        self.data = dsrt.DialogueCorpus(path=path, config=config)
        
        return self.data
    
    
    ###############################
    #   Argument Initialization   #
    ###############################

    def init_global_args(self, parser):
        parser.add_argument('subcommand', help='the subcommand to be run')
        
        return

    def init_train_args(self, parser):
        '''Only invoked conditionally if subcommand is 'train' '''
        parser.add_argument('-C', '--configuration', help='the path to the configuration file to use')
        parser.add_argument('-D', '--dataset', help='the path to the dialogue corpus to train on')

        return

    def init_converse_args(self, parser):
        '''Only invoked conditionally if subcommand is 'converse' '''
        parser.add_argument('-M', '--model', help='the path to the (pretrained) dialogue model to use')

        return
                        
                        
if __name__ == '__main__':
    # build the application
    App()
