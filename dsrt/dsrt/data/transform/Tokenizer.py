from dsrt.config import DataConfig

class Tokenizer:
	def __init__(self, properties, config=DataConfig()):
		self.properties = properties
		self.config = config
	
	def transform(self, dialogues):
		return self.tokenize_dialogues(dialogues)
        
    def tokenize_dialogues(self, dialogues):
        return [self.tokenize_dialogue(d) for d in dialogues]

    def tokenize_dialogue(self, dialogue):
        utterances = dialogue.split('\t')[:-1]
        return [self.tokenize_utterance(u) for u in utterances]
    
    def tokenize_utterance(self, utterance):
        return utterance.split(' ')