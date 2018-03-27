from dsrt.config import DataConfig

class Padder:
	def __init__(self, properties, config=DataConfig()):
		self.properties = properties
		self.config = config
		
	def transform(self, dialogues):
		return self.pad_dialogues(dialogues)
		
	def pad_dialogues(self, dialogues):
        """
        Pad the entire dataset.
        This involves adding padding at the end of each sentence, and in the case of
        a hierarchical model, it also involves adding padding at the end of each dialogue,
        so that every training sample (dialogue) has the same dimension.
        """
        self.log('info', 'Padding the dialogues ...')
        
        empty_turn = [self.config['pad-d']] * (self.properties['max-utterance-length'] + 1)
        
        for i, d in enumerate(dialogues):
            for j, u in enumerate(d):
                dif = self.config['max-utterance-length'] - len(u) + 1
                dialogues[i][j] += [self.config['pad-u']] * dif
                
            # only pad the dialogue if we're training a hierarchical model
            if self.config['hierarchical']:
                dif = self.config['max-dialogue-length'] - len(d)
                dialogues[i] += [empty_turn] * dif
        
        return dialogues