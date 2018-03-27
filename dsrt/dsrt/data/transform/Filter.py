from dsrt.data import DataConfig

class Filter:
	def __init__(self, properties, config=DataConfig()):
		self.properties = properties
		self.config = config
		
	def transform(self, dialogues):
		return self.filter_dialogues_by_length(dialogues)
	
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