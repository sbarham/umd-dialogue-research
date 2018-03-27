from dsrt.config import DataConfig

class AdjacencyPairer:
	def __init__(self, properties, config=DataConfig()):
		self.properties = properties
		self.config = config
		
	def transform(self, dialogues):
		return self.dialogues_to_adjacency_pairs(dialogues)
		
	def dialogues_to_adjacency_pairs(self, dialogues):
        return [ap for d in dialogues for ap in self.dialogue_to_adjacency_pairs(d)]

    def dialogue_to_adjacency_pairs(self, dialogue):
        adjacency_pairs = []
        for i in range(len(dialogue)):
            if i + 1 < len(dialogue):
                adjacency_pairs += [[dialogue[i], dialogue[i + 1]]]
        
        return adjacency_pairs