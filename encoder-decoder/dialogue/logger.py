class Logger:
    def __init__(self, config=Config()):
        self.LEVEL = config['logging-level']

    def log(self, msg):
        print("msg") 
