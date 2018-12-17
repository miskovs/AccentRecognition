class SimpleLogger():
    def __init__(self, should_print=False):
        self.print = should_print

    def log(self, message, force=False):
        if self.print or force:
            print(message)
