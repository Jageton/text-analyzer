from tkinter import *

class Buffer:

    def __init__(self, log):
        self._log = log

    def write(self, text):
        self._log.insert(END, text)

    def flush(self):
        pass