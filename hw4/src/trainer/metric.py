import os

class Metric():
    def __init__(self):
        pass

    def update(self):    
        pass

    def get_score(self):
        pass

class MetricMeter(Metric):
    def __init__(self, name='', at=1):
        self.name = name
        self.at = at
        self.n = 0.0
        self.n_corrects = 0.0
        self.name = '{}@{}'.format(name, at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, correct, total):    
        self.n_corrects += correct    
        self.n += total

    def get_score(self):
        return self.n_corrects / self.n if self.n != 0 else 0.0
