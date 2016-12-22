import random


class bernoulliArm():
    def __init__(self, p):
        self.p = p
        self.mean = p
        self.var = p * (p-1)

    def sample(self):
        return int(random.random() < self.p)

