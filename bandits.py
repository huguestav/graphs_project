import random
import numpy as np


class bernoulliArm():
    def __init__(self, p):
        self.p = p
        self.mean = p
        self.var = p * (p-1)

    def sample(self):
        return int(random.random() < self.p)

class gaussianArm():
    def __init__(self, mu, sigma):
        self.mean = mu
        self.var = sigma

    def sample(self):
        R = random.gauss(self.mean, self.var)
        return R*(R>0)*(R<1) + 1.0*(R>1)
        return (random.gauss(self.mean, self.var))

class expArm():
    def __init__(self, lambdap):
        self.lambdap = lambdap
        self.mean = 1./self.lambdap
        self.var = 1./(self.lambdap)**2

    def sample(self):
        R = random.expovariate(self.lambdap)
        return R*(R<1) + 1.0*(R>1)
        #random.expovariate(self.lambdap)

class betaArm():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha/float(alpha+beta)
        self.var = (alpha*beta)/float((alpha+beta)**2*(alpha+beta+1))
    def sample(self):
        return random.betavariate(self.alpha, self.beta)

