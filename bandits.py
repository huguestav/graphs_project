import random
import numpy as np


class bernoulliArm():
    def __init__(self, p):
        self.p = p
        self.mean = p
        self.var = p * (p-1)

    def sample(self):
        return int(random.random() < self.p)

#class gaussianArm():
#    def __init__(self, mu, sigma):
#        self.mean = mu
#        self.var = sigma
#
#    def sample(self):
#        R = random.gauss(self.mean, self.var)
#        return R*(R>0)*(R<1) + 1.0*(R>1)

class expArm():
    def __init__(self, lambdap):
        self.lambdap = lambdap
        self.mean = (1./self.lambdap)*(1-np.exp(-self.lambdap))
        self.var = 1

    def sample(self):
        return min(-1/self.lambdap*np.log(random.random()),1)


class betaArm():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha/float(alpha+beta)
        self.var = (alpha*beta)/float((alpha+beta)**2*(alpha+beta+1))
    def sample(self):
        return random.betavariate(self.alpha, self.beta)
