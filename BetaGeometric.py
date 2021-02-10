import numpy as np
import grapheme
from mpmath import gamma

np.random.seed(17)


class BetaGeometric:
    def __init__(self, alpha, beta):
        '''
          Beta Geometric Distribution
          Parameters:
            alpha   = No. of trials
            beta    = No. of failures
        '''
        self.alpha = alpha
        self.beta = beta

    def update_parameters(self, data):
        '''
          Update Paramters of Geometric Distribtuion based on given data
          Parameters:
            data = Observed data
        '''
        self.alpha = self.alpha + len(data)
        self.beta = self.beta + np.sum([(grapheme.length(x) - 1) for x in data])

    def probability(self, x):
        '''
          Probability of x with Cummulative Density Function of Beta Geometric Distribution
          Parameter:
            x   = Data value
          Output:
            p   = Calculated Probability
        '''
        beta = lambda a, b: (gamma(a) * gamma(b)) / gamma(a + b)
        p = beta(self.alpha + 1, self.beta + x - 1) / beta(self.alpha, self.beta)

        return float(p)