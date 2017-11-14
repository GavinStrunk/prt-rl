'''
Created on Nov 14, 2017

@author: gstrunk
'''
from __future__ import division

import numpy as np

class Bandit:
    def __init__(self,m, mean_max=10):
        self.m = m
        self.mean = mean_max
        self.N = 1

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1.0 - (1.0 / self.N)) * self.mean + (1.0 / self.N) * x