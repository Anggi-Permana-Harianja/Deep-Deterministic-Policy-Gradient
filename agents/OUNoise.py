'''Ornstein-Uhlenbeck Noise
   spesific noise process that has some desired properties
'''

import numpy as np
import copy

class OUNoise:
    '''Ornstein-Uhlbeck process'''
    def __init__(self, size, mu, theta, sigma):
        '''initialize parameters and noise process'''
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset() #will be another method
        
    def reset(self):
        '''reset the internal state ( = noise) to mean(mu)'''
        self.state = copy.copy(self.mu)
        
    def sample(self):
        '''update internal state and return it as a noise sample'''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state