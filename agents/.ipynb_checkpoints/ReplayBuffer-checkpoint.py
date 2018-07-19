'''REPLAY BUFFER CLASS
   used to store experience that we could populate better to tweak the performance
'''

import random
from collections import namedtuple, deque

class ReplayBuffer:
    '''Fixed-size buffer to store experience tuples'''
    def __init__(self, buffer_size, batch_size):
        '''initialize a ReplayBuffer object
           params
           -------
               buffer_size: maximum size of buffer
               batch_size: size of each training batch
        '''
        self.memory = deque(maxlen = buffer_size) #internal memory that will be deque
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        
    '''add experience to our memory'''
    def add(self, state, action, reward, next_state, done):
        '''add a new experience to memory'''
        e = self.experience(state, action, reward, next_state, done)
        '''add to memory'''
        self.memory.append(e)
        
    '''get the sample(deque) of stored experience'''
    def sample(self, batch_size = 64):
        '''randomly sample a batch of experiences from memory in size of batch_size'''
        return random.sample(self.memory, k = self.batch_size)
        
    def __len__(self):
        return len(self.memory)