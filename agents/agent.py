'''Deep Deterministic Policy Gradient Agent(that inherited actor and critic model) '''
from agents.actor import *
from agents.critic import *
from agents.OUNoise import *
from agents.ReplayBuffer import *
from task import *

from keras import layers, models, optimizers
from keras import backend as K
from collections import namedtuple, deque
import numpy as np
import copy
import random


class DDPG():
    '''reinforcement learning agent that learns using DDPG '''
    
    '''argument task are taken from task.py '''
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        '''scoring '''
        self.score = 0
        self.best_score = -np.inf
        self.count = 0
        
        '''ACTOR/POLICY MODEL '''
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        
        '''CRITIC/VALUE MODEL '''
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        '''initialize target model parameters with local model parameters '''
        #model.set_weights and model.get_weights are Keras functions
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
       
        '''model process using UONoise object '''
        self.exploration_mu = 0.5
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        '''replay memory '''
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        '''algorithm parameters '''
        self.gamma = 0.99 #discount factor
        self.tau = 0.01 #for soft update of target parameters
        
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        
        return state
    
    '''learn and save experience after action taken'''
    def step(self, action, reward, next_state, done):
        #update score
        self.total_reward += reward
        self.count += 1
        
        #save experience/reward
        #self.last_state taken from reset_episeode method
        self.memory.add(self.last_state, action, reward, next_state, done)
        
        '''learn, if enough samples are available in memory
           self.learn() is method within this class
        '''        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample() #deque experiences from memory
            self.learn(experiences)
            
        #roll over last state and action
        self.last_state = next_state
        
    '''returns actions taken from given state as per current policy '''
    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        '''model.predict is a keras function '''
        action = self.actor_local.model.predict(state)[0]
        
        return list(action + self.noise.sample()) #add some noise for exploration
    
    '''LEARN USING NN, update policy and value parameters using given batch of experience tuples '''
    def learn(self, experiences):
        #convert experience tuples to separate arrays for each element (state, actions, rewards, etc)
        '''get states, actions, rewards, dones, next_states from experieces 
           experiences is a tuple thats why e.states, etc
        '''
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        '''get predicted next-state actions and Q-values from target models
           Q_targets_next = critic_target(next_state, actor_target(next_state))
        '''
        #model.predict_on_batch() is a keras function
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        '''compute Q targets for current states and train critic model(local) '''
        #model.train_on_batch is keras function
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x = [states, actions], y = Q_targets)
        
        '''TRAIN actor model 
           critic_local.get_action_gradients is method from Criric class named self.get_action_gradients
           actor_local.train_fn is a method of custom training function from Actor class named self.train_fn
        '''
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1]) #custom training function
        
        '''soft update target models, soft_update method will deined below'''
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)
        
        '''track best score'''
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
    
    '''soft update method '''
    def soft_update(self, local_model, target_model):
        '''soft update model parameters 
           local_model argument is self.critic_target_model
               local_model.get_weights() is Critic/Actor method self.critic/actor _local.model()
               model.get_weights() is keras function
        '''
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        
        assert len(local_weights) == len(target_weights) #local and target model parameter must have same dimensions
        
        '''CALCULATE soft update'''
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)