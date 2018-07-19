'''Deep Deterministic Policy Gradient, Actor/Policy class'''
from keras import layers, models
from keras.optimizers import Adam
from keras import backend as K

class Actor:
    '''Actor/Policy model'''
    def __init__(self, state_size, action_size, action_low, action_high):
        '''Initialize parameters and build model
           params
           ------
           
           state_size, int: dimension of each state
           action_size, int: dimension of each action
           action_low, array: min value of each action dimension
           action_high, array: max value of each action dimension
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        
        #initialize any other variable
        '''we will build NN in this method'''
        self.build_model()
        
    '''build Actor model with NN '''
    def build_model(self):
        '''build an actor network that maps states -> actions'''
        #define input layers(equal to states)
        states = layers.Input(shape = (self.state_size, ), name = 'states')
        
        #add hidden layers
        net = layers.Dense(units = 32, activation = 'relu')(states)
        net = layers.Dense(units = 64, activation = 'relu')(net)
        net = layers.Dense(units = 32, activation = 'relu')(net)
        
        '''try different layer sizes, activations, add batch normalization, regularizers etc.'''
        
        #add final output layer with sigmoid activation
        '''raw_actions will be transformed to [0, 1] '''
        raw_actions = layers.Dense(units = self.action_size, activation = 'sigmoid', name = 'raw_actions')(net)
        #scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name = 'actions')(raw_actions)
        
        '''CREATE MODEL FROM ALL ARCHITECTURE ABOVE'''
        self.model = models.Model(inputs = states, outputs = actions)
        
        '''DDPG has its loss function'''
        #define loss function using action-value (Q-value) gradients
        action_gradients = layers.Input(shape = (self.action_size, ))
        loss = K.mean(-action_gradients * actions)
        
        #incorporate any additional losses here, e.q from regularizers
        
        #define optimizers and training function
        optimizer = Adam()
        updates_op = optimizer.get_updates(self.model.trainable_weights, [], loss)
        self.train_fn = K.function(
            inputs = [self.model.input, action_gradients, K.learning_phase()], 
            outputs = [], 
            updates = updates_op)
        