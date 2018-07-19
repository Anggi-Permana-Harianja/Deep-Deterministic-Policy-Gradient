'''Deep Deterministic Policy Gradient Crtitic/Value model'''
from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    '''critic/value model '''
    def __init__(self, state_size, action_size):
        '''initialize parameters and build model 
           params
           ------
               state_size, int: dimension of each state
               action_size, int: dimension of each action
        '''
        self.state_size = state_size
        self.action_size = action_size
        
        '''method to build NN'''
        self.build_model()
        
    '''method to build NN'''
    def build_model(self):
        '''build a critic/value network that maps (state, action) pairs -> Q-values'''
        
        '''define input layers for both states and actions'''
        states = layers.Input(shape = (self.state_size, ), name = 'states')
        actions = layers.Input(shape = (self.action_size, ), name = 'action')
        
        '''hidden layers for states inputs '''
        net_states = layers.Dense(units = 32, activation = 'relu')(states)
        net_states = layers.Dense(units = 64, activation = 'relu')(net_states)
        net_states = layers.Dense(units = 32, activation = 'relu')(net_states)
        
        '''hidden layers for actions inputs '''
        net_actions = layers.Dense(units = 32, activation = 'relu')(actions)
        net_actions = layers.Dense(units = 64, activation = 'relu')(net_actions)
        net_actions = layers.Dense(units = 32, activation = 'relu')(net_actions)
        
        '''try different layer size, batch normalization, regularization, etc '''
        
        '''combine state and action layers created above'''
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        
        '''add layers if needed '''
        
        '''add final output layer to produce action-value (Q-values) '''
        Q_values = layers.Dense(units = 1, name = 'q_values')(net)
        
        '''CREATE MODEL FROM ARCHITECTURE ABOVE'''
        self.model = models.Model(inputs = [states, actions], outputs = Q_values)
        
        #define optimizer and compile model for training with MSE built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer = optimizer, loss = 'mse')
        
        '''compute action gradients, derivative of Q values w.r.t. to actions'''
        action_gradients = K.gradients(Q_values, actions)
        
        '''define an additional function to fetch action gradients to be used by actor model '''
        self.get_action_gradients = K.function(
            inputs = [*self.model.input, K.learning_phase()], 
            outputs = action_gradients)