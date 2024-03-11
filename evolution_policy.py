import numpy as np
import math
import itertools
from src.deep_model_policy import Policy

# Activation functions ########################################################

def identity(x):
    return x

def tanh(x):
    return np.tanh(x)

def relu(x):
    x_and_zeros = np.array([x, np.zeros(x.shape)])
    return np.max(x_and_zeros, axis=0)

# Dense Multi-Layer Neural Network ############################################

class NeuralNetworkPolicy(Policy):

    def __init__(self, state_size, action_size, h_size, evaluation_mode=False):   # h_size = number of neurons on the hidden layer
        # Set the neural network activation functions (one function per layer)
        self.activation_functions = (relu, tanh)

        # Make a neural network with 1 hidden layer of `h_size` units
        weights = (np.zeros([state_size + 1, h_size]),
                   np.zeros([h_size + 1, action_size]))

        self.shape_list = weights_shape(weights)
        print("Number of parameters per layer:", self.shape_list)

        self.num_params = len(flatten_weights(weights))
        print("Number of parameters (neural network weights) to optimize:", self.num_params)


    def act(self, state, theta):
        weights = unflatten_weights(theta, self.shape_list)

        return feed_forward(inputs=state,
                            weights=weights,
                            activation_functions=self.activation_functions)
    
    def step(self, state, action, reward, next_state, done):
        pass

    def save(self, filename):
        pass


def feed_forward(inputs, weights, activation_functions, verbose=False):

    x = activation_functions[0](inputs @ weights[0][:-1] + weights[0][-1])
    layer_output = activation_functions[1](x @ weights[1][:-1] + weights[1][-1])
    return layer_output


def weights_shape(weights):
    return [weights_array.shape for weights_array in weights]


def flatten_weights(weights):
    """Convert weight parameters to a 1 dimension array (more convenient for optimization algorithms)"""
    nested_list = [weights_2d_array.flatten().tolist() for weights_2d_array in weights]
    flat_list = list(itertools.chain(*nested_list))
    return flat_list


def unflatten_weights(flat_list, shape_list):
    """The reverse function of `flatten_weights`"""
    length_list = [shape[0] * shape[1] for shape in shape_list]

    nested_list = []
    start_index = 0

    for length, shape in zip(length_list, shape_list):
        nested_list.append(np.array(flat_list[start_index:start_index+length]).reshape(shape))
        start_index += length

    return nested_list



class LogisticRegression(Policy):

    def __init__(self, observation_space, action_space):
        self.num_params = observation_space*action_space
        self.action_space = action_space

    def act(self, state, theta):
        return draw_action(state, theta, self.action_space)
    
    def __call__(self, state, theta):
        return self.act(state, theta)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def logistic_regression(s, theta):
    logits = np.dot(s, np.transpose(theta))
    return softmax(logits)

def draw_action(s, theta, action_space):
    theta = np.array([theta[int(len(theta)/action_space*i):int(len(theta)/action_space*(i+1))] for i in range(action_space)])
    probabilities = logistic_regression(s, theta)
    action = np.random.choice(range(len(probabilities)), p=probabilities.ravel())
    return action