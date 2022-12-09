"""Neuron Class"""

from numpy import array, dot, random, ndarray
from numpy import exp


class Layer:
	"""Neuron Class"""
	def __init__(self, inputs, n_neurons, activation_function):
		"""Constructor"""
		self.weights = random.rand(inputs, n_neurons)
		self.bias = random.rand(n_neurons)
		self.activation_function = self.__activation_function(activation_function)
		self.output = self.estimate_fed_forward()

	def estimate_fed_forward(self, inputs=None):
		"""Estimate Output"""
		if inputs is None:
			return self.activation_function(dot(self.weights, self.bias))
		else:
			return self.activation_function(dot(self.weights, inputs))

	def __activation_function(self, activation_function):
		"""Activation Function"""
		activation_function = activation_function.lower()
		if activation_function == 'sigmoid':
			return self.__sigmoid
		elif activation_function == 'tanh':
			return self.__tanh
		elif activation_function == 'relu':
			return self.__relu
		else:
			return self.__sigmoid
	
	def __sigmoid(self, x):
		"""Sigmoid Function"""
		return 1 / (1 + exp(-x))
	
	def __tanh(self, x):
		"""Tanh Function"""
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

	def __relu(self, x):
		"""Relu Function"""
		return x * (x > 0)


def main():
	# xor true table for 3 inputs
	inputs = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
	xor_outputs = [0,1,1,0,1,0,0,1]
	# create a perceptron with two hidden layers and 3 inputs
	nn_architecture = [3,3,1]




if __name__ == "__main__":
	main()

#%%
from numpy import random
