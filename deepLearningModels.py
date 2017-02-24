import keras
from keras import models
from keras import layers

# Note: A lot code in this module was left un-modularized so changing certain parameters and testing with
# different options was easier.

class Model:
    def __init__(self, inputDimensions, learningRate):
        pass

class OneNeuronNN(Model):
    """
    A NN consisting of just one neuron
    """

    def __init__(self, inputDimensions, learningRate=0.005, activation="linear"):
        """
        :param inputDimensions: The number of input dimensions
        :param learningRate: The learning rate
        :param activation: Activation function to use
        """
        Model.__init__(self, inputDimensions, learningRate)

        self.activation = activation
        self.learningRate = learningRate

        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=4, input_dim=inputDimensions,
                                    activation=self.activation, init='uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class OneHiddenLayerReLULinearNN(Model):
    """
    A NN consisting of one hidden layer which uses ReLU as its activation function and linear of the output layer
    """

    def __init__(self, inputDimensions, learningRate = 0.01):
        """
        :param inputDimensions: The number of input dimensions
        :param learningRate: The learning rate
        """
        Model.__init__(self, inputDimensions, learningRate)

        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.learningRate = learningRate

        self.activation = "ReLU + linear"
        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions,
                                    activation="relu", init='uniform'))

        self.model.add(layers.Dense(outputDimensions, activation="linear", init='uniform'))

        optimizer = keras.optimizers.SGD()
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class OneHiddenLayerReLULinearAdamNN(Model):
    """
    A NN consisting of one hidden layer which uses ReLU as its activation function, linear for the output layer,
    and Adam as the optimizer
    """

    def __init__(self, inputDimensions, learningRate = 0.01):
        """
        :param inputDimensions: The number of input dimensions
        :param learningRate: The learning rate
        """
        Model.__init__(self, inputDimensions, learningRate)

        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.learningRate = learningRate

        self.activation = "ReLU + linear"
        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions,
                                    activation="relu", init='uniform'))

        self.model.add(layers.Dense(outputDimensions, activation="linear", init='uniform'))

        optimizer = keras.optimizers.Adam(lr=0.00001)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class OneHiddenLayerTanhLinearNN(Model):
    """
    A NN consisting of one hidden layer which uses tanh as its activation function and linear for the output layer
    """

    def __init__(self, inputDimensions, learningRate=0.01):
        """
        :param inputDimensions: The number of input dimensions
        :param learningRate: The learning rate
        """
        Model.__init__(self, inputDimensions, learningRate)

        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.learningRate = learningRate

        self.activation = "Tanh + linear"
        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions,
                                    activation="tanh", init='lecun_uniform'))

        self.model.add(layers.Dense(outputDimensions, activation="linear", init='lecun_uniform'))

        optimizer = keras.optimizers.SGD()
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class TwoHiddenLayersTanhLinearNN(Model):
    """
    A NN consisting of two hidden layers which use tanh as their activation function and linear for the output layer
    """

    def __init__(self, inputDimensions, learningRate = 0.01):
        """
        :param inputDimensions: The number of input dimensions
        :param learningRate: The learning rate
        """
        Model.__init__(self, inputDimensions, learningRate)

        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.activation = "Tanh + linear"
        self.learningRate = learningRate

        self.model = models.Sequential()
        self.model.add(layers.Dense(hiddenLayerNeurons, input_dim=inputDimensions,
                                    activation="tanh", init='lecun_uniform'))

        self.model.add(layers.Dense(int(hiddenLayerNeurons/2), activation="tanh", init='lecun_uniform'))
        self.model.add(layers.Dense(outputDimensions, activation="linear", init='lecun_uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
