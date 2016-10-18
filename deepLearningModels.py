import keras
from keras import models
from keras import layers

class OneNeuronNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4

        self.activation = "linear"
        self.learningRate = 0.005

        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=outputDimensions, input_dim=inputDimensions, activation=self.activation, init='uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class OneHiddenLayerTanhNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.activation = "tanh"
        self.learningRate = 0.01

        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions, activation=self.activation, init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation=self.activation, init='uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class OneHiddenLayerRMSPropNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.activation = "tanh"
        self.learningRate = 0.01

        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions, activation=self.activation, init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation=self.activation, init='uniform'))

        optimizer = keras.optimizers.RMSprop()
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class OneHiddenLayerReLULinearNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = 20#int((inputDimensions + outputDimensions) / 2)

        self.learningRate = 0.01

        self.activation = "ReLU + linear"
        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions, activation="relu", init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation="linear", init='uniform'))

        optimizer = keras.optimizers.SGD()
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class OneHiddenLayerReLULinearAdamNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.learningRate = 0.01

        self.activation = "ReLU + linear"
        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions, activation="relu", init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation="linear", init='uniform'))

        optimizer = keras.optimizers.Adam(lr=0.00001)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class OneHiddenLayerTanhLinearNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.learningRate = 0.005

        self.activation = "Tanh + linear"
        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions, activation="tanh", init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation="linear", init='uniform'))

        optimizer = keras.optimizers.SGD()
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class TwoHiddenLayersTanhNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.activation = "tanh"
        self.learningRate = 0.01

        self.model = models.Sequential()
        self.model.add(layers.Dense(hiddenLayerNeurons, input_dim=inputDimensions, activation=self.activation, init='uniform'))
        self.model.add(layers.Dense(hiddenLayerNeurons, activation=self.activation, init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation=self.activation, init='uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class TwoHiddenLayersTanhLinearNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.activation = "Tanh + linear"
        self.learningRate = 0.01

        self.model = models.Sequential()
        self.model.add(layers.Dense(hiddenLayerNeurons, input_dim=inputDimensions, activation="tanh", init='uniform'))
        self.model.add(layers.Dense(hiddenLayerNeurons, activation="tanh", init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation="linear", init='uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

class TwoHiddenLayersReLULinearNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.activation = "ReLU + linear"
        self.learningRate = 0.01

        self.model = models.Sequential()
        self.model.add(layers.Dense(hiddenLayerNeurons, input_dim=inputDimensions, activation="relu", init='uniform'))
        self.model.add(layers.Dense(6, activation="relu", init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation="linear", init='uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])