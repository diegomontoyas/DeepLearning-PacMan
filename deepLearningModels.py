import keras
from keras import models
from keras import layers

class OneHiddenLayerTanhNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.activation = "tanh"
        self.learningRate = 0.05

        self.model = models.Sequential()
        self.model.add(layers.Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions, activation=self.activation, init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation=self.activation, init='uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='msle', metrics=['accuracy'])

class TwoHiddenLayeraTanhNN:

    def __init__(self, inputDimensions):
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions + outputDimensions) / 2)

        self.activation = "tanh"
        self.learningRate = 0.03

        self.model = models.Sequential()
        self.model.add(layers.Dense(hiddenLayerNeurons, input_dim=inputDimensions, activation=self.activation, init='uniform'))
        self.model.add(layers.Dense(hiddenLayerNeurons, activation=self.activation, init='uniform'))
        self.model.add(layers.Dense(outputDimensions, activation=self.activation, init='uniform'))

        optimizer = keras.optimizers.SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='msle', metrics=['accuracy'])
