import shelve

from keras.layers import Dense
from keras.models import Sequential
import keras

import ghostAgents
import layout
import pacmanAgents
import qlearningAgents
from ExperienceReplayHelper import ExperienceReplayHelper
from game import *
from learningAgents import ReinforcementAgent
import numpy as np
from featureExtractors import *
from pacman import ClassicGameRules


class TrainingRoom:

    def __init__(self, layoutName, trainingEpisodes, replayFile, featuresExtractor, initialEpsilon, finalEpsilon, discount = 0.8, batchSize = 32, epsilonSteps = 1000):
        self.featuresExtractor = featuresExtractor
        self.batchSize = batchSize
        self.initialEpsilon = initialEpsilon
        self.finalEpsilon = finalEpsilon
        self.epsilonSteps = epsilonSteps
        self.epsilon = self.initialEpsilon
        self.discount = discount
        self.trainingEpisodes = trainingEpisodes
        self.layoutName = layoutName

        print("Loading data...")
        self.replayMemory = shelve.open(replayFile).values()

        # Init neural network
        sampleState = self.replayMemory[0][0]
        qState = self.featuresExtractor.getFeatures(sampleState, Directions.NORTH)

        inputDimensions = len(qState)
        outputDimensions = 4
        hiddenLayerNeurons = int((inputDimensions+outputDimensions)/2)

        self.model = Sequential()
        self.model.add(Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions, activation="tanh", init='uniform'))
        #self.model.add(Dense(hiddenLayerNeurons/2, activation="tanh", init='uniform'))
        self.model.add(Dense(outputDimensions, activation="tanh", init='uniform'))

        optimizer = keras.optimizers.SGD(lr=0.02)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        #Stats

    def sampleReplayBatch(self):
        return random.sample(self.replayMemory, self.batchSize)

    def train(self):
        import threading
        print("Beginning " + str(self.trainingEpisodes) + " offline training episodes")

        episodes = 0
        trainingLossSum = 0

        while episodes < self.trainingEpisodes:

            rawBatch = self.sampleReplayBatch()

            trainingBatchQStates = []
            trainingBatchTargetQValues = []

            for aState, anAction, aReward, aNextState in rawBatch:

                aReward = util.rescale(aReward, -510, 1000, -1, 1)

                aQState = self.featuresExtractor.getFeatures(state=aState, action=None)
                aNextQState = self.featuresExtractor.getFeatures(state=aNextState, action=None)

                actionsQValues = self.model.predict(np.array([aQState]))[0]

                nextActionsQValues = self.model.predict(np.array([aNextQState]))[0]
                maxNextActionQValue = max(nextActionsQValues)

                # Update rule
                if aNextState.isWin() or aNextState.isLose():
                    updatedQValueForAction = aReward
                else:
                    updatedQValueForAction = (aReward + self.discount * maxNextActionQValue)

                targetQValues = actionsQValues.copy()
                targetQValues[Directions.getIndex(anAction)] = updatedQValueForAction

                trainingBatchQStates.append(aQState)
                trainingBatchTargetQValues.append(targetQValues)

            trainingLossSum += self.model.train_on_batch(x=np.array(trainingBatchQStates), y=np.array(trainingBatchTargetQValues))[0]
            episodes += 1
            self.epsilon = max(self.finalEpsilon, 1.00 - float(episodes) / float(self.epsilonSteps))

            if episodes % 100 == 0:
                print("Completed " + str(episodes) + " training episodes.")
                print("Average training loss: " + str(trainingLossSum/100))
                trainingLossSum = 0

        print("Finished training")

        while True:
            self.playOnce()

    def playOnce(self):

        import graphicsDisplay
        display = graphicsDisplay.PacmanGraphics(frameTime=0.01)

        theLayout = layout.getLayout(self.layoutName)
        if theLayout == None: raise Exception("The layout " + self.layoutName + " cannot be found")

        rules = ClassicGameRules()
        agents = [qlearningAgents.TrainedAgent(nn=self.model, featuresExtractor=self.featuresExtractor)] \
                 + [ghostAgents.DirectionalGhost(i + 1) for i in range(theLayout.getNumGhosts())]

        game = rules.newGame(theLayout, agents[0], agents[1:], display)

        currentState = game.state
        display.initialize(currentState.data)

        while not (currentState.isWin() or currentState.isLose()):
            action = agents[0].getAction(currentState)
            currentState = currentState.generateSuccessor(0, action)
            display.update(currentState.data)

            try:
                for ghostIndex in range(1, len(agents)):
                    currentState = currentState.generateSuccessor(ghostIndex,
                                                                  agents[ghostIndex].getAction(currentState))
                    display.update(currentState.data)
            except Exception, E:
                pass

if __name__ == '__main__':
    trainingRoom = TrainingRoom(layoutName="smallGrid",
                                trainingEpisodes=40000,
                                replayFile="/Users/Diego/Desktop/replayMem_smallGrid.txt",
                                featuresExtractor=PositionsDirectionsExtractor(),
                                initialEpsilon=1,
                                finalEpsilon=1)

    trainingRoom.train()
