# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from keras.layers import Dense
from keras.models import Sequential
import keras

from experienceReplayHelper import ExperienceReplayHelper
from game import *
from learningAgents import ReinforcementAgent
import numpy as np
from featureExtractors import *

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.model = None
        self.featuresExtractor = DistancesExtractor()
        self.replayMemory = []

        self.batchSize = 32

        self.maxReplayMemorySize = 20000
        self.minReplayMemorySize = 5000

        self.initialEpsilon = 1
        self.finalEpsilon = 1
        self.epsilonSteps = 10000
        self.epsilon = self.initialEpsilon

        self.updateCount = 0

    def initModel(self, sampleState):
        """
        Initializes the deep learning model
        :param sampleState: A sample state to determine dimensionality
        """
        util.raiseNotDefined()

    def remember(self, state, action, reward, nextState):

        if len(self.replayMemory) > self.maxReplayMemorySize:
            self.replayMemory.pop(0)

        qState = self.featuresExtractor.getFeatures(state, action)
        nextQState = self.featuresExtractor.getFeatures(nextState, action)
        isNextStateFinal = nextState.isWin() or nextState.isLose()

        self.replayMemory.append((qState, action, reward, nextQState, isNextStateFinal))

    def sampleReplayBatch(self, size):
        return random.sample(self.replayMemory, size)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        if self.model is None: self.initModel(state)

        # Pick Action
        legalActions = self.getLegalActions(state)
        legalActions.remove(Directions.STOP)

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        else:
            qState = self.featuresExtractor.getFeatures(state, None)
            qValues = list(enumerate(self.model.predict(np.array([qState]))[0]))
            qValues = sorted(qValues, key=lambda x: x[1], reverse=True)

            #index, element = max(enumerate(qValues), key=itemgetter(1))

            for index, qValue in qValues:
                action = Directions.fromIndex(index)
                if action in legalActions:
                    return action

        return None

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        util.raiseNotDefined()


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class DeepQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)

        self.featuresExtractor = PositionsDirectionsExtractor()
        self.discount = 0

    def initModel(self, sampleState):

        qState = self.featuresExtractor.getFeatures(sampleState, Directions.NORTH)

        inputDimensions = len(qState)
        outputDimensions = 4
        hiddenLayerNeurons = 20#int((inputDimensions+outputDimensions)/2)

        # Init neural network
        self.model = Sequential()
        self.model.add(Dense(output_dim=hiddenLayerNeurons, input_dim=inputDimensions, activation="tanh", init='uniform'))
        self.model.add(Dense(hiddenLayerNeurons/2, activation="tanh", init='uniform'))
        self.model.add(Dense(outputDimensions, activation="tanh", init='uniform'))

        optimizer = keras.optimizers.SGD(lr=0.01)
        #optimizer = 'rmsprop'

        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    def update(self, state, action, nextState, reward):
        """
           Update Q-Function based on transition
        """
        if self.model is None: self.initModel(state)

        self.remember(state, action, util.rescale(reward, -510, 1000, -1, 1), nextState)

        if len(self.replayMemory) < 1000:#self.minReplayMemorySize:
            return

        rawBatch = self.sampleReplayBatch(self.batchSize)

        trainingBatchQStates = []
        trainingBatchTargetQValues = []

        for aQState, anAction, aReward, aNextQState, isNextStateFinal in rawBatch:

            actionsQValues = self.model.predict(np.array([aQState]))[0]

            nextActionsQValues = self.model.predict(np.array([aNextQState]))[0]
            maxNextActionQValue = max(nextActionsQValues)

            # Update rule
            if isNextStateFinal:
                updatedQValueForAction = aReward
            else:
                updatedQValueForAction = (aReward + self.discount * maxNextActionQValue)

            targetQValues = actionsQValues.copy()
            targetQValues[Directions.getIndex(anAction)] = updatedQValueForAction

            trainingBatchQStates.append(aQState)
            trainingBatchTargetQValues.append(targetQValues)

        self.model.train_on_batch(x=np.array(trainingBatchQStates), y=np.array(trainingBatchTargetQValues))
        self.updateCount += 1
        self.epsilon = max(self.finalEpsilon, 1.00 - float(self.updateCount) / float(self.epsilonSteps))

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print ("Weights: " + str(self.model.model.layers[0].get_weights()))

