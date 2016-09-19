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
from keras.layers import Dense, Activation
from keras.models import Sequential

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random,util,math
from operator import itemgetter

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

        self.model = Sequential()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # TODO: Proably eligible for elimination
        util.raiseNotDefined()
        #return self.model.predict(state)[Directions.getIndex(action)]

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
        # Pick Action
        legalActions = self.getLegalActions(state)
        legalActions.remove(Directions.STOP)

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        else:
            qValues = self.model.predict(state)
            index, element = max(enumerate(qValues), key=itemgetter(1))
            return Directions.fromIndex(index)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getQLearningStateForState(self, state):
        util.raiseNotDefined()


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
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
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class NonDeepQAgent(PacmanQAgent):

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        inputDimensions = 100
        outputDimensions = 4

        # Init one-neuron neural network
        self.model = Sequential()
        self.model.add(Dense(output_dim=outputDimensions, input_dim=inputDimensions, activation="softmax"))
        self.model.compile(optimizer='rmsprop', loss='mse')

    def getQLearningStateForState(self, state):

        walls = state.getWalls().flatten()
        pacmanPosition = state.getPacmanPosition()
        ghostPositions = state.getGhostPositions().flatten()
        ghostDirections = [s.getDirection() for s in state.getGhostStates()]
        pacmanDirection = state.getPacmanState().getPosition()

        return [walls, pacmanPosition, ghostPositions, ghostDirections, pacmanDirection]

    def update(self, state, action, nextState, reward):
        """
           Update Q-Function based on transition
        """

        qState = self.getQLearningStateForState(state)
        actionsQValues = self.model.predict(qState)

        nextQState = self.getQLearningStateForState(nextState)
        nextActionsQValues = self.model.predict(nextQState)
        maxNextActionQValue = max(nextActionsQValues)

        #Update rule
        updatedQValueForAction = reward + self.discount * maxNextActionQValue

        targetQValues = actionsQValues
        targetQValues[Directions.getIndex(action)] = updatedQValueForAction

        self.model.fit(x=qState, y=targetQValues, nb_epoch=1)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:

            # you might want to print your weights here for debugging
            print ("Weights: " + str(self.model.layers[0].get_weights()))
            pass
