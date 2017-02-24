# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
import game
import util
import random
import numpy as np
import featureExtractors


class CarefulGreedyAgent(Agent):

    def __init__(self, index=0):
        Agent.__init__(self, index)
        self.lastAction = None

    def getAction(self, state):

        import featureExtractors
        #qState = featureExtractors.ShortSightedBinaryExtractor().getFeatures(state, None)
        return self._getAction(state)[0]

    def _getAction(self, state):
        dangerousActions = {}
        pacmanVisionRadius = 1

        legalActions = state.getLegalActions()
        pacmanPosition = state.getPacmanPosition()
        ghostPositions = state.getGhostPositions()
        ghostStates = state.getGhostStates()

        random.shuffle(legalActions)

        for action in legalActions:
            if action == Directions.STOP: continue

            nextPacmanState = state.generatePacmanSuccessor(action)
            nextPacmanPosition = nextPacmanState.getPacmanPosition()

            if not nextPacmanState.isLose(): return action, dangerousActions

            for ghostNumber, ghostPosition in enumerate(ghostPositions):
                yDiff = ghostPosition[1] - pacmanPosition[1]
                xDiff = ghostPosition[0] - pacmanPosition[0]

                if nextPacmanState.isLose():
                    # We would loose
                    dangerousActions[action] = True
                else:
                    # nextYDiff = ghostPosition[1] - nextPacmanPosition[1]
                    # nextXDiff = ghostPosition[0] - nextPacmanPosition[0]
                    #
                    # ghostInSameColumn = ghostPosition[0] == nextPacmanPosition[0]
                    # ghostInSameRow = ghostPosition[1] == nextPacmanPosition[1]
                    #
                    # ghostAbove = ghostInSameColumn and yDiff > 0
                    # ghostBelow = ghostInSameColumn and yDiff < 0
                    # ghostRight = ghostInSameRow and xDiff > 0
                    # ghostLeft = ghostInSameRow and xDiff < 0
                    #
                    # # We would get closer to a ghost by going that way
                    # if abs(nextYDiff) < pacmanVisionRadius and abs(nextYDiff) < abs(yDiff):
                    #     if (ghostAbove and Directions.NORTH in legalActions) \
                    #             or (ghostBelow and Directions.SOUTH in legalActions):
                    #         dangerousActions[action] = True
                    #
                    # if abs(nextXDiff) < pacmanVisionRadius and abs(nextXDiff) < abs(xDiff):
                    #     if (ghostLeft and Directions.WEST in legalActions) \
                    #             or (ghostRight and Directions.EAST in legalActions):
                    #         dangerousActions[action] = True

                    ghostDirection = ghostStates[ghostNumber].getDirection()

                    # There's a ghost which would kill us once we turn (eg. ghost is going east
                    # and so we are, but as soon as we go north it would kill us)

                    #                      G->  ||             G = Ghost
                    #      ==================   ||             P = PacMan
                    #                         P^||

                    # if nextYDiff == 0:
                    #     if (nextXDiff == -1 and ghostDirection == Directions.EAST) \
                    #             or (nextXDiff == 1 and ghostDirection == Directions.WEST):
                    #         dangerousActions[action] = True
                    #
                    # if nextXDiff == 0:
                    #     if (nextYDiff == -1 and ghostDirection == Directions.NORTH) \
                    #             or (nextYDiff == 1 and ghostDirection == Directions.SOUTH):
                    #         dangerousActions[action] = True

        #print("Dangerous: " + str(list(dangerousActions.keys())))
        dangerousActionsList = list(dangerousActions.keys())

        recommendableActions = filter(lambda x: x not in dangerousActions, legalActions)
        if not recommendableActions: return Directions.STOP, dangerousActionsList
        greedyAction = self.getGreedyAction(state, recommendableActions)

        if greedyAction is None:
            return Directions.STOP, dangerousActionsList
        else:
            return greedyAction, dangerousActionsList

        # if random.uniform(0, 1) < 0.8 \
        #         and self.lastAction not in dangerousActions and self.lastAction in legalActions:
        #     return self.lastAction

        # shuffle(legalActions)
        # for action in legalActions:
        #     if action not in dangerousActions and action != Directions.STOP:
        #         self.lastAction = action
        #         print("Decided to go " + action)
        #         return action
        #
        # return legalActions[0]

    def getGreedyAction(self, state, availableActions):

        if Directions.STOP in availableActions: availableActions.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in availableActions]
        scored = [(scoreEvaluation(state), action) for state, action in successors]

        if not scored: return None

        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

class RandomAgent(Agent):
    def getAction(self, state):
        legalActions = state.getLegalActions()
        if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)

        import featureExtractors
        tunnels = featureExtractors.getTunnelsAroundPacman(state)

        return random.choice(legalActions)

def scoreEvaluation(state):
    return state.getScore()


class TrainedAgent():
    def __init__(self,
                 checkPointFile = "./training files/training stats/With Replay File/Medium Grid/Distances/1479091048.chkpt",
                 extractor = featureExtractors.DistancesExtractor()):

        import keras

        self.featuresExtractor = extractor
        self.model = keras.models.load_model(checkPointFile)

    def getAction(self, rawState):

        legalActions = rawState.getLegalActions()
        legalActions.remove(Directions.STOP)

        qState = self.featuresExtractor.getFeatures(rawState, None)

        qValues = list(enumerate(self.model.model.predict(np.array([qState]))[0]))
        qValues = sorted(qValues, key=lambda x: x[1], reverse=True)

        for index, qValue in qValues:
            action = Directions.fromIndex(index)
            if action in legalActions:
                return action
