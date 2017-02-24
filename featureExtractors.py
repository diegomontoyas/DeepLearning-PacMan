# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"
import itertools

from game import Directions, Actions
import util
import numpy as np

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist, (pos_x, pos_y)
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        features["eats-food"] = 1.0 if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y] else 0

        # if there is no danger of ghosts then add the food feature
        #if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
        #    features["eats-food"] = 1.0

        cFood = closestFood((next_x, next_y), food, walls)
        dist = cFood[0] if cFood else None

        features["closest-food"] = float(dist) / (walls.width * walls.height) if dist is not None else 1

        # if dist is not None:
        #     # make the distance a number less than one otherwise the update
        #     # will diverge wildly
        #     features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class SimpleListExtractor(FeatureExtractor):
    """
    Uses `SimpleExtractor` to get the features and returns a flattened (unnormalized) vector compatible
     with neural networks.
    """

    def getFeatures(self, state, action):
        qState = np.array(SimpleExtractor().getFeatures(state, action).values()).astype(dtype=float)
        return qState

class PositionsExtractor(FeatureExtractor):
    """
    Extracts just the positions of PacMan and the ghosts as a flattened vector of coordinates, unnormalized vector.
    """

    def getFeatures(self, state, action):
        pacmanPosition = np.array(state.getPacmanPosition()).flatten()
        ghostPositions = np.array(state.getGhostPositions()).flatten()

        return np.concatenate((ghostPositions, pacmanPosition)).astype(dtype=float)

class CompletePositionsExtractor(FeatureExtractor):
    """
    Extracts the positions of PacMan and the Ghosts as pairs of coordinates, along with:

    - The legal actions as a sub-vector of 1's and 0's, one for each direction.
    - Food around PacMan as a sub-vector of 1's and 0's, one for each direction.
    - The shortest distance to the closest food.
    - If there are scared ghosts around PacMan, as a sub-vector of 1's and 0's, one for each direction.
    - The directions of the ghosts as a list of numbers, one for each ghost.
    - Food around PacMan as a sub-vector of 1's and 0's, one for each direction.

    The final vector is normalized between 0 and 1.
    """

    def getFeatures(self, state, action):
        maxBoardSize = max(getBoardSize(state))

        pacmanPosition = np.array(state.getPacmanPosition()).flatten()/float(maxBoardSize)
        ghostPositions = np.array(state.getGhostPositions()).flatten()/float(maxBoardSize)
        standardInfo = getCompleteAdditionalInfo(state)

        return np.concatenate((pacmanPosition, ghostPositions, standardInfo)).astype(dtype=float)

class DistancesExtractor(FeatureExtractor):
    """
    Calculates the horizontal and vertical distances from PacMan to each Ghost as pairs of numbers and inverts them
    by subtracting them from the size of the board. This extractor also includes:

    - The legal actions as a sub-vector of 1's and 0's, one for each direction.
    - Food around PacMan as a sub-vector of 1's and 0's, one for each direction.
    - The shortest distance to the closest food.
    - If there are scared ghosts around PacMan, as a sub-vector of 1's and 0's, one for each direction.
    - The directions of the ghosts as a list of numbers, one for each ghost.
    - Food around PacMan as a sub-vector of 1's and 0's, one for each direction.

    The final vector is normalized between 0 and 1.
    """

    def getFeatures(self, state, action):
        maxBoardSize = max(getBoardSize(state))

        x, y = state.getPacmanPosition()
        distances = np.array([[x-gx, y-gy] for gx, gy in state.getGhostPositions()]).flatten()/float(maxBoardSize)
        standardInfo = getCompleteAdditionalInfo(state)

        qState = np.concatenate((distances, standardInfo)).astype(dtype=float)
        return qState

class InverseDistancesExtractor(FeatureExtractor):
    """
    Extracts the horizontal and vertical distances from PacMan to each Ghost as pairs of numbers, along with:

    - The legal actions as a sub-vector of 1's and 0's, one for each direction.
    - Food around PacMan as a sub-vector of 1's and 0's, one for each direction.
    - The shortest distance to the closest food.
    - If there are scared ghosts around PacMan, as a sub-vector of 1's and 0's, one for each direction.
    - The directions of the ghosts as a list of numbers, one for each ghost.
    - Food around PacMan as a sub-vector of 1's and 0's, one for each direction.

    The final vector is normalized between 0 and 1.
    """

    def getFeatures(self, state, action):
        boardSize = np.array(getBoardSize(state))
        maxBoardSize = max(boardSize)

        x, y = state.getPacmanPosition()
        distances = np.array([[x-gx, y-gy] for gx, gy in state.getGhostPositions()])
        inverseDistances = np.array([boardSize * np.array([c if c != 0 else 1 for c in np.sign(dis)]) - dis for dis in distances]).flatten() / float(maxBoardSize)

        legalActions = getLegalActions(state)
        food = getFoodAroundPacman(state)

        closestFoodTuple = closestFood((state.getPacmanPosition()), state.getFood(), state.getWalls())
        closestFoodDistance = closestFoodTuple[0] if closestFoodTuple else maxBoardSize
        inverseClosestFoodDistance = np.array([maxBoardSize - closestFoodDistance]) / float(maxBoardSize)

        areGhostsScared = getScaredGhosts(state)
        ghostDirections = getGhostDirections(state)
        capsules = getCapsulesAroundPacman(state)

        qState = np.concatenate((inverseDistances, inverseClosestFoodDistance, areGhostsScared, capsules, legalActions, food, ghostDirections)).astype(dtype=float)
        return qState

class ShortSightedBinaryExtractor(FeatureExtractor):
    """
    Calculates if there is any ghost around PacMan as a sub-vector of 0's and 1's, one for every direction, along with:

    - The legal actions as a sub-vector of 1's and 0's, one for each direction.
    - Food around PacMan as a sub-vector of 1's and 0's, one for each direction.
    - The shortest distance to the closest food.
    - If there are scared ghosts around PacMan, as a sub-vector of 1's and 0's, one for each direction.
    - The directions of the ghosts as a list of numbers, one for each ghost.
    - Food around PacMan as a sub-vector of 1's and 0's, one for each direction.

    The final vector is normalized between 0 and 1.
    """

    def getFeatures(self, state, action):
        ghostsNearby = getGhostsAroundPacman(state)
        standardInfo = getCompleteAdditionalInfo(state)

        qState = np.concatenate((ghostsNearby, standardInfo)).astype(dtype=float)
        return qState

class DangerousActionsExtractor(FeatureExtractor):
    """
    Calculates if the action that PacMan is about to take is dangerous, using the decisions taken by
     the `CarefulGreedyAgent`. It also includes the legal actions as a sub-vector of 1's and 0's, one
     for each direction.
    """

    def getFeatures(self, state, action):
        from pacmanAgents import CarefulGreedyAgent

        dangerousActions = CarefulGreedyAgent()._getAction(state)[1]
        dangerousActionsBools = np.array([action in dangerousActions for action in Directions.asList() if action != Directions.STOP]).astype(float)
        legalActions = getLegalActions(state)

        return np.concatenate((dangerousActionsBools, legalActions)).astype(dtype=float)

################################################
#                                              #
#              HELPER FUNCTIONS                #
#                                              #
################################################

def getCompleteAdditionalInfo(state):

    legalActions = getLegalActions(state)
    food = getFoodAroundPacman(state)
    closestFoodDistance = getClosestFoodDistance(state)
    areGhostsScared = getScaredGhosts(state)
    ghostDirections = getGhostDirections(state)
    capsules = getCapsulesAroundPacman(state)

    return np.concatenate((closestFoodDistance, areGhostsScared, capsules, legalActions, food, ghostDirections))

def getBoardSize(state):
    layout = state.data.layout
    return (layout.width, layout.height)

def getClosestFoodDistance(state):
    maxBoardSize = max(getBoardSize(state))
    closestFoodTuple = closestFood((state.getPacmanPosition()), state.getFood(), state.getWalls())
    closestFoodDistance = closestFoodTuple[0] if closestFoodTuple else maxBoardSize
    return np.array([closestFoodDistance])/float(maxBoardSize)

def getGhostDirections(state):
    return np.array([Directions.getIndex(s.getDirection()) for s in state.getGhostStates()]) / 4.0

def getScaredGhosts(state):
    return [s.scaredTimer > 0 for s in state.getGhostStates()]

def getLegalActions(state):
    legalActions = state.getLegalActions()
    if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
    return np.array([Directions.fromIndex(i) in legalActions for i in range(4)])

def getFoodAroundPacman(state):

    x, y = state.getPacmanPosition()
    foodMatrix = state.getFood()
    return np.array([foodMatrix[x+dx][y+dy] for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]])

def getCapsulesAroundPacman(state):

    x, y = state.getPacmanPosition()
    remainingCapsules = state.getCapsules()
    return np.array([(x+dx, y+dy) in remainingCapsules for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]])

def getGhostsAroundPacman(state):

    ghostPositions = state.getGhostPositions()
    x, y = state.getPacmanPosition()
    ghostsNearby = [0] * 8

    for ghostX, ghostY in ghostPositions:

        dx = ghostX-x
        dy = ghostY-y

        if abs(dx) <= 1 and abs(dy) <= 1:
            if dx == 0:
                ghostsNearby[0 if dy > 0 else 1] = 1

            elif dy == 0:
                ghostsNearby[2 if dx < 0 else 3] = 1

            elif dy > 0:
                ghostsNearby[4 if dx < 0 else 5] = 1

            else:
                ghostsNearby[6 if dx < 0 else 7] = 1

    return ghostsNearby

def getTunnelsAroundPacman(state):

    def getPossibleActions(pos, walls):
        possible = []
        x, y = pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(dir)

        try: possible.remove(Directions.STOP)
        except: pass
        return possible

    tunnels = [False]*4
    stateCopy = state.deepCopy()

    x, y = stateCopy.getPacmanPosition()
    walls = stateCopy.getWalls()

    for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, 1), (0, -1)]):
        testPos = x+dx, y+dy

        if walls[testPos[0]][testPos[1]]:
            continue

        tunnels[i] = len(getPossibleActions(testPos, walls)) <= 2

    return tunnels
