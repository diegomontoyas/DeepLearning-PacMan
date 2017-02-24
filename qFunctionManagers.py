import deepLearningModels
import util
from game import Directions, random
import numpy as np

class QFunctionManager:
    """
    An abstract class which Q function managers should inherit from
    """

    def __init__(self):
        self.trainingRoom = None

    def update(self, transitionsBatch):
        """
        Update the Q-Values from the given batch of transitions
        :param transitionsBatch: List of tuples (qState, action, nextQState, reward)
        """
        raise Exception("Not implemented")

    def getAction(self, rawState, epsilon):
        """
        Choose the best action to take
        :param rawState: The state as provided by the game
        :param epsilon: Epsilon value to use when taking the action
        :return: Action (Direction) to take
        """
        raise Exception("Not implemented")

    def getStatsNotes(self):
        pass

class NNQFunctionManager(QFunctionManager):
    """
    A manager that uses a Neural Network to approximate the Q-Function
    """

    def __init__(self, trainingRoom, model=None, checkPointFile = None):
        """
        Initializes a Q-function manager that uses a neural network behind the scenes
        :param trainingRoom: The training room in which we are going to train.
        :param model: (optional) The model of a neural network to use. If None, a default one is created.
        :param checkPointFile: (optional) The file from which to restore a previously persisted model.
        """
        QFunctionManager.__init__(self)

        self.trainingRoom = trainingRoom

        if model is not None:
            self.model = model

        elif checkPointFile is None:
            # Init neural network
            sampleState = self.trainingRoom.makeGame(False).state
            qState = self.trainingRoom.featuresExtractor.getFeatures(sampleState, Directions.NORTH)
            self.model = deepLearningModels.OneHiddenLayerTanhLinearNN(len(qState))

        else:
            import keras
            self.model = keras.models.load_model(checkPointFile)
            self.model.activation = None
            self.model.learningRate = None

    def update(self, transitionsBatch):
        """
        Update the Q-Values from the given batch of transitions
        :param transitionsBatch: List of tuples (qState, action, nextQState, reward, isStateFinal, list of legal actions)
        """

        trainingBatchQStates = []
        trainingBatchTargetQValues = []

        # Convert raw states to our q-states and calculate update policy for each transition in batch
        for aQState, anAction, aReward, aNextQState, isTerminal, nextStateLegalActions in transitionsBatch:

            # aReward = util.rescale(aReward, -510, 1000, -1, 1)

            actionsQValues = self.model.model.predict(np.array([aQState]))[0]
            targetQValues = actionsQValues.copy()

            # Update rule
            if isTerminal:
                updatedQValueForAction = aReward

            else:
                nextActionsQValues = self.model.model.predict(np.array([aNextQState]))[0]
                nextStateLegalActionsIndices = [Directions.getIndex(action) for action in nextStateLegalActions]

                try: nextStateLegalActionsIndices.remove(4)
                except: pass

                nextStateLegalActionsQValues = np.array(nextActionsQValues)[nextStateLegalActionsIndices]
                maxNextActionQValue = max(nextStateLegalActionsQValues)
                updatedQValueForAction = (aReward + self.trainingRoom.discount * maxNextActionQValue)

            targetQValues[Directions.getIndex(anAction)] = updatedQValueForAction

            trainingBatchQStates.append(aQState)
            trainingBatchTargetQValues.append(targetQValues)

        return self.model.model.train_on_batch(x=np.array(trainingBatchQStates), y=np.array(trainingBatchTargetQValues))

    def getAction(self, rawState, epsilon):
        legalActions = rawState.getLegalActions()
        legalActions.remove(Directions.STOP)

        qState = self.trainingRoom.featuresExtractor.getFeatures(rawState, None)

        if util.flipCoin(epsilon):
            return random.choice(legalActions)

        else:
            qValues = list(enumerate(self.model.model.predict(np.array([qState]))[0]))
            qValues = sorted(qValues, key=lambda x: x[1], reverse=True)

            for index, qValue in qValues:
                action = Directions.fromIndex(index)
                if action in legalActions:
                    return action

    def getStatsNotes(self):
        """
        :return: Additional notes to add to the beginning of the stats file.
        """

        return " Model: " + self.model.__class__.__name__ \
            + " ActivationFunction: " + str(self.model.activation or "") \
            + " NNLearningRate: " + str(self.model.learningRate or "")

    def saveCheckpoint(self, file):
        """
        Save the model to the provided file
        :param file: Location
        """
        self.model.model.save(file)

class NonDeepQFunctionManager(QFunctionManager):
    """
    Abstract class from which non NN based managers should inherit.
    """

    def getQValue(self, qState, action):
        raise Exception("Not implemented")

    def getMaxQValue(self, rawState):

        legalActions = rawState.getLegalActions()
        try: legalActions.remove(Directions.STOP)
        except: pass

        nextStateLegalActionsQValues = [self.getQValue(rawState, action) for action in legalActions]
        return max(nextStateLegalActionsQValues or [0])

    def getAction(self, rawState, epsilon):

        legalActions = rawState.getLegalActions()
        legalActions.remove(Directions.STOP)

        if util.flipCoin(epsilon):
            return random.choice(legalActions)

        else:
            qValues = [(Directions.getIndex(action), self.getQValue(rawState, action)) for action in legalActions]
            qValues = sorted(qValues, key=lambda x: x[1], reverse=True)

            for index, qValue in qValues:
                action = Directions.fromIndex(index)
                if action in legalActions:
                    return action

class ApproximateQFunctionManager(NonDeepQFunctionManager):
    """
    A manager that uses hand updated weights to approximate the Q-Function.
    """

    def __init__(self, learningRate):
        QFunctionManager.__init__(self)

        self.learningRate = learningRate

        sampleState = self.trainingRoom.replayMemory[0][0] if self.trainingRoom.replayMemory else self.trainingRoom.makeGame(False)[0].state
        qState = self.trainingRoom.featuresExtractor.getFeatures(sampleState, Directions.NORTH)
        self.weights = [random.uniform(0, 1) for i in range(len(qState))]

    def getQValue(self, rawState, action):
        qState = self.trainingRoom.featuresExtractor.getFeatures(rawState, action)

        qValue = 0.0
        for i, weight in enumerate(self.weights):
            qValue += (weight * qState[i])

        return qValue

    def update(self, transitionsBatch):

        # Convert raw states to our q-states and calculate update policy for each transition in batch
        for aState, anAction, aReward, aNextState in transitionsBatch:

            aQState = self.trainingRoom.featuresExtractor.getFeatures(state=aState, action=anAction)
            maxNextActionQValue = self.getMaxQValue(aNextState)

            for i, value in enumerate(aQState):
                self.weights[i] += self.learningRate * (aReward + self.trainingRoom.discount * maxNextActionQValue - self.getQValue(aState, anAction)) * value

            return -1, -1

class TableBasedQFunctionManager(NonDeepQFunctionManager):
    """
    A manager that uses a traditional table to store the Q-values
    """

    def __init__(self, learningRate):
        QFunctionManager.__init__(self)

        self.learningRate = learningRate
        self.qValues = {}

    def getQValue(self, qState, action):
        return self.qValues.get((str(qState), action)) or 0

    def update(self, transitionsBatch):

        # Convert raw states to our q-states and calculate update policy for each transition in batch
        for aState, anAction, aReward, aNextState in transitionsBatch:

            aQState = self.trainingRoom.featuresExtractor.getFeatures(state=aState, action=anAction)

            maxNextActionQValue = self.getMaxQValue(aNextState)

            self.qValues[(str(aQState), anAction)] = self.getQValue(aQState, anAction) \
                + self.learningRate * (aReward + self.trainingRoom.discount * maxNextActionQValue - self.getQValue(aQState, anAction))

            return -1, -1