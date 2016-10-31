import deepLearningModels
import pacmanAgents
import util
from game import Directions, random
import numpy as np

class QFunctionManager:
    """
    An abstract class which Q function managers should inherit from
    """

    def __init__(self, trainingRoom):
        """
        :param trainingRoom: The room (TrainingRoom) in which we are training in
        """
        self.trainingRoom = trainingRoom

    def update(self, transitionsBatch):
        """
        Update the Q-Values from the given batch of transitions
        :param transitionsBatch: List of tuples (qState, action, nextQState, reward)
        """
        raise Exception("Not implemented")

    def getAction(self, rawState, epsilon):
        """
        Choose the best action to take
        :param rawState: The state
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

    def __init__(self, trainingRoom, checkPointFile = None):
        QFunctionManager.__init__(self, trainingRoom)

        # Init neural network
        sampleState = self.trainingRoom.replayMemory[0][0] if self.trainingRoom.replayMemory else self.trainingRoom.makeGame(False)[0].state
        qState = self.trainingRoom.featuresExtractor.getFeatures(sampleState, Directions.NORTH)

        if checkPointFile is None:
            self.model = deepLearningModels.TwoHiddenLayersLargeTanhLinearNN(len(qState))
        else:
            import keras
            self.model = keras.models.load_model(checkPointFile)

    def update(self, transitionsBatch):

        trainingBatchQStates = []
        trainingBatchTargetQValues = []

        # Convert raw states to our q-states and calculate update policy for each transition in batch
        for aState, anAction, aReward, aNextState in transitionsBatch:

            # aReward = util.rescale(aReward, -510, 1000, -1, 1)

            aQState = self.trainingRoom.featuresExtractor.getFeatures(state=aState, action=anAction)
            aNextQState = self.trainingRoom.featuresExtractor.getFeatures(state=aNextState, action=None)

            actionsQValues = self.model.model.predict(np.array([aQState]))[0]

            targetQValues = actionsQValues.copy()

            # Update rule
            if aNextState.isWin() or aNextState.isLose():
                updatedQValueForAction = aReward

            else:
                nextActionsQValues = self.model.model.predict(np.array([aNextQState]))[0]
                nextStateLegalActionsIndices = [Directions.getIndex(action) for action in aNextState.getLegalActions()]

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

        return " Model: " + type(self.model).__name__ \
            + " ActivationFunction: " + str(self.model.activation) \
            + " NNLearningRate: " + str(self.model.learningRate)

    def saveCheckpoint(self, file):
        self.model.model.save(file)

class NonDeepQFunctionManager(QFunctionManager):
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
    A manager that uses hand updated weights to approximate the Q-Function
    """

    def __init__(self, trainingRoom):
        QFunctionManager.__init__(self, trainingRoom)

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
                self.weights[i] += self.trainingRoom.learningRate * (aReward + self.trainingRoom.discount * maxNextActionQValue - self.getQValue(aState, anAction)) * value

            return -1, -1

class TableBasedQFunctionManager(NonDeepQFunctionManager):
    """
    A manager that uses a traditional table to store the Q-values
    """

    def __init__(self, trainingRoom):
        QFunctionManager.__init__(self, trainingRoom)

        self.qValues = {}

    def getQValue(self, qState, action):
        return self.qValues.get((str(qState), action)) or 0

    def update(self, transitionsBatch):

        # Convert raw states to our q-states and calculate update policy for each transition in batch
        for aState, anAction, aReward, aNextState in transitionsBatch:

            aQState = self.trainingRoom.featuresExtractor.getFeatures(state=aState, action=anAction)

            maxNextActionQValue = self.getMaxQValue(aNextState)

            self.qValues[(str(aQState), anAction)] = self.getQValue(aQState, anAction) \
                + self.trainingRoom.learningRate * (aReward + self.trainingRoom.discount * maxNextActionQValue - self.getQValue(aQState, anAction))

            return -1, -1