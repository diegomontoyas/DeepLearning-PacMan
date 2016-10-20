import deepLearningModels
import pacmanAgents
import util
from game import Directions, random
import numpy as np

class QFunctionManager:
    def __init__(self, trainingRoom):
        self.trainingRoom = trainingRoom

    def update(self, transitionsBatch):
        """
        :param transitionsBatch: List of tuples (qState, action, nextQState, reward)
        """
        pass

    def getAction(self, rawState, epsilon):
        pass

class NNQFunctionManager(QFunctionManager):
    def __init__(self, trainingRoom):
        QFunctionManager.__init__(self, trainingRoom)

        # Init neural network
        sampleState = self.trainingRoom.replayMemory[0][0] if self.trainingRoom.replayMemory else self.trainingRoom.makeGame(False)[0].state
        qState = self.trainingRoom.featuresExtractor.getFeatures(sampleState, Directions.NORTH)

        self.model = deepLearningModels.OneHiddenLayerReLULinearNN(len(qState))

    def update(self, transitionsBatch):

        trainingBatchQStates = []
        trainingBatchTargetQValues = []

        # Convert raw states to our q-states and calculate update policy for each transition in batch
        for aState, anAction, aReward, aNextState in transitionsBatch:

            # aReward = util.rescale(aReward, -510, 1000, -1, 1)

            aQState = self.trainingRoom.featuresExtractor.getFeatures(state=aState, action=anAction)
            aNextQState = self.trainingRoom.featuresExtractor.getFeatures(state=aNextState, action=anAction)

            actionsQValues = self.model.model.predict(np.array([aQState]))[0]

            updatedQValueForAction = None
            targetQValues = actionsQValues.copy()

            # Update rule
            if aNextState.isWin() or aNextState.isLose():
                updatedQValueForAction = aReward

            else:
                nextActionsQValues = self.model.model.predict(np.array([aNextQState]))[0]
                nextStateLegalActionsIndices = [Directions.getIndex(action) for action in aNextState.getLegalActions()]

                try:
                    nextStateLegalActionsIndices.remove(4)
                except:
                    pass

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

        qState = self.trainingRoom.featuresExtractor.getFeatures(rawState, Directions.NORTH)

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