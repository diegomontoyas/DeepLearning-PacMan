import shelve

from keras.layers import Dense
from keras.models import Sequential
import keras

import ghostAgents
import layout
import pacmanAgents
import qlearningAgents
from game import *
from featureExtractors import *
from pacman import ClassicGameRules
import deepLearningModels

class TrainingRoom:

    def __init__(self, layoutName, trainingEpisodes, replayFile, featuresExtractor, initialEpsilon, finalEpsilon, discount = 0.8, batchSize = 32, memoryLimit = 50000, epsilonSteps = 1000):
        self.featuresExtractor = featuresExtractor
        self.batchSize = batchSize
        self.discount = discount
        self.trainingEpisodes = trainingEpisodes
        self.layoutName = layoutName
        self.memoryLimit = memoryLimit
        self.initialEpsilon = initialEpsilon
        self.finalEpsilon = finalEpsilon
        self.epsilonSteps = epsilonSteps
        self.epsilon = initialEpsilon

        print("Loading data...")
        self.replayMemory = shelve.open(replayFile).values() if replayFile is not None else []

        # Init neural network
        sampleState = self.replayMemory[0][0]
        qState = self.featuresExtractor.getFeatures(sampleState, Directions.NORTH)

        self.model = deepLearningModels.OneHiddenLayerTanhNN(len(qState))

        #Stats
        self.stats = util.Stats(isOffline=True,
                                discount=discount,
                                trainingEpisodes=trainingEpisodes,
                                activationFunction=self.model.activation,
                                learningRate=self.model.learningRate,
                                featuresExtractor=featuresExtractor,
                                initialEpsilon=None,
                                finalEpsilon=None,
                                batchSize=batchSize,
                                epsilonSteps=None,
                                notes="Non random data")

    def sampleReplayBatch(self):
        return random.sample(self.replayMemory, self.batchSize)

    def train(self):
        startTime = time.time()
        print("Beginning " + str(self.trainingEpisodes) + " offline training episodes")

        game, agents, display, rules = self.makeGame(displayActive=False)
        currentState = game.state

        episodes = 0
        trainingLossSum = 0

        while episodes < self.trainingEpisodes:

            # Update replay memory
            action = agents[0].getAction(currentState, self.epsilon)
            newState = util.getSuccessor(agents, display, currentState, action)
            reward = newState.getScore() - currentState.getScore()
            self.replayMemory.append((currentState, action, reward, newState))
            currentState = newState

            if len(self.replayMemory) > self.memoryLimit:
                self.replayMemory.pop(0)
            if newState.isWin() or newState.isLose():
                game, agents, display, rules = self.makeGame(displayActive=False)
                currentState = game.state

            rawBatch = self.sampleReplayBatch()

            trainingBatchQStates = []
            trainingBatchTargetQValues = []

            for aState, anAction, aReward, aNextState in rawBatch:

                aReward = util.rescale(aReward, -510, 1000, -1, 1)

                aQState = self.featuresExtractor.getFeatures(state=aState, action=anAction)
                aNextQState = self.featuresExtractor.getFeatures(state=aNextState, action=anAction)

                actionsQValues = self.model.model.predict(np.array([aQState]))[0]

                nextActionsQValues = self.model.model.predict(np.array([aNextQState]))[0]
                maxNextActionQValue = max(nextActionsQValues)

                updatedQValueForAction = None

                # Update rule
                if aNextState.isWin() or aNextState.isLose():
                    updatedQValueForAction = aReward
                else:
                    updatedQValueForAction = (aReward + self.discount * maxNextActionQValue)

                targetQValues = actionsQValues.copy()
                targetQValues[Directions.getIndex(anAction)] = updatedQValueForAction

                trainingBatchQStates.append(aQState)
                trainingBatchTargetQValues.append(targetQValues)

            loss = self.model.model.train_on_batch(x=np.array(trainingBatchQStates), y=np.array(trainingBatchTargetQValues))[0]
            trainingLossSum += loss

            if episodes % 20 == 0:
                self.stats.record(loss)

            episodes += 1
            self.epsilon = max(self.finalEpsilon, 1.00 - float(episodes) / float(self.epsilonSteps))

            if episodes % 100 == 0:
                print("Completed " + str(episodes) + " training episodes.")
                print("Average training loss: " + str(trainingLossSum/100))
                trainingLossSum = 0

        print("Finished training")

        endTime = time.time()

        n = 0
        scoreSum = 0

        while True:
            scoreSum += self.playOnce()
            n += 1

            if n == 20:
                self.stats.close(averageScore20Games=scoreSum/20.0, learningTime=(endTime-startTime)/60.0)

    def playOnce(self):

        game, agents, display, rules = self.makeGame(displayActive=True)
        currentState = game.state
        display.initialize(currentState.data)

        while not (currentState.isWin() or currentState.isLose()):
            action = agents[0].getAction(currentState, self.epsilon)
            currentState = util.getSuccessor(agents, display, currentState, action)

        return currentState.getScore()

    def makeGame(self, displayActive):

        if not displayActive:
            import textDisplay
            display = textDisplay.NullGraphics()
        else:
            import graphicsDisplay
            display = graphicsDisplay.PacmanGraphics(frameTime=0.01)

        theLayout = layout.getLayout(self.layoutName)
        if theLayout == None: raise Exception("The layout " + self.layoutName + " cannot be found")

        rules = ClassicGameRules()
        agents = [pacmanAgents.TrainedAgent(nn=self.model.model, featuresExtractor=self.featuresExtractor)] \
                 + [ghostAgents.DirectionalGhost(i + 1) for i in range(theLayout.getNumGhosts())]

        game = rules.newGame(theLayout, agents[0], agents[1:], display)

        return game, agents, display, rules

if __name__ == '__main__':
    trainingRoom = TrainingRoom(layoutName="smallGrid",
                                trainingEpisodes=10000,
                                replayFile="./training files/replayMem_smallGrid_2.txt",
                                featuresExtractor=DistancesFoodExtractor(),
                                initialEpsilon=1,
                                finalEpsilon=0.1)
    trainingRoom.train()
