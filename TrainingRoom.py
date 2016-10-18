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

    def __init__(self, layoutName, trainingEpisodes, replayFile, featuresExtractor, initialEpsilon, finalEpsilon, discount = 0.8, batchSize = 32, memoryLimit = 50000, epsilonSteps = None, minExperience = 5000):
        self.featuresExtractor = featuresExtractor
        self.batchSize = batchSize
        self.discount = discount
        self.trainingEpisodes = trainingEpisodes
        self.layoutName = layoutName
        self.memoryLimit = memoryLimit
        self.initialEpsilon = initialEpsilon
        self.finalEpsilon = finalEpsilon
        self.epsilonSteps = epsilonSteps if epsilonSteps is not None else trainingEpisodes * 0.3
        self.epsilon = initialEpsilon
        self.minExperience = minExperience

        print("Loading data...")
        self.replayMemory = shelve.open(replayFile).values() if replayFile is not None else []

        # Init neural network
        sampleState = self.replayMemory[0][0] if self.replayMemory else self.makeGame(False, pacmanAgents.RandomAgent())[0].state
        qState = self.featuresExtractor.getFeatures(sampleState, Directions.NORTH)

        self.model = deepLearningModels.OneHiddenLayerReLULinearNN(len(qState))

        #Stats
        self.stats = util.Stats(isOffline=True,
                                discount=discount,
                                trainingEpisodes=trainingEpisodes,
                                model=self.model,
                                minExperiences=minExperience,
                                activationFunction=self.model.activation,
                                learningRate=self.model.learningRate,
                                featuresExtractor=featuresExtractor,
                                initialEpsilon=initialEpsilon,
                                finalEpsilon=finalEpsilon,
                                batchSize=batchSize,
                                epsilonSteps=self.epsilonSteps,
                                notes="Random data")

    def sampleReplayBatch(self):
        return random.sample(self.replayMemory, self.batchSize)

    def beginTraining(self):
        import Queue
        self._queue = Queue.Queue()
        self._train()

    def _train(self):
        startTime = time.time()
        print("Beginning " + str(self.trainingEpisodes) + " training episodes")
        print("Collecting minimum experience before training...")

        game, agents, display, rules = self.makeGame(displayActive=False)

        previousState = game.state
        currentState = util.getSuccessor(agents, display, previousState, agents[0].getAction(previousState, self.epsilon))

        episodes = 0
        trainingLossSum = 0
        accuracySum = 0
        rewardSum = 0
        wins = 0
        games = 0

        while episodes < self.trainingEpisodes:

            # Update replay memory
            action = agents[0].getAction(currentState, self.epsilon)
            newState = util.getSuccessor(agents, display, currentState, action)
            reward = newState.getScore() - currentState.getScore()
            previousState = currentState
            currentState = newState
            self.replayMemory.append((currentState, action, reward, newState))

            rewardSum += reward

            if len(self.replayMemory) > self.memoryLimit:
                self.replayMemory.pop(0)

            if newState.isWin() or newState.isLose():
                game, agents, display, rules = self.makeGame(displayActive=False)
                currentState = game.state

                wins += 1 if newState.isWin() else 0
                games += 1

            if len(self.replayMemory) < self.minExperience:
                continue

            # Take a raw batch from replay memory
            rawBatch = self.sampleReplayBatch()

            trainingBatchQStates = []
            trainingBatchTargetQValues = []

            # Convert raw states to our q-states and calculate update policy for each transition in batch
            for aState, anAction, aReward, aNextState in rawBatch:

                #aReward = util.rescale(aReward, -510, 1000, -1, 1)

                aQState = self.featuresExtractor.getFeatures(state=aState, action=anAction)
                aNextQState = self.featuresExtractor.getFeatures(state=aNextState, action=anAction)
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
                    updatedQValueForAction = (aReward + self.discount * maxNextActionQValue)

                targetQValues[Directions.getIndex(anAction)] = updatedQValueForAction

                trainingBatchQStates.append(aQState)
                trainingBatchTargetQValues.append(targetQValues)

            loss, accuracy = self.model.model.train_on_batch(x=np.array(trainingBatchQStates), y=np.array(trainingBatchTargetQValues))
            trainingLossSum += loss
            accuracySum += accuracy

            self.epsilon = max(self.finalEpsilon, 1.00 - float(episodes) / float(self.epsilonSteps))

            if episodes % 20 == 0:

                averageLoss = trainingLossSum/20.0
                averageAccuracy = accuracySum/20.0

                print("______________________________________________________")
                print("Episode: " + str(episodes))
                print("Average training loss: " + str(averageLoss))
                print("Average accuracy: " + str(averageAccuracy))
                print("Total reward for last episodes: " + str(rewardSum))
                print("Epsilon: " + str(self.epsilon))
                print("Total wins: " + str(wins))

                self.stats.record(averageLoss, averageAccuracy, wins, self.epsilon)
                trainingLossSum = 0
                rewardSum = 0
                accuracySum = 0

            if episodes % 100 == 0:
                self._queue.put(lambda: self.playOnce(displayActive=True))

            episodes += 1

        print("Finished training, turning off epsilon...")
        print("Calculating average score...")

        endTime = time.time()

        n = 0
        scoreSum = 0

        while True:
            scoreSum += self.playOnce(displayActive = n > 20)
            n += 1

            if n == 20:
                avg = scoreSum/20.0
                self.stats.close(averageScore20Games=avg, learningTime=(endTime-startTime)/60.0)
                print("Average score: "+ str(avg))

    def playOnce(self, displayActive):

        game, agents, display, rules = self.makeGame(displayActive=displayActive)
        currentState = game.state
        display.initialize(currentState.data)

        while not (currentState.isWin() or currentState.isLose()):
            action = agents[0].getAction(state=currentState, epsilon=0)
            currentState = util.getSuccessor(agents, display, currentState, action)

        return currentState.getScore()

    def makeGame(self, displayActive, pacmanAgent = None):

        if not displayActive:
            import textDisplay
            display = textDisplay.NullGraphics()
        else:
            import graphicsDisplay
            display = graphicsDisplay.PacmanGraphics(frameTime=0.01)

        theLayout = layout.getLayout(self.layoutName)
        if theLayout == None: raise Exception("The layout " + self.layoutName + " cannot be found")

        rules = ClassicGameRules()
        agents = [pacmanAgent or pacmanAgents.TrainedAgent(nn=self.model.model, featuresExtractor=self.featuresExtractor)] \
                 + [ghostAgents.DirectionalGhost(i + 1) for i in range(theLayout.getNumGhosts())]

        game = rules.newGame(theLayout, agents[0], agents[1:], display)

        return game, agents, display, rules

if __name__ == '__main__':
    trainingRoom = TrainingRoom(layoutName="smallClassic",
                                trainingEpisodes=2000,
                                replayFile=None,#"./training files/replayMem_mediumClassic.txt",
                                featuresExtractor=SimpleListExtractor(),
                                initialEpsilon=1,
                                finalEpsilon=0.05)
    trainingRoom.beginTraining()
