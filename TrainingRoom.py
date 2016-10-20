import shelve

import deepLearningModels
import ghostAgents
import layout
import pacmanAgents
import QFunctionManagers
from featureExtractors import *
from game import *
from pacman import ClassicGameRules

class TrainingRoom:

    def __init__(self, layoutName, trainingEpisodes, replayFile, featuresExtractor, initialEpsilon, finalEpsilon, useDeepNN=True,
                 learningRate=None, discount = 0.8, batchSize = 32, memoryLimit = 50000, epsilonSteps = None, minExperience = 5000):

        self.featuresExtractor = featuresExtractor
        self.batchSize = batchSize
        self.discount = discount
        self.trainingEpisodes = trainingEpisodes
        self.layoutName = layoutName
        self.memoryLimit = memoryLimit
        self.initialEpsilon = initialEpsilon
        self.finalEpsilon = finalEpsilon
        self.epsilonSteps = epsilonSteps if epsilonSteps is not None else trainingEpisodes * 0.5
        self.epsilon = initialEpsilon
        self.minExperience = minExperience
        self.learningRate = learningRate

        print("Loading data...")
        self.replayMemory = shelve.open(replayFile).values() if replayFile is not None else []

        self.qFuncManager = QFunctionManagers.NNQFunctionManager(self) if useDeepNN else QFunctionManagers.ApproximateQFunctionManager(self)

        #Stats
        self.stats = util.Stats(isOffline=True,
                                discount=discount,
                                trainingEpisodes=trainingEpisodes,
                                minExperiences=minExperience,
                                learningRate=learningRate,
                                featuresExtractor=featuresExtractor,
                                initialEpsilon=initialEpsilon,
                                finalEpsilon=finalEpsilon,
                                batchSize=batchSize,
                                epsilonSteps=self.epsilonSteps,
                                notes=self.qFuncManager.getStatsNotes())

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
        currentState = game.state

        episodes = 0
        trainingLossSum = 0
        accuracySum = 0
        rewardSum = 0
        wins = 0
        games = 0
        deaths = 0

        while episodes < self.trainingEpisodes:

            # Update replay memory
            action = self.qFuncManager.getAction(currentState, epsilon=self.epsilon)
            newState = util.getSuccessor(agents, display, currentState, action)
            reward = newState.getScore() - currentState.getScore()
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
                deaths += 1 if newState.isLose() else 0

            if len(self.replayMemory) < self.minExperience:
                continue

            # Take and converts a batch from replay memory
            batch = self.sampleReplayBatch()
            loss, accuracy = self.qFuncManager.update(batch)
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
                print("Number of deaths: " + str(deaths))

                self.stats.record([averageLoss, averageAccuracy, wins, self.epsilon])
                trainingLossSum = 0
                rewardSum = 0
                accuracySum = 0
                deaths = 0

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
            action = self.qFuncManager.getAction(currentState, epsilon=0)
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
        agents = [pacmanAgents.TrainedAgent()] \
                 + [ghostAgents.DirectionalGhost(i + 1) for i in range(theLayout.getNumGhosts())]

        game = rules.newGame(theLayout, agents[0], agents[1:], display)

        return game, agents, display, rules

if __name__ == '__main__':
    trainingRoom = TrainingRoom(layoutName="mediumClassic",
                                trainingEpisodes=100,
                                replayFile=None,#"./training files/replayMem_mediumClassic.txt",
                                useDeepNN=False,
                                batchSize=1,
                                minExperience=1,
                                learningRate=0.2,
                                featuresExtractor=SimpleListExtractor(),
                                initialEpsilon=1,
                                finalEpsilon=0.05)
    trainingRoom.beginTraining()
