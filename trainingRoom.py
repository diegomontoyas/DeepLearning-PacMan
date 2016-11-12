import shelve

import deepLearningModels
import ghostAgents
import layout
import pacmanAgents
import qFunctionManagers
from featureExtractors import *
from game import *
from pacman import ClassicGameRules

class TrainingRoom:
    """
    Where training happens
    """

    def __init__(self, layoutName, trainingEpisodes, replayFile, featuresExtractor, initialEpsilon, finalEpsilon,
                 learningRate=None, discount = 0.8, batchSize = 32, memoryLimit = 50000, epsilonSteps = None,
                 minExperience = 5000, useExperienceReplay = True):

        self.featuresExtractor = featuresExtractor
        self.batchSize = batchSize
        self.discount = discount
        self.trainingEpisodes = trainingEpisodes
        self.layoutName = layoutName
        self.memoryLimit = memoryLimit
        self.initialEpsilon = initialEpsilon
        self.finalEpsilon = finalEpsilon
        self.epsilonSteps = epsilonSteps if epsilonSteps is not None else min(trainingEpisodes * 0.2, 100000)
        self.epsilon = initialEpsilon
        self.minExperience = minExperience
        self.learningRate = learningRate
        self.useExperienceReplay = useExperienceReplay
        self.replayFile = replayFile

        print("Loading replay data...")
        self.replayMemory = shelve.open(replayFile).values() if replayFile is not None else []

    def sampleReplayBatch(self):
        """
        Take a random batch of self.batchSize of transitions from replay memory
        """
        return random.sample(self.replayMemory, self.batchSize)

    def beginTraining(self, qFunctionManager):
        """
        Begin the training
        :param qFunctionManager: The manager to use as Q function
        """
        import Queue
        self._queue = Queue.Queue()
        self.qFuncManager = qFunctionManager
        self._train()

    def _train(self):
        startTime = time.time()
        print("Beginning " + str(self.trainingEpisodes) + " training episodes")

        self.stats = util.Stats(isOffline=self.replayFile is not None,
                                discount=self.discount,
                                trainingEpisodes=self.trainingEpisodes,
                                minExperiences=self.minExperience,
                                learningRate=self.learningRate,
                                featuresExtractor=self.featuresExtractor,
                                initialEpsilon=self.initialEpsilon,
                                finalEpsilon=self.finalEpsilon,
                                batchSize=self.batchSize,
                                epsilonSteps=self.epsilonSteps,
                                useExperienceReplay=self.useExperienceReplay,
                                notes=self.qFuncManager.getStatsNotes())

        print("Collecting minimum experience before training...")

        # # games, agents, displays, rules
        # gamesInfo = [self.makeGame(displayActive=False) for _ in range(2)]
        # currentStates = [g[0].state for g in gamesInfo]

        game = self.makeGame(displayActive=False)
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
            newState = util.getSuccessor(game.agents, game.display, currentState, action)
            reward = newState.getScore() - currentState.getScore()
            currentState = newState

            self.replayMemory.append((currentState, action, reward, newState))

            if abs(reward)>1:
                for _ in range(4 if abs(reward)<=20 else 10):
                    self.replayMemory.append((currentState, action, reward, newState))

            rewardSum += reward

            if len(self.replayMemory) > self.memoryLimit:
                self.replayMemory.pop(0)

            if newState.isWin() or newState.isLose():
                game = self.makeGame(displayActive=False)
                currentState = game.state

                wins += 1 if newState.isWin() else 0
                games += 1
                deaths += 1 if newState.isLose() else 0

            if len(self.replayMemory) < self.minExperience:
                continue

            # Take and convert a batch from replay memory
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

                self.stats.record([episodes, averageLoss, averageAccuracy, rewardSum, self.epsilon, wins, deaths])
                trainingLossSum = 0
                rewardSum = 0
                accuracySum = 0
                deaths = 0

                try:
                    self.qFuncManager.saveCheckpoint(self.stats.fileName + ".chkpt")
                except:
                    pass

            if episodes % 100 == 0:
                pass
                #self._queue.put(lambda: self.playOnce(displayActive=True))

            episodes += 1

            if not self.useExperienceReplay:
                self.replayMemory = []

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
        """
        Play one game with what we have learned so far. Actions are taken with an epsilon of 0
        :param displayActive: True or False to indicate if the display should be active
        :return: The score achieved in the game after winning or loosing.
        """

        game = self.makeGame(displayActive=displayActive)
        currentState = game.state
        game.display.initialize(currentState.data)

        while not (currentState.isWin() or currentState.isLose()):
            action = self.qFuncManager.getAction(currentState, epsilon=0)
            currentState = util.getSuccessor(game.agents, game.display, currentState, action)

        return currentState.getScore()

    def makeGame(self, displayActive):
        """
        Make a game
        :param displayActive: True or False to indicate if the display should be active
        :return: game, agents, display, rules
        """

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

        return game

if __name__ == '__main__':

    trainingRoom = TrainingRoom(layoutName="mediumGrid",
                                trainingEpisodes=10000,
                                replayFile=None,#"./training files/replayMem_mediumClassic.txt",
                                batchSize=600,
                                discount=0.95,
                                minExperience=600,
                                learningRate=0.2,
                                featuresExtractor=DistancesExtractor(),
                                initialEpsilon=1,
                                finalEpsilon=0.05,
                                useExperienceReplay=True)

    trainingRoom.beginTraining(qFunctionManagers.NNQFunctionManager(trainingRoom))
    checkPointFile = "./training files/training stats/1477957563.chkpt"