import shelve

import ghostAgents
import layout
import pacmanAgents
import qFunctionManagers
import featureExtractors
import util
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
        self.epsilonSteps = epsilonSteps if epsilonSteps is not None else min(trainingEpisodes * 0.6, 100000)
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

        if isinstance(self.qFuncManager, qFunctionManagers.NNQFunctionManager):
            self.replayMemory = [(self.featuresExtractor.getFeatures(state=s[0], action=s[1]),
                                s[1], s[2],
                                self.featuresExtractor.getFeatures(state=s[3], action=None),
                                s[3].isWin() or s[3].isLose(),
                                s[3].getLegalActions()) for s in self.replayMemory]

        startTime = time.time()
        print("Beginning " + str(self.trainingEpisodes) + " training episodes")

        self.stats = util.Stats(replayFile=self.replayFile,
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

        totalWins = 0
        totalDeaths = 0

        lastEpisodesRewardSum = 0
        lastEpisodesDeaths = 0
        lastEpisodesWins = 0

        while episodes < self.trainingEpisodes:

            # Update replay memory
            action = self.qFuncManager.getAction(currentState, epsilon=self.epsilon)
            newState = util.getSuccessor(game.agents, game.display, currentState, action)
            reward = newState.getScore() - currentState.getScore()

            if isinstance(self.qFuncManager, qFunctionManagers.NNQFunctionManager):
                qState = self.featuresExtractor.getFeatures(state=currentState, action=None)
                newQState = self.featuresExtractor.getFeatures(state=newState, action=None)
                experience = (qState, action, reward, newQState, newState.isWin() or newState.isLose(), newState.getLegalActions())

            else:
                experience = (currentState, action, reward, newState)

            self.replayMemory.append(experience)

            if abs(reward)>1:
                for _ in range(4 if abs(reward)<=20 else 10):
                    self.replayMemory.append(experience)

            currentState = newState

            lastEpisodesRewardSum += reward

            if len(self.replayMemory) > self.memoryLimit:
                self.replayMemory.pop(0)

            if newState.isWin() or newState.isLose():
                game = self.makeGame(displayActive=False)
                currentState = game.state

                if newState.isWin():
                    totalWins += 1
                else:
                    totalDeaths += 1
                    lastEpisodesDeaths += 1

            if len(self.replayMemory) < self.minExperience:
                continue

            # Take and convert a batch from replay memory
            batch = self.sampleReplayBatch()
            loss, accuracy = self.qFuncManager.update(batch)
            trainingLossSum += loss
            accuracySum += accuracy

            self.epsilon = max(self.finalEpsilon, 1.00 - float(episodes) / float(self.epsilonSteps))

            if episodes % 100 == 0 and episodes != 0:

                averageLoss = trainingLossSum/20.0
                averageAccuracy = accuracySum/20.0

                print("______________________________________________________")
                print("Episode: " + str(episodes))
                print("Average training loss: " + str(averageLoss))
                print("Average accuracy: " + str(averageAccuracy))
                print("Total reward for last episodes: " + str(lastEpisodesRewardSum))
                print("Epsilon: " + str(self.epsilon))
                print("Total wins: " + str(totalWins))
                print("Total deaths: " + str(totalDeaths))
                print("Deaths during last episodes: " + str(lastEpisodesDeaths))
                print("Wins during last episodes: " + str(lastEpisodesWins))

                self.stats.record([episodes, averageLoss, averageAccuracy, lastEpisodesRewardSum, self.epsilon, totalWins, totalDeaths, lastEpisodesDeaths, lastEpisodesWins])

                trainingLossSum = 0
                accuracySum = 0

                lastEpisodesRewardSum = 0
                lastEpisodesDeaths = 0
                lastEpisodesWins = 0

                try:
                    self.qFuncManager.saveCheckpoint(self.stats.fileName + ".chkpt")
                except:
                    pass

            # if episodes % 100 == 0:
            #     self._queue.put(lambda: self.playOnce(displayActive=True))

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
                                trainingEpisodes=60000,
                                replayFile="./training files/replayMem_mediumGrid.txt",
                                batchSize=600,
                                discount=0.95,
                                minExperience=600,
                                learningRate=0.2,
                                featuresExtractor=featureExtractors.CompletePositionsExtractor(),
                                initialEpsilon=1,
                                finalEpsilon=0.05,
                                useExperienceReplay=True)

    trainingRoom.beginTraining(qFunctionManagers.NNQFunctionManager(trainingRoom))
    #checkPointFile = "./training files/training stats/1477957563.chkpt"
