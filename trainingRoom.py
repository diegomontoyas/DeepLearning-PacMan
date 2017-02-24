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

    def __init__(self, featuresExtractor, layoutName, trainingEpisodes, replayFile, initialEpsilon,
                 finalEpsilon, discount=0.8, batchSize=32, useExperienceReplay=True,
                 memoryLimit=50000, epsilonSteps=None, minExperience=5000):
        """
        Initialize a training room.

        :param featuresExtractor: An instance of the features extractor to use. Valid instances are subclasses of
            `FeatureExtractor` and usually come from the `featureExtractors.py` module.

        :param layoutName: The name of the PacMan layout to use

        :param trainingEpisodes: The number of episodes to train for. One episode means training one batch of
            experience replay memory.

        :param replayFile: The path to the file of a pre-calculated replay memory. To generate one of these files
            we use the `experienceReplayHelper.py` module.

        :param initialEpsilon: The initial value of epsilon.
        :param finalEpsilon: The final vallue of epsilon.

        :param discount: The discount factor
        :param batchSize: The size of the batch to take from replay memory (if useExperienceReplay==True)

        :param useExperienceReplay: If the experience replay memory should be used. If False, the memory is emptied
            after it reaches a size of `batchSize`.

        :param memoryLimit: The size limit for the replay memory. After this limit is reached new transitions
            start to replace old transitions.

        :param epsilonSteps: The amount of steps it takes for epsilon to go from `initialEpsilon` to `finalEpsilon`.

        :param minExperience: The minimum amount of transitions that are necessary to collect before one
            episode of training can take place.

        """

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
        self.minExperience = minExperience if useExperienceReplay else batchSize
        self.useExperienceReplay = useExperienceReplay
        self.replayFile = replayFile
        self.replayMemory = []

    def beginTraining(self, qFunctionManager):
        """
        Begins the training
        :param qFunctionManager: The manager to use as Q function
        """
        self.qFuncManager = qFunctionManager
        self._prepareReplayMemory()

        # Initialize stats module
        self.stats = util.Stats(replayFile=self.replayFile,
                                discount=self.discount,
                                trainingEpisodes=self.trainingEpisodes,
                                minExperiences=self.minExperience,
                                featuresExtractor=self.featuresExtractor,
                                initialEpsilon=self.initialEpsilon,
                                finalEpsilon=self.finalEpsilon,
                                batchSize=self.batchSize,
                                epsilonSteps=self.epsilonSteps,
                                useExperienceReplay=self.useExperienceReplay,
                                notes=self.qFuncManager.getStatsNotes())

        self._train()

    def _sampleReplayBatch(self):
        """
        Take a random batch of self.batchSize of transitions from replay memory
        :return: A random list of transitions (tuples)
        """
        return random.sample(self.replayMemory, self.batchSize)

    def _prepareReplayMemory(self):
        """
        Prepares the replay memory by loading any replay data from the replay file and pre-calculating information
         to improve training time.
        """
        print("Loading replay data from replay file...")
        self.replayMemory = shelve.open(self.replayFile).values() if self.replayFile is not None else []

        # Optimize training time by pre-calculating the representations for all states in replay memory
        if isinstance(self.qFuncManager, qFunctionManagers.NNQFunctionManager):

            self.replayMemory = [
                (self.featuresExtractor.getFeatures(state=s[0], action=s[1]),
                 s[1], s[2],
                 self.featuresExtractor.getFeatures(state=s[3], action=None),
                 s[3].isWin() or s[3].isLose(),
                 s[3].getLegalActions())

                for s in self.replayMemory]

    def _train(self):
        startTime = time.time()
        print("Beginning " + str(self.trainingEpisodes) + " training episodes")
        print("Collecting minimum experience before training...")

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

            # Get an action
            action = self.qFuncManager.getAction(currentState, epsilon=self.epsilon)

            # Execute it
            newState = util.getSuccessor(game.agents, game.display, currentState, action)

            # Find the reward
            reward = newState.getScore() - currentState.getScore()

            if isinstance(self.qFuncManager, qFunctionManagers.NNQFunctionManager):
                # If we have a NNQFunctionManager we need to pre-calculate the processed state representations

                qState = self.featuresExtractor.getFeatures(state=currentState, action=None)
                newQState = self.featuresExtractor.getFeatures(state=newState, action=None)
                experience = (
                    qState, action, reward, newQState,
                    newState.isWin() or newState.isLose(),
                    newState.getLegalActions())
            else:
                experience = (currentState, action, reward, newState)

            # Update replay memory
            self.replayMemory.append(experience)

            if len(self.replayMemory) > self.memoryLimit:
                self.replayMemory.pop(0)

            # Test: Prioritize relevant experiences a bit
            # if abs(reward) > 1:
            #     for _ in range(4 if abs(reward) <= 20 else 10):
            #         self.replayMemory.append(experience)

            currentState = newState
            lastEpisodesRewardSum += reward

            # If the game is over create a new one
            if newState.isWin() or newState.isLose():
                game = self.makeGame(displayActive=False)
                currentState = game.state

                if newState.isWin():
                    totalWins += 1
                else:
                    totalDeaths += 1
                    lastEpisodesDeaths += 1

            # If we don't have the minimum experience we can not continue
            if len(self.replayMemory) < self.minExperience:
                continue

            # Take a batch from replay memory and train
            batch = self._sampleReplayBatch()
            loss, accuracy = self.qFuncManager.update(batch)
            trainingLossSum += loss
            accuracySum += accuracy

            # Decrease epsilon
            self.epsilon = max(self.finalEpsilon, 1.00 - float(episodes) / float(self.epsilonSteps))

            # Bookkeeping
            if episodes % 100 == 0 and episodes != 0:
                self._recordCheckpointStats(accuracySum, episodes, lastEpisodesDeaths, lastEpisodesRewardSum,
                                            lastEpisodesWins, totalDeaths, totalWins, trainingLossSum)

                trainingLossSum = 0
                accuracySum = 0
                lastEpisodesRewardSum = 0
                lastEpisodesDeaths = 0
                lastEpisodesWins = 0

            episodes += 1

            # If we are not using experience replay, we clear the transitions we just experienced
            if not self.useExperienceReplay:
                self.replayMemory = []

        print("Finished training, turning off epsilon...")
        print("Calculating average score...")
        self._finishTrainingAndCalculateAvgScore(startTime)

    def _recordCheckpointStats(self, accuracySum, episodes, lastEpisodesDeaths, lastEpisodesRewardSum,
                               lastEpisodesWins, totalDeaths, totalWins, trainingLossSum):
        """
        Records statistics in the corresponding statistics file
        """
        averageLoss = trainingLossSum / 20.0
        averageAccuracy = accuracySum / 20.0

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

        self.stats.record(
            [episodes, averageLoss, averageAccuracy, lastEpisodesRewardSum, self.epsilon, totalWins,
             totalDeaths, lastEpisodesDeaths, lastEpisodesWins])

        try:
            self.qFuncManager.saveCheckpoint(self.stats.fileName + ".chkpt")
        except: pass

    def _finishTrainingAndCalculateAvgScore(self, startTime):
        """
        Calculate the average score using the results of the training a register the stats.
        :param startTime: The time at which the trainig began.
        """
        endTime = time.time()
        n = 0
        scoreSum = 0

        while True:
            scoreSum += self.playOnce(displayActive=n > 20)
            n += 1

            if n == 20:
                avg = scoreSum / 20.0
                self.stats.close(averageScore20Games=avg, learningTime=(endTime - startTime) / 60.0)
                print("Average score: " + str(avg))

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
        :return: A `Game` instance
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
        agents = [pacmanAgents.RandomAgent()] \
                 + [ghostAgents.DirectionalGhost(i + 1) for i in range(theLayout.getNumGhosts())]

        game = rules.newGame(theLayout, agents[0], agents[1:], display)

        return game

if __name__ == '__main__':
    trainingRoom = TrainingRoom(layoutName="mediumGrid",
                                trainingEpisodes=1000,
                                replayFile=None,  # "./training files/replayMem_mediumGrid.txt",
                                batchSize=600,
                                discount=0.95,
                                minExperience=600,
                                featuresExtractor=featureExtractors.CompletePositionsExtractor(),
                                initialEpsilon=1,
                                finalEpsilon=0.05,
                                useExperienceReplay=True)

    trainingRoom.beginTraining(qFunctionManagers.NNQFunctionManager(trainingRoom))