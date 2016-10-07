import shelve


class ExperienceReplayHelper:
    def __init__(self, file):
        self.file = file
        self.replayMemory = shelve.open(self.file)

    def remember(self, state, action, reward, nextState):
        from game import Directions
        self.replayMemory[str(state.__hash__()) + str(Directions.getIndex(action))] = (state, action, reward, nextState)

    def persist(self):
        self.replayMemory.sync()
        #print("PERSISTING DATA")

    def sampleBatch(self, size):
        pass

    def buildExperience(self, layoutName, displayActive=False, limit=None):

        import pacmanAgents, ghostAgents
        from pacman import ClassicGameRules
        from game import Directions
        import layout
        from pacman import GameState

        theLayout = layout.getLayout(layoutName)
        if theLayout == None: raise Exception("The layout " + layoutName + " cannot be found")

        display = None

        # Choose a display format
        if not displayActive:
            import textDisplay
            display = textDisplay.NullGraphics()
        else:
            import graphicsDisplay
            display = graphicsDisplay.PacmanGraphics(frameTime=0.01)

        rules = ClassicGameRules()
        agents = [pacmanAgents.GreedyAgent()] + [ghostAgents.DirectionalGhost(i + 1) for i in range(theLayout.getNumGhosts())]
        game = rules.newGame(theLayout, agents[0], agents[1:], display)
        initialState = game.state
        display.initialize(initialState.data)

        exploredStateHashes = {initialState.__hash__()}
        pendingStates = {initialState}
        counter = 0

        while pendingStates:
            pendingState = pendingStates.pop()

            for action in pendingState.getLegalActions():
                if action == Directions.STOP: continue

                def logStateAndUpdateDisplay(newState):
                    display.update(newState.data)
                    reward = newState.data.score - pendingState.data.score
                    self.remember(pendingState, action, reward, newState)
                    #print("Saw state. Took action: " + action + ". Received reward " + str(reward))

                try:
                    # Execute the action
                    newState = pendingState.generateSuccessor(0, action)
                    logStateAndUpdateDisplay(newState)

                    for ghostIndex in range(1, len(agents)):
                        newState = newState.generateSuccessor(ghostIndex, agents[ghostIndex].getAction(newState))
                        logStateAndUpdateDisplay(newState)

                    counter += 1

                    if not (newState.isWin() or newState.isLose()) and newState.__hash__() not in exploredStateHashes:
                        exploredStateHashes.add(newState.__hash__())
                        pendingStates.add(newState)

                except Exception, e:
                    #print(e)
                    pass

            if counter % 100 == 0:
                self.persist()

            if counter % 2000 == 0:
                print("Explored " + str(counter) + " states")

            if limit is not None and counter > limit:
                break

        display.finish()
        self.persist()
        self.replayMemory.close()
        print("Done")


    def buildRandomExperience(self, layoutName, displayActive=False, limit=None):
        import pacmanAgents, ghostAgents
        from pacman import ClassicGameRules
        from game import Directions
        import layout
        from pacman import GameState

        theLayout = layout.getLayout(layoutName)
        if theLayout == None: raise Exception("The layout " + layoutName + " cannot be found")

        display = None

        # Choose a display format
        if not displayActive:
            import textDisplay
            display = textDisplay.NullGraphics()
        else:
            import graphicsDisplay
            display = graphicsDisplay.PacmanGraphics(frameTime=0.01)

        counter = 0
        finished = False

        while not finished:
            rules = ClassicGameRules()
            agents = [pacmanAgents.RandomAgent()] + [ghostAgents.DirectionalGhost(i + 1) for i in range(theLayout.getNumGhosts())]

            game = rules.newGame(theLayout, agents[0], agents[1:], display)

            currentState = game.state
            display.initialize(currentState.data)

            def logStateAndUpdateDisplay(state, action, force):
                display.update(state.data)
                reward = state.data.score - currentState.data.score

                if force or not (state.isWin() or state.isLose()):
                    self.remember(currentState, action, reward, state)

            while not (currentState.isWin() or currentState.isLose()):
                action = agents[0].getAction(currentState)
                currentState = currentState.generateSuccessor(0, action)
                logStateAndUpdateDisplay(currentState, action, force=False)

                try:
                    for ghostIndex in range(1, len(agents)):
                        currentState = currentState.generateSuccessor(ghostIndex, agents[ghostIndex].getAction(currentState))
                        logStateAndUpdateDisplay(currentState, action, force= ghostIndex==len(agents)-1)
                except Exception, E:
                    pass

                counter += 1

                if counter % 100 == 0:
                    self.persist()

                if counter % 2000 == 0:
                    print("Explored " + str(counter) + " states")

            if limit is not None and counter > limit:
                finished = True

        display.finish()
        self.persist()
        self.replayMemory.close()
        print("Done")

if __name__ == '__main__':
    file = "/Users/Diego/Desktop/replayMem_smallGrid.txt"
    #ExperienceReplayHelper(file).buildRandomExperience(layoutName="smallGrid", displayActive=False)
    ExperienceReplayHelper("smallGrid").buildExperience(layoutName="smallGrid", displayActive=False)
