import shelve


class ExperienceReplayHelper:
    def __init__(self, identifier):
        self.identifier = identifier
        self._fileName = "replayMem_" + self.identifier + ".txt"

        self.replayMemory = shelve.open(self._fileName)

    def remember(self, state, action, reward, nextState):
        from game import Directions
        self.replayMemory[str(state.__hash__()) + str(Directions.getIndex(action))] = (state, action, reward, nextState)

    def persist(self):
        self.replayMemory.sync()
        print("PERSISTING DATA")

    def sampleBatch(self, size):
        pass

    def buildExperience(self, layoutName, displayActive=False):

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

        pendingStates = [initialState]
        counter = 0

        while pendingStates:
            counter += 1

            pendingState = pendingStates.pop(0)

            for action in pendingState.getLegalActions():
                if action == Directions.STOP: continue

                def logStateAndUpdateDisplay(newState):
                    display.update(newState.data)
                    reward = newState.data.score - pendingState.data.score
                    self.remember(pendingState, action, reward, newState)
                    print("Saw state. Took action: " + action + ". Received reward " + str(reward))

                try:
                    # Execute the action
                    newState = pendingState.generateSuccessor(0, action)
                    logStateAndUpdateDisplay(newState)

                    newState = newState.generateSuccessor(1, agents[1].getAction(newState))
                    logStateAndUpdateDisplay(newState)

                    newState = newState.generateSuccessor(2, agents[2].getAction(newState))
                    logStateAndUpdateDisplay(newState)

                    if not (newState.isWin() or newState.isLose()):
                        pendingStates.append(newState)

                except Exception, e:
                    print(e)

            if counter % 50 == 0:
                self.persist()

        display.finish()
        self.persist()
        self.replayMemory.close()


if __name__ == '__main__':
    ExperienceReplayHelper("smallGrid").buildExperience(layoutName="smallClassic", displayActive=True)
