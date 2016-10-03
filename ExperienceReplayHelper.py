import json


class ExpereienceReplayHelper:
    def __init__(self, identifier):
        self.identifier = identifier
        self._fileName = "replayMem_" + self.identifier + ".txt"

        try:
            self.replayMemory = json.load(open(self._fileName))
        except:
            self.replayMemory = {}

    def remember(self, state, action, reward, nextState):
        from game import Directions
        self.replayMemory[(state, action)] = (state, action, reward, nextState)

    def persist(self):
        json.dump(self.replayMemory, open(self._fileName, 'w'))

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

        while pendingStates:
            pendingState = pendingStates.pop(0)

            for action in pendingState.getLegalActions():
                if action == Directions.STOP: continue

                # Execute the action
                try:
                    newState = pendingState.generateSuccessor(0, action)
                    display.update(newState.data)

                    newState = newState.generateSuccessor(1, agents[1].getAction(newState))
                    display.update(newState.data)

                    newState = newState.generateSuccessor(2, agents[2].getAction(newState))
                    display.update(newState.data)

                    pendingStates.append(newState)

                    reward = newState.data.score - pendingState.data.score
                    print("Enqueued state. Took action: " + action + ". Received reward " + str(reward))

                    self.remember(pendingState, action, reward, newState)
                    self.persist()

                except Exception,e:
                    pass

        display.finish()
        self.persist()


if __name__ == '__main__':
    ExpereienceReplayHelper("smallGrid").buildExperience(layoutName="smallClassic", displayActive=True)
