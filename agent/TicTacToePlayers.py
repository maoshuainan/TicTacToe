"""
TicTacToe game players.

Each player should satisfy following requirement.

Assumption:
    1. All states are canonical. Donot transform the board again.
    2. Since all states are canonical, player should be consider as 1.

Params:
    1. game - Game state object.

Returns:
    1. action - An action index number for next step.

Author: Leo Mao, github.com/maoshuainan
Date: Aug 8, 2023.

Based on the OthelloPlayers by Surag Nair.

"""
import numpy as np
from utils import dotdict
from .minimax.Minimax import minimax
from .mcts.MonteCarloTreeSearch import monteCarloTreeSearch
from .alphazero.MCTS import MCTS
from .alphazero.pytorch.NNet import NNetWrapper as nn

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, state):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(state, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, state):
        # display(board)
        valid = self.game.getValidMoves(state, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a

class MinimaxPlayer():
    def __init__(self, game):
        self.game = game
    
    def play(self, state):
        depth = 10
        a,price = minimax(self.game, state, depth, 1)
        return a
    
class MCTSPlayer():
    def __init__(self, game):
        self.game = game
    
    def play(self, state):
        a = monteCarloTreeSearch(self.game, state, simulations_number=1000)
        return a

class AlphaZeroPlayer():
    def __init__(self, game):
        self.game = game
        self.args = dotdict({
            'numIters': 10,
            'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
            'tempThreshold': 15,        #
            'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
            'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
            'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
            'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
            'cpuct': 1,

            'checkpoint': './agent/alphazero/pytorch/',
            'load_model': False,
            'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
            'numItersForTrainExamplesHistory': 20,

        })
        self.nnet = nn(self.game)
        try:
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
        finally:
            print("No checkpoint file. Please train the net first.")
        self.mct = MCTS(game = self.game, nnet=self.nnet ,args = self.args)
    
    def play(self, state):
        a = np.argmax(self.mct.getActionProb(state, temp=0))
        return a