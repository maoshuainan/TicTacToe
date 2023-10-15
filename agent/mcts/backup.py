'''
Monte Carlo Tree Search algorithm for tictactoe.

'''

from game.TicTacToeLogic import Board
from game.TicTacToeGame import TicTacToeGame
from collections import defaultdict
import numpy as np

def monteCarloTreeSearch(game:TicTacToeGame, state, player = 1, simulations_number = 1000):
    root = MonteCarloTree(game, state, player)
    for _ in range(0, simulations_number):
        # 1. Selection &  2. Expansion
        v = tree_policy(root)
        # 3. Simulation
        reward = v.rollout()
        # 4. Backpropagation
        v.backpropagate(reward)
    # exploitation only
    return root.best_child(c_param=0.).action

def tree_policy(root):
    current_node = root
    while not current_node.is_terminal_node():
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
    return current_node

class MonteCarloTree:
    def __init__(self, game:TicTacToeGame, state, player, action=None, parent=None):
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        
        self.game = game
        self.board = Board(game.n)
        self.board.pieces = np.copy(state)
        self.player = player
        # action index number
        self.action = action

        self.parent = parent
        self.children = []
        
    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = []
            valids = self.game.getValidMoves(self.board.pieces, self.player)
            for action in range(len(valids)-1):
                if(valids[action]):
                    self._untried_actions.append(action)
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[self.player]
        loses = self._results[-1 * self.player]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_board = Board(self.game.n)
        next_board.pieces = np.copy(self.board.pieces)
        move = (int(action / self.game.n), action % self.game.n)
        next_board.execute_move(move, self.player)
        child_node = MonteCarloTree(self.game, next_board.pieces,-self.player, action=action, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.game.getGameEnded(self.board.pieces, self.player)

    def rollout(self):
        current_rollout_state = self.board.pieces
        player = self.player
        while self.game.getGameEnded(current_rollout_state, player) == 0:
            valids = self.game.getValidMoves(current_rollout_state, player)
            possible_moves = []
            for action in range(len(valids)-1):
                if(valids[action]):
                    possible_moves.append(action)
            action = self.rollout_policy(possible_moves)
            current_rollout_state, player = self.game.getNextState(current_rollout_state, player, action)
        reward = self.game.getGameEnded(current_rollout_state, self.player)
        if reward is not 1 and reward is not -1 :
            reward = 0
        return reward

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def __repr__(self) -> str:
        s = "action: {}, stats: {} / {}, q: {}".format(
            self.action, self._results[self.player], self._number_of_visits, self.q
        )
        return s