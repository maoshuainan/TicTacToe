from game.TicTacToeLogic import Board
from game.TicTacToeGame import TicTacToeGame
from collections import defaultdict
import numpy as np

def monteCarloTreeSearch(game:TicTacToeGame, state, player = 1, simulations_number = 1000):
    root = MonteCarloTree(game, state, player)
    for _ in range(simulations_number):
        node = root.select()
        if not node.game.getGameEnded(node.state, node.player) == 0:
            result = node.rollout()
        else :
            result = node.game.getGameEnded(node.state, node.player)
        node.backpropagate(result)
    return root.best_child(c_param=0.).action

class MonteCarloTree:
    """蒙特卡洛树
    属性：
    1. 模拟次数。
    2. 模拟结果。
    3. 总的模拟结果。
    4. 子节点数组.
    方法：
    1. 选择
    2. 扩展
    3. 模拟
    4. 反向传播
    
    
    """

    def __init__(self, game:TicTacToeGame, state, player, parent = None, action=None):
        self._num_of_visits=0
        self._results=defaultdict(int)
        
        self.game=game
        self.state=state
        self.action=action
        self.player=player
        
        self.parent = parent
        self.children=[]
    
    @property
    def q(self):
        wins = self._results[self.player]
        loses = self._results[-self.player]
        return wins-loses
    
    @property
    def n(self):
        return self._num_of_visits
    
    @property
    def is_terminal_node(self):
        if self.game.getGameEnded(self.state, self.player):
            return True
        return False
    
    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = []
            valids = self.game.getValidMoves(self.state, self.player)
            for action in range(len(valids)-1):
                if(valids[action]):
                    self._untried_actions.append(action)
        return self._untried_actions
    
    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def select(self):
        current_node=self
        while not current_node.is_terminal_node:
            if not current_node.is_fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    def expand(self):
        action = self.untried_actions.pop()
        state = np.copy(self.state)
        state, player = self.game.getNextState(state, self.player, action)
        node = MonteCarloTree(self.game, state, player, self, action)
        self.children.append(node)
        return node
    
    def rollout(self):
        state = np.copy(self.state)
        player = self.player
        while self.game.getGameEnded(state, player)==0:
            valids = self.game.getValidMoves(state, player)
            possible_moves = []
            for action in range(len(valids)-1):
                if(valids[action]):
                    possible_moves.append(action)
            action = self.rollout_policy(possible_moves)
            state, player = self.game(state, player, action)
        result = self.game.getGameEnded(state, self.player)
        if result is not 1 and result is not -1 :
            result = 0
        return result
    
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def backpropagate(self, result):
        self._num_of_visits += 1
        self._results[result] += 1
        if self.parent :
            self.parent.backpropagate(result)
        
    
    def best_child(self, c_param = 1.4):
        choices_weight = [ ((c.q) / (c.n)) + c_param*np.sqrt((2 * np.log(self.n) / (c.n))) for c in self.children]
        return self.children[np.argmax(choices_weight)]