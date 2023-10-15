'''
Minimax algorithm for tictactoe.

Params:
    1. game - TicTacToeGame object.
    2. state - 2d numpy array contains board pieces.
    3. depth - the number of tree layers.
    4. player - 1 for AI, -1 for others.

Return:
    1. action - best action index number in valid moves.
    2. price - the price of the best action for the state.

'''

import numpy as np
from game.TicTacToeLogic import Board

def minimax(game, state, depth, player):
    board = Board(game.n)
    board.pieces = np.copy(state)
    if depth == 0 or evaluate(board) != 0:
        return (-1,-1), evaluate(board)
    valids = game.getValidMoves(state, player)
    if(valids[-1]):
        return (-1,-1), 0
    pricelist = {}
    for action in range(len(valids)-1):
        if valids[action]:
            newboard = Board(game.n)
            newboard.pieces = np.copy(state)
            move = (int(action / game.n), action % game.n)
            newboard.execute_move(move, player)
            a, pricelist[action] = minimax(game, newboard.pieces, depth-1, -player)
    # print(state)
    # print(pricelist)
    if (player == 1):
        a = max(pricelist, key=lambda x:pricelist[x])
    else:
        a = min(pricelist, key=lambda x:pricelist[x])
    
    return a, pricelist[a]
        
def evaluate(board:Board):
    if(board.is_win(1)):
        return 1
    elif(board.is_win(-1)):
        return -1
    else:
        return 0