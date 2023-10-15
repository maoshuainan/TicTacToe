from game.TicTacToeGame import TicTacToeGame
from agent.TicTacToePlayers import *
from Arena import Arena

def main():
    g = TicTacToeGame()
    p1 = MCTSPlayer(g).play
    p2 = RandomPlayer(g).play

    arena = Arena(p1, p2, g, display=TicTacToeGame.display)

    print(arena.playGames(2, verbose=True))

if __name__ == "__main__":
    main()
