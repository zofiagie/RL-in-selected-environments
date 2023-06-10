import numpy as np
import copy
import random


class Board:
    """Board representation for tic-tac-toe"""
    def __init__(self):
        self.size = 3
        self.grid = np.zeros((3, 3))

    def place(self, player, point):
        "A function that place move on the board"
        assert self.is_on_grid(point)
        assert self.empty(point)
        self.grid[point] = player

    def is_on_grid(self, point):
        return (0 <= point[0] <= self.size - 1) and (0 <= point[1] <= self.size - 1)

    def empty(self, point):
        return self.grid[point] == 0

class GameState:
    """The class that checks the current state of the game"""
    def __init__(self, board, next_player):
        self.board = board
        self.next_player = next_player

    def apply_move(self, move):
        "The function that applies a move and changes the next player"
        next_board = copy.deepcopy(self.board)
        next_board.place(self.next_player, move)
        return GameState(next_board, -self.next_player)

    @classmethod
    def new_game(cls):
        board = Board()
        return GameState(board, 1)

    def legal_moves(self):
        if self.is_over():
            return []
        moves = []
        for row in range(self.board.size):
            for col in range(self.board.size):
                move = (row, col)
                if self.board.is_on_grid(move) and self.board.empty(move):
                    moves.append(move)
        return moves

    def has_3_in_a_row(self, player):
        "The function checks if the selected player won the game"
        for col in range(3):
            if all(self.board.grid[row][col] == player for row in range(3)):
                return True
        for row in range(3):
            if all(self.board.grid[row][col] == player for col in range(3)):
                return True
        if self.board.grid[0, 0] == player and self.board.grid[1, 1] == player and self.board.grid[2, 2] == player:
            return True
        if self.board.grid[2, 0] == player and self.board.grid[1, 1] == player and self.board.grid[0, 2] == player:
            return True

    def is_over(self):
        "The function that checks whether the game has already ended"
        if self.has_3_in_a_row(1):
            return True
        if self.has_3_in_a_row(-1):
            return True
        if np.count_nonzero(self.board.grid == 0) == 0:
            return True
        return False

    def winner(self):
        "The function that returns the winner"
        if self.has_3_in_a_row(1):
            return 1
        if self.has_3_in_a_row(-1):
            return -1
        return None


class RandomAgent():
    "The class that simulates a random game"
    def select_move(self, game_state):
        return random.choice(game_state.legal_moves())