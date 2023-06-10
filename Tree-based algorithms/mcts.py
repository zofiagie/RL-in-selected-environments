import random
import math
from game_state import RandomAgent


class MCTSNode(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {-1: 0, 1: 0}
        self.num_rollouts = 0
        self.children = []
        self.left_moves = game_state.legal_moves()

    def add_random_child(self):
        "Adding next child to the MCTS tree"
        index = random.randint(0, len(self.left_moves)-1)
        new_move = self.left_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def winner_tracker(self, winner):
        "Tracker of winning games after selecting move"
        if winner is not None:
            self.win_counts[winner] += 1
        else:
            self.win_counts[1] += 0.1    # drawing > lossing
            self.win_counts[-1] += 0.1
        self.num_rollouts += 1

    def is_final_move(self):
        return self.game_state.is_over()

    def any_moves_left(self):
        return len(self.left_moves) > 0

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSAgent():
    "Implemenation of MCTS algorithm"
    def __init__(self, num_rounds, c):
        self.num_rounds = num_rounds
        self.c = c

    def select_move(self, game_state):
        "Selecting move using MCTS"
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root
            while (not node.any_moves_left()) and (not node.is_final_move()):
                node = self.select_child(node)

            if node.any_moves_left():
                node = node.add_random_child()

            winner = self.simulate_random_game(node.game_state)

            while node is not None:
                node.winner_tracker(winner)
                node = node.parent

        best_move, best_pct = None, -1
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        return best_move

    def select_child(self, node):
        "Selection of a child using the UCT formula"
        best_score = -1
        best_child = None
        for child in node.children:
            w = child.winning_frac(node.game_state.next_player)
            uct_score = w + self.c * math.sqrt(math.log(sum(child.num_rollouts for child in node.children)) / child.num_rollouts)
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    def simulate_random_game(self, game):
        "Simulation of random game needed"
        ra = RandomAgent()
        while not game.is_over():
            move = ra.select_move(game)
            game = game.apply_move(move)
        return game.winner()