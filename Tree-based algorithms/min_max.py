import random


def best_result(game_state):
    "Minmax implementation helper function. Used to create a tree"
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return 1
        elif game_state.winner() is None:
            return 0
        else:
            return -1
    best_score = -1
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)
        score = -best_result(next_state)
        if score > best_score:
            best_score = score
    return best_score


def minmax(game_state):
    "Implementation of the minmax algorithm"
    winning_moves = []
    draw_moves = []
    losing_moves = []
    for possible_move in game_state.legal_moves():  # Iteration over all possible moves
        next_state = game_state.apply_move(possible_move)
        score = -best_result(next_state)
        if score == 1:
            winning_moves.append(possible_move)
        elif score == 0:
            draw_moves.append(possible_move)
        else:
            losing_moves.append(possible_move)
    if winning_moves:
        return random.choice(winning_moves)
    if draw_moves:
        return random.choice(draw_moves)
    return random.choice(losing_moves)


class MinMaxAgent:
    def select_move(self, game_state):
        result = minmax(game_state)
        return result[0], result[1]

