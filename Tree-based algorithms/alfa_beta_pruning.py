def get_score(game_state, player=1):
    if game_state.winner() == player:
        return 1
    elif game_state.winner() is None:
        return 0
    else:
        return -1


def alpha_beta_result(game_state, depth, alpha, beta, player):
    "Alpha-beta pruning implementation"
    row, col = -1, -1
    if game_state.is_over() or depth == 0:
        return [row, col, get_score(game_state)]

    else:
        for candidate_move in game_state.legal_moves():
            next_state = game_state.apply_move(candidate_move)
            score = alpha_beta_result(next_state, depth - 1, alpha, beta, -player)
            if player == 1:  # X is always the max player
                if score[2] > alpha:
                    alpha = score[2]
                    row = candidate_move[0]
                    col = candidate_move[1]
            else:
                if score[2] < beta:
                    beta = score[2]
                    row = candidate_move[0]
                    col = candidate_move[1]

            if alpha >= beta:
                break

        if player == 1:
            return [row, col, alpha]

        else:
            return [row, col, beta]


class AlphaBetaAgent():
    def __init__(self, player, depth='max'):
        self.player = player
        self.depth = depth

    def select_move(self, game_state):
        if self.depth == 'max':
            result = alpha_beta_result(game_state, len(game_state.legal_moves()), -9999, 9999, self.player)
        else:
            result = alpha_beta_result(game_state, self.depth, -9999, 9999, self.player)
        return result[0], result[1]
