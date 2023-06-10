from game_state import GameState
from min_max import MinMaxAgent
from alfa_beta_pruning import AlphaBetaAgent
from mcts import MCTSAgent



def point_from_coords(text):
    point = text.split(',')
    return int(point[0]), int(point[1])


def play(bot_agent):
    game = GameState.new_game()

    human_player = 1

    while not game.is_over():
        print('Player: ', game.next_player)
        if game.next_player == human_player:
            human_move = input('x, y: ')
            move_x, move_y = point_from_coords(human_move)
        else:
            move = bot_agent.select_move(game)
            move_x, move_y = move[0], move[1]
        game = game.apply_move((move_x, move_y))

        print(game.board.grid)
    winner = game.winner()
    if winner is None:
        print("It's a draw.")
    else:
        print('Winner: ', winner)


if __name__ == '__main__':
    print('Playing with minmax algorithm')
    play(MinMaxAgent())
    print('Playing with minmax algorithm with alpha-beta pruning')
    play(AlphaBetaAgent(-1))
    print('Playing with MCTS algorithm')
    play(MCTSAgent(1000, 1.5))

