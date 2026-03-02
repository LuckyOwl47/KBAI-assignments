from heuristics import Heuristic, SimpleHeuristic, SuperDuperHeuristic
from players import PlayerController, HumanPlayer, MinMaxPlayer, AlphaBetaPlayer
from board import Board
from typing import List
import numpy as np
from numba import jit
<<<<<<<<< Temporary merge branch 1
from time import perf_counter # Added runtime measure
=========
from heuristics import SimpleHeuristic
from time import perf_counter


>>>>>>>>> Temporary merge branch 2


def start_game(game_n: int, board: Board, players: List[PlayerController]) -> int:
    """Starting a game and handling the game logic

    Args:
        game_n (int): n in a row required to win
        board (Board): board to play on
        players (List[PlayerController]): players of the game

    Returns:
        int: id of the winning player, or -1 if the game ends in a draw
    """
    print('Start game!')
    current_player_index: int = 0 # index of the current player in the players list
    winner: int = 0

    p_time = {players[0].player_id: 0.0, players[1].player_id: 0.0} # Added player runtime variable

    # Main game loop
    while winner == 0:
        current_player: PlayerController = players[current_player_index]
        t0 = perf_counter() # Added counter
        move: int = current_player.make_move(board)
        p_time[current_player.player_id] += perf_counter() - t0 # added 

        while not board.play(move, current_player.player_id):
            move = current_player.make_move(board)

        current_player_index = 1 - current_player_index
        winner = winning(board.get_board_state(), game_n)

    # Printing out winner, final board and number of evaluations after the game 
    print(board)

    if winner < 0:
        print('Game is a draw!')
    else:
        print(f'Player {current_player} won!')

    for p in players:
        print(f'Player {p} evaluated a boardstate {p.get_eval_count()} times!')

    for p in players:
        print(f'Player {p} total move time: {p_time[p.player_id]:.3f}s') # Added a print the runtime


@jit(nopython=True, cache=True)
def winning(state: np.ndarray, game_n: int) -> int:
    """Determines whether a player has won, and if so, which one

    Args:
        state (np.ndarray): the board to check
        game_n (int): n in a row required to win

    Returns:
        int: 1 or 2 if the respective player won, -1 if the game is a draw, 0 otherwise
    """
    player: int
    counter: int

    # Vertical check
    for col in state:
        counter = 0
        player = -1
        for field in col[::-1]:
            if field == 0:
                break
            elif field == player:
                counter += 1
                if counter >= game_n:
                    return player
            else:
                counter = 1 
                player = field
            
    # Horizintal check
    for row in state.T:
        counter = 0
        player = -1
        for field in row:
            if field == 0:
                counter = 0
                player = -1
            elif field == player:
                counter += 1
                if counter >= game_n:
                    return player
            else:
                counter = 1
                player = field

    # Ascending diagonal check
    for i, col in enumerate(state[:- game_n + 1]):
        for j, field in enumerate(col[game_n - 1:]):
            if field == 0:
                continue
            player = field
            for x in range(game_n):
                if state[i + x, j + game_n - 1 - x] != player:
                    player = -1
                    break
            if player != -1:
                return player
            
    # Descending diagonal check
    for i, col in enumerate(state[game_n - 1:]):
        for j, field in enumerate(col[game_n - 1:]):
            if field == 0:
                continue
            player = field
            for x in range(game_n):
                if state[i + game_n - 1 - x, j + game_n - 1 - x] != player:
                    player = -1
                    break
            if player != -1:
                return player
        
    # Check for a draw
    if np.all(state[:, 0]):
        return -1 # The board is full, game is a draw

    return 0 # Game is not over 
    

def get_players(game_n: int, depth_player_1: int, depth_player_2: int) -> List[PlayerController]:
=========
    return 0 # Game is not over '''

'''  
def get_players(game_n: int) -> list[PlayerController]:
    # unique heuristics per player
    h1: Heuristic = SimpleHeuristic(game_n)
    h2: Heuristic = SimpleHeuristic(game_n)

    # pick exactly two players with distinct player_id
    alpha = AlphaBetaPlayer(player_id=1, game_n=game_n, depth=1, heuristic=h1)
    mcts  = MonteCarloPlayer(player_id=2, game_n=game_n, rollouts=5000, heuristic=h2, exploration=1.41, time_limit_ms=500)

    players = [alpha, mcts]

    # sanity checks
    assert players[0].player_id in {1, 2}
    assert players[1].player_id in {1, 2}
    assert players[0].player_id != players[1].player_id, 'The players must have an unique player_id'
    assert players[0].heuristic is not players[1].heuristic, 'The players must have an unique heuristic'
    assert len(players) == 2
    return players


'''
def get_players(game_n: int) -> List[PlayerController]:
>>>>>>>>> Temporary merge branch 2
    """Gets the two players

    Args:
        game_n (int): n in a row required to win

    Raises:
        AssertionError: if the players are incorrectly initialised

    Returns:
        List[PlayerController]: list with two players
        """
    
    ### if a human wants to play, uncomment this ###
    # human1: PlayerController = HumanPlayer(1, game_n, heuristic1)
    heuristic1: Heuristic = SuperDuperHeuristic(game_n)
    heuristic2: Heuristic = SuperDuperHeuristic(game_n)

    min_max: PlayerController = MinMaxPlayer(player_id=1, game_n=game_n, depth=depth_player_1, heuristic=heuristic2)
    alpha_beta = AlphaBetaPlayer(player_id=2, game_n=game_n, depth=depth_player_2, heuristic=heuristic1)

<<<<<<<<< Temporary merge branch 1
    # TODO: Implement other PlayerControllers (MinMaxPlayer and AlphaBetaPlayer)

    players: List[PlayerController] = [min_max, alpha_beta]
=========

    h = SimpleHeuristic(game_n=4)
    alpha_beta = AlphaBetaPlayer(player_id=1, game_n=4, depth=5, heuristic=heuristic1)
    min_max: PlayerController = MinMaxPlayer(1, game_n, depth=3, heuristic=heuristic2)
    bot = Bot(player_id=2, game_n=game_n, heuristic=heuristic3)
    # TODO: Implement other PlayerControllers (MinMaxPlayer and AlphaBetaPlayer)

    players: List[PlayerController] = [alpha_beta, bot]
>>>>>>>>> Temporary merge branch 2

    assert players[0].player_id in {1, 2}, 'The player_id of the first player must be either 1 or 2'
    assert players[1].player_id in {1, 2}, 'The player_id of the second player must be either 1 or 2'
    assert players[0].player_id != players[1].player_id, 'The players must have an unique player_id'
    assert players[0].heuristic is not players[1].heuristic, 'The players must have an unique heuristic'
    assert len(players) == 2, 'Not the correct amount of players'

    return players


if __name__ == '__main__':
    game_n: int = 4 # n in a row required to win
<<<<<<<<< Temporary merge branch 1
    depth_player_1: int = 3 # added depth parameters
    depth_player_2: int = 3 
    width: int = 7  # width of the board
    height: int = 7 # height of the board
=========
    width: int = 7  # width of pthe board
    height: int = 6 # height of the board
    depth: int = 1 # depth for MinMax and AlphaBeta
>>>>>>>>> Temporary merge branch 2

    # Check whether the game_n is possible
    assert 1 < game_n <= min(width, height), 'game_n is not possible'

    board: Board = Board(width, height)
    
    start_game(game_n, board, get_players(game_n, depth_player_1, depth_player_2))