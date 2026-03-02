from heuristics import Heuristic, SimpleHeuristic
from players import Bot, PlayerController, HumanPlayer, MinMaxPlayer, AlphaBetaPlayer, MonteCarloPlayer
from board import Board
from typing import List
import numpy as np
from numba import jit
from heuristics import SimpleHeuristic
from time import perf_counter




def start_game(game_n: int, board: Board, players: List[PlayerController]) -> int:
    """Starting a game and handling the game logic."""
    print('Start game!')
    current_player_index: int = 0  # index of the current player
    winner: int = 0

    # --- timing accumulators per player id ---
    p_time = {players[0].player_id: 0.0, players[1].player_id: 0.0}
    p_moves = {players[0].player_id: 0,   players[1].player_id: 0}

    # Main game loop
    while winner == 0:
        current_player: PlayerController = players[current_player_index]

        # Time just the decision step
        t0 = perf_counter()
        move: int = current_player.make_move(board)
        dt = perf_counter() - t0

        # Accumulate timing stats
        pid = current_player.player_id
        p_time[pid] += dt
        p_moves[pid] += 1

        # Apply the move; if invalid, ask again until valid
        while not board.play(move, current_player.player_id):
            move = current_player.make_move(board)

        # Check game status
        winner = winning(board.get_board_state(), game_n)

        # Swap to the other player
        current_player_index = 1 - current_player_index

    # --- Post-game reporting ---
    print(board)

    if winner < 0:
        print('Game is a draw!')
    else:
        # Find the winner object to print a nice label
        winner_player = next(p for p in players if p.player_id == winner)
        # {winner_player} uses __str__ -> X/O; {winner_player!r} uses __repr__ -> MinMax/AlphaBeta
        print(f'Player {winner_player!r} won!')

    # Eval counts per player
    for p in players:
        # left: pretty label via __repr__, right: piece on board via __str__
        print(f'{p!r} (piece {p}) evaluated a boardstate {p.get_eval_count()} times!')

    # Timing summary per player
    for p in players:
        pid = p.player_id
        total = p_time[pid]
        moves = p_moves[pid]
        avg_ms = (total / moves) * 1000 if moves else 0.0
        print(f'{p!r} avg move time: {avg_ms:.2f} ms over {moves} moves (total {total:.3f}s)')

    return winner


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
    heuristic1: Heuristic = SimpleHeuristic(game_n)
    heuristic2: Heuristic = SimpleHeuristic(game_n)
    heuristic3: Heuristic = SimpleHeuristic(game_n)

    # if we want more players
    # heuristic4: Heuristic = SimpleHeuristic(game_n)


    h = SimpleHeuristic(game_n=4)
    alpha_beta = AlphaBetaPlayer(player_id=1, game_n=4, depth=5, heuristic=heuristic1)
    min_max: PlayerController = MinMaxPlayer(1, game_n, depth=3, heuristic=heuristic2)
    bot = Bot(player_id=2, game_n=game_n, heuristic=heuristic3)
    # TODO: Implement other PlayerControllers (MinMaxPlayer and AlphaBetaPlayer)

    players: List[PlayerController] = [alpha_beta, bot]

    assert players[0].player_id in {1, 2}, 'The player_id of the first player must be either 1 or 2'
    assert players[1].player_id in {1, 2}, 'The player_id of the second player must be either 1 or 2'
    assert players[0].player_id != players[1].player_id, 'The players must have an unique player_id'
    assert players[0].heuristic is not players[1].heuristic, 'The players must have an unique heuristic'
    assert len(players) == 2, 'Not the correct amount of players'

    return players


if __name__ == '__main__':
    game_n: int = 4 # n in a row required to win
    width: int = 7  # width of pthe board
    height: int = 6 # height of the board
    depth: int = 1 # depth for MinMax and AlphaBeta

    # Check whether the game_n is possible
    assert 1 < game_n <= min(width, height), 'game_n is not possible'

    board: Board = Board(width, height)
    start_game(game_n, board, get_players(game_n))