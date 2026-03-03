import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from typing import Dict
import sys
import io
import warnings
warnings.filterwarnings('ignore')

from heuristics import SuperDuperHeuristic
from players import MinMaxPlayer, AlphaBetaPlayer
from board import Board
from app import winning

'''
This module contains benchmarks comparing MinMax and AlphaBeta on different board sizes,
'''
def run_game(game_n: int, width: int, height: int, depth: int) -> Dict:
    """
    INPUT: - game_n: the number of pieces in a row needed to win (e.g. 5 for Gomoku)
           - width, height: dimensions of the board N X N
           - depth: search depth for both MinMax and AlphaBeta
    OUTPUT: a dict containing stats about the game.
            NOTE: Not all stats are used in the current benchmarks, but they were used druing experimentation or for averaging results.
            
            Here are the stats we keep track of:
            - minmax_avg_evals: average evaluations per move for MinMax
            - minmax_total_evals: total evaluations for MinMax
            - minmax_avg_time: average time per move for MinMax
            - minmax_total_time: total time for MinMax
            - alphabeta_avg_evals: average evaluations per move for AlphaBeta
            - alphabeta_total_evals: total evaluations for AlphaBeta
            - alphabeta_avg_time: average time per move for AlphaBeta
            - alphabeta_total_time: total time for AlphaBeta
            - total_moves: total moves made in the game
            - winner: 0 for no winner, 1 for MinMax wins, 2

    This run_game function is identical to the one in app.py, only difference is that 
    it takes more args (the parameters we want to tweak for benchmarking) and outputs a 
    dict that is used to plot the graphs and tables in this module. By making a new function, 
    we can iterate over the different parameters more easily.
    """

    # ESTABLISHING THE PLAYERS
    h1 = SuperDuperHeuristic(game_n)
    h2 = SuperDuperHeuristic(game_n)

    mm = MinMaxPlayer(player_id=1, game_n=game_n, depth=depth, heuristic=h1)
    ab = AlphaBetaPlayer(player_id=2, game_n=game_n, depth=depth, heuristic=h2)

    players = [mm, ab]
    board = Board(width, height)

# Goods things to keep track of during the game:
    current_idx = 0 # index of the current player (0 for MinMax, 1 for AlphaBeta)
    winner = 0 # 0 means no winner yet, 1 means MinMax wins, 2 means AlphaBeta wins, -1 means draw
    move_count = {1: 0, 2: 0} # counts of moves made by each player (for calculating averages)
    time_accum = {1: 0.0, 2: 0.0} # Total time for each player (for calculating averages)
    total_moves = 0 # Total moves in the game (for reporting and sanity checks)

    while winner == 0:
        p = players[current_idx]
        pid = p.player_id

        old_stdout = sys.stdout
        sys.stdout = io.StringIO() # Remove print statements from playouts for cleaner output
        t0 = perf_counter()
        move = p.make_move(board)
        dt = (perf_counter() - t0) * 1000 # convert to milliseconds
        sys.stdout = old_stdout ######

        if not board.play(move, pid):
            continue

        move_count[pid] += 1
        time_accum[pid] += dt
        total_moves += 1
        winner = winning(board.get_board_state(), game_n)
        current_idx = 1 - current_idx

    mm_moves = max(move_count[1], 1)
    ab_moves = max(move_count[2], 1)

    return {
        'minmax_avg_evals': mm.get_eval_count() / mm_moves,
        'minmax_total_evals': mm.get_eval_count(),
        'minmax_avg_time': time_accum[1] / mm_moves,
        'minmax_total_time': time_accum[1],
        'alphabeta_avg_evals': ab.get_eval_count() / ab_moves,
        'alphabeta_total_evals': ab.get_eval_count(),
        'alphabeta_avg_time': time_accum[2] / ab_moves,
        'alphabeta_total_time': time_accum[2],
        'total_moves': total_moves,
        'winner': winner
    }


def plot_ratio_graphs():
    """
    Plots evaluation ratio (AlphaBeta / MinMax) vs depth (1-7)
    for a small board (5x5, game_n=5) and a big board (10x10, game_n=10).
    """
    depths = list(range(1, 8))
    small_ratio = []
    big_ratio = []

    # Small board
    print("=" * 50)
    print("Small board (5x5, game_n=5)")
    print("=" * 50)

    ### LOOP TO PLOT RATIO GRAPHS OVER ALL DEPTHS FOR SMALL BOARD
    for d in depths:
        print(f"  depth={d} ...", end=" ", flush=True)
        res = run_game(game_n=5, width=5, height=5, depth=d)
        ratio = res['minmax_avg_evals'] / res['alphabeta_avg_evals'] 
        small_ratio.append(ratio)
        print(f"done  (AB/MM = {ratio:.3f}, moves={res['total_moves']})")

    print()
    print("=" * 50)
    print("Big board (10x10, game_n=10)")
    print("=" * 50)
    
    ### LOOP TO PLOT RATIO GRAPHS OVER ALL DEPTHS FOR BIG BOARD
    ### FOR SANITY CHECK (takes long to run)
    for d in depths:
        print(f"  depth={d} ...", end=" ", flush=True)
        res = run_game(game_n=10, width=10, height=10, depth=d)
        ratio = res['minmax_avg_evals'] / res['alphabeta_avg_evals'] 
        big_ratio.append(ratio)
        print(f"done  (AB/MM = {ratio:.3f}, moves={res['total_moves']})")

    print("\nPlotting...\n")

    # Plot small board
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, small_ratio, 'o-', color='tab:blue', linewidth=2, markersize=8)
    ax.set_xlabel('Search depth', fontsize=13)
    ax.set_ylabel('Evaluation ratio (MinMax / AlphaBeta)', fontsize=13)
    ax.set_title('Small board (5×5, game_n=5)', fontsize=14)
    ax.set_xticks(depths)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_small_board.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plot big board
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, big_ratio, 's-', color='tab:orange', linewidth=2, markersize=8)
    ax.set_xlabel('Search depth', fontsize=13)
    ax.set_ylabel('Evaluation ratio (MinMax / AlphaBeta)', fontsize=13)
    ax.set_title('Big Board (10×10, game_n=10)', fontsize=14)
    ax.set_xticks(depths)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_big_board.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Ratio plots saved.")



def print_table():
    """
    Prints a table comparing MinMax and AlphaBeta on a 5x5 board
    at depth=3 for game_n = 3, 4, 5.
    Shows total evaluations per game, total runtime per game, and the eval ratio.
    """
    table_game_ns = [3, 4, 5]
    table_depth = 3
    table_rows = []

    print("=" * 100)
    print("TABLE: 5×5 board, depth=3, varying game_n (totals per game)")
    print("=" * 100)

    ### LOOP TO RUN GAMES FOR DIFFERENT game_n ON 5X5 BOARD AND COLLECT STATS FOR THE TABLE
    for gn in table_game_ns:
        print(f"  Running game_n={gn} ...", end=" ", flush=True)
        res = run_game(game_n=gn, width=5, height=5, depth=table_depth)
        table_rows.append(res)
        print("done")

    print()

    # good presentation of the stats in a table format
    header = (
        f"{'Win condition':>6} | "
        f"{'MM Evals':>15} | "
        f"{'MM Time (ms)':>18} | "
        f"{'AB Total Evals':>15} | "
        f"{'AB Total Time (s)':>18} | "
        f"{'Ratio (MM/AB)':>14}"
    )
    print(header)
    print("-" * len(header))

    for gn, res in zip(table_game_ns, table_rows):
        mm_evals = res['minmax_total_evals']
        mm_time = res['minmax_total_time']
        ab_evals = res['alphabeta_total_evals']
        ab_time = res['alphabeta_total_time']
        ratio = mm_evals / ab_evals

        print(
            f"{gn:>6} | "
            f"{mm_evals:>15} | "
            f"{mm_time:>18.3f} | "
            f"{ab_evals:>15} | "
            f"{ab_time:>18.3f} | "
            f"{ratio:>14.3f}"
        )

    print()



if __name__ == '__main__':
    plot_ratio_graphs()
    print()
    print_table()
    print("Done.")