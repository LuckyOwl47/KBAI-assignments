# benchmarks.py
# Usage: python benchmarks.py

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from typing import Dict, List
import sys
import io
import warnings
warnings.filterwarnings('ignore')

from heuristics import SuperDuperHeuristic
from players import MinMaxPlayer, AlphaBetaPlayer
from board import Board
from app import winning


def run_game(game_n: int, width: int, height: int, depth: int,
             max_moves: int = 200) -> Dict:
    """
    Play a full game between MinMax (player 1) and AlphaBeta (player 2).
    Returns per-move average evaluations and runtime for both players.
    """
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
    time_accum = {1: 0.0, 2: 0.0} #
    total_moves = 0

    while winner == 0 and total_moves < max_moves:
        p = players[current_idx]
        pid = p.player_id

        old_stdout = sys.stdout
        sys.stdout = io.StringIO() # Remove print statements from playouts for cleaner output
        t0 = perf_counter()
        move = p.make_move(board)
        dt = perf_counter() - t0
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
    for d in depths:
        print(f"  depth={d} ...", end=" ", flush=True)
        res = run_game(game_n=5, width=5, height=5, depth=d)
        ratio = res['alphabeta_avg_evals'] / res['minmax_avg_evals'] if res['minmax_avg_evals'] > 0 else 0
        small_ratio.append(ratio)
        print(f"done  (AB/MM = {ratio:.4f}, moves={res['total_moves']})")

    # Big board
    print()
    print("=" * 50)
    print("Big board (10x10, game_n=10)")
    print("=" * 50)
    for d in depths:
        print(f"  depth={d} ...", end=" ", flush=True)
        res = run_game(game_n=10, width=10, height=10, depth=d)
        ratio = res['alphabeta_avg_evals'] / res['minmax_avg_evals'] if res['minmax_avg_evals'] > 0 else 0
        big_ratio.append(ratio)
        print(f"done  (AB/MM = {ratio:.4f}, moves={res['total_moves']})")

    print("\nPlotting...\n")

    # Plot small board
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, small_ratio, 'o-', color='tab:blue', linewidth=2, markersize=8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1 (no improvement)')
    ax.set_xlabel('Search Depth', fontsize=13)
    ax.set_ylabel('Evaluation Ratio (AlphaBeta / MinMax)', fontsize=13)
    ax.set_title('Small Board (5×5, game_n=5)', fontsize=14)
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
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1 (no improvement)')
    ax.set_xlabel('Search Depth', fontsize=13)
    ax.set_ylabel('Evaluation Ratio (AlphaBeta / MinMax)', fontsize=13)
    ax.set_title('Big Board (10×10, game_n=10)', fontsize=14)
    ax.set_xticks(depths)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_big_board.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Ratio plots saved.")

'''
def print_table():
    """
    Prints a table comparing MinMax and AlphaBeta on a 5x5 board
    at depth=3 for game_n = 3, 4, 5.
    Shows evaluations per move, runtime per move, and the eval ratio.
    """
    table_game_ns = [3, 4, 5]
    table_depth = 3
    table_rows = []

    print("=" * 90)
    print("TABLE: 5×5 board, depth=3, varying game_n")
    print("=" * 90)

    for gn in table_game_ns:
        print(f"  Running game_n={gn} ...", end=" ", flush=True)
        res = run_game(game_n=gn, width=5, height=5, depth=table_depth)
        table_rows.append(res)
        print("done")

    print()
    header = (
        f"{'game_n':>6} | "
        f"{'MM Evals/Move':>14} | "
        f"{'MM Time/Move (ms)':>18} | "
        f"{'AB Evals/Move':>14} | "
        f"{'AB Time/Move (ms)':>18} | "
    )
    print(header)
    print("-" * len(header))

    for gn, res in zip(table_game_ns, table_rows):
        mm_evals = res['minmax_avg_evals']
        mm_time = res['minmax_avg_time'] * 1000
        ab_evals = res['alphabeta_avg_evals']
        ab_time = res['alphabeta_avg_time'] * 1000

        print(
            f"{gn:>6} | "
            f"{mm_evals:>14.1f} | "
            f"{mm_time:>18.2f} | "
            f"{ab_evals:>14.1f} | "
            f"{ab_time:>18.2f} | "
        )

    print()
''' 

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
        f"{'Ratio (AB/MM)':>14}"
    )
    print(header)
    print("-" * len(header))

    for gn, res in zip(table_game_ns, table_rows):
        mm_evals = res['minmax_total_evals']
        mm_time = res['minmax_total_time']
        ab_evals = res['alphabeta_total_evals']
        ab_time = res['alphabeta_total_time']
        ratio = ab_evals / mm_evals if mm_evals > 0 else 0

        print(
            f"{gn:>6} | "
            f"{mm_evals:>15} | "
            f"{mm_time:>18.3f} | "
            f"{ab_evals:>15} | "
            f"{ab_time:>18.3f} | "
            f"{ratio:>14.4f}"
        )

    print()



if __name__ == '__main__':
    #plot_ratio_graphs()
    #print()
    print_table()
    print("Done.")