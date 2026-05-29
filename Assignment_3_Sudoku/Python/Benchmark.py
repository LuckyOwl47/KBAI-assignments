"""
Benchmarks the Sudoku solver in Game.py.

This file is purely for *measurement*. All solving logic — including the
queue heuristics — lives in Game.py. Here we only count work done, time
each run, and tabulate the results across every heuristic and puzzle.

CountingGame subclasses Game and changes nothing about *how* solving
works; it only wraps methods to tally how much work happens.
"""

import time
import os
from Sudoku import Sudoku
from Game import Game


class CountingGame(Game):
    """A Game that counts the work it does, without changing how it solves."""

    def __init__(self, sudoku, heuristic=None):
        super().__init__(sudoku, heuristic=heuristic)
        self.arc_revisions = 0
        self.domain_reductions = 0
        self.backtracks = 0

    def _revise(self, xi, xj) -> bool:
        self.arc_revisions += 1
        before = xi.get_domain_size()
        revised = super()._revise(xi, xj)
        self.domain_reductions += before - xi.get_domain_size()
        return revised

    def _backtrack(self) -> bool:
        self.backtracks += 1
        return super()._backtrack()


def run(sudoku_path, heuristic):
    g = CountingGame(Sudoku(sudoku_path), heuristic=heuristic)
    start = time.perf_counter()
    g.solve()
    elapsed = time.perf_counter() - start
    return {
        "solved":            g.valid_solution(),
        "time_ms":           elapsed * 1000,
        "arc_revisions":     g.arc_revisions,
        "domain_reductions": g.domain_reductions,
        "backtracks":        g.backtracks,
    }


# (label shown in the table, heuristic method name passed to Game)
STRATEGIES = [
    ("No heuristic",    None),
    ("MRV",             "_mrv"),
    ("Degree",          "_degree"),
    ("Finalized first", "_finalized"),
    ("LCV",             "_lcv"),
]


def benchmark():
    folder  = os.path.join(os.path.dirname(__file__), "Sudokus")
    puzzles = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])

    hdr = (f"  {'Strategy':<18} {'Solved':<8} {'Time (ms)':<12}"
           f" {'Arc Revisions':<16} {'Reductions':<14} {'Backtracks'}")

    for filename in puzzles:
        path = os.path.join(folder, filename)
        name = filename.replace(".txt", "")

        print(f"\n{'=' * 82}")
        print(f"  {name}")
        print(f"{'=' * 82}")
        print(hdr)
        print(f"  {'─'*16} {'─'*6} {'─'*10} {'─'*14} {'─'*12} {'─'*10}")

        for label, heuristic in STRATEGIES:
            s      = run(path, heuristic)
            solved = "✓" if s["solved"] else "✗"
            print(f"  {label:<18} {solved:<8} {s['time_ms']:<12.3f}"
                  f" {s['arc_revisions']:<16} {s['domain_reductions']:<14}"
                  f" {s['backtracks']}")

    print(f"\n{'=' * 82}")


if __name__ == "__main__":
    benchmark()