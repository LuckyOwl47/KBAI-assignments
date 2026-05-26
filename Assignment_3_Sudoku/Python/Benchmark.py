import time
import os
from Sudoku import Sudoku
from Game import Game


class BenchmarkedGame(Game):
    """
    Extends Game with counters to measure algorithmic work.
    """

    def __init__(self, sudoku):
        super().__init__(sudoku)
        self.arc_revisions   = 0   # total arcs pulled from the queue
        self.domain_reductions = 0 # total values eliminated from domains
        self.backtracks      = 0   # total guesses made

    def _ac3(self) -> bool:
        board = self.sudoku.get_board()
        queue = []
        for i in range(9):
            for j in range(9):
                xi = board[i][j]
                for xj in xi.get_neighbours():
                    queue.append((xi, xj))

        while queue:
            xi, xj = queue.pop(0)
            self.arc_revisions += 1          # count every arc we look at
            if self._revise(xi, xj):
                if xi.get_domain_size() == 0 and not xi.is_finalized():
                    return False
                if xi.is_finalized():
                    for xk in xi.get_neighbours():
                        if xk.is_finalized() and xk.get_value() == xi.get_value():
                            return False
                for xk in xi.get_other_neighbours(xj):
                    queue.append((xk, xi))
        return True

    def _revise(self, xi, xj) -> bool:
        if xi.is_finalized():
            return False
        xj_values = [xj.get_value()] if xj.is_finalized() else xj.get_domain()
        revised = False
        for x in list(xi.get_domain()):
            if not any(y != x for y in xj_values):
                xi.remove_from_domain(x)
                self.domain_reductions += 1  # count every value eliminated
                revised = True
        return revised

    def _backtrack(self) -> bool:
        board = self.sudoku.get_board()

        best, best_size = None, 10
        for i in range(9):
            for j in range(9):
                cell = board[i][j]
                if not cell.is_finalized() and cell.get_domain_size() < best_size:
                    best, best_size = cell, cell.get_domain_size()

        if best is None:
            return True

        for value in list(best.get_domain()):
            self.backtracks += 1             # count every guess
            state = self._save_state()

            best.set_value(value)
            best.domain = []

            if self._ac3() and self._backtrack():
                return True

            self._restore_state(state)

        return False


def run_ac3_only(sudoku_path):
    """Run AC-3 without backtracking and return stats."""
    g = BenchmarkedGame(Sudoku(sudoku_path))
    start = time.perf_counter()
    g._ac3()
    elapsed = time.perf_counter() - start

    board = g.sudoku.get_board()
    unsolved = sum(1 for i in range(9) for j in range(9)
                   if not board[i][j].is_finalized())
    solved = unsolved == 0 and g.valid_solution()

    return {
        "solved":            solved,
        "unsolved_cells":    unsolved,
        "time_ms":           elapsed * 1000,
        "arc_revisions":     g.arc_revisions,
        "domain_reductions": g.domain_reductions,
        "backtracks":        0,
    }


def run_full(sudoku_path):
    """Run AC-3 + backtracking and return stats."""
    g = BenchmarkedGame(Sudoku(sudoku_path))
    start = time.perf_counter()
    g.solve()
    elapsed = time.perf_counter() - start
    solved = g.valid_solution()

    return {
        "solved":            solved,
        "unsolved_cells":    0 if solved else "?",
        "time_ms":           elapsed * 1000,
        "arc_revisions":     g.arc_revisions,
        "domain_reductions": g.domain_reductions,
        "backtracks":        g.backtracks,
    }


def benchmark():
    folder = os.path.join(os.path.dirname(__file__), "Sudokus")
    puzzles = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])

    print("=" * 72)
    print(f"{'BENCHMARK RESULTS':^72}")
    print("=" * 72)

    for mode, runner in [("AC-3 only", run_ac3_only), ("AC-3 + Backtracking", run_full)]:
        print(f"\n{'─' * 72}")
        print(f"  {mode}")
        print(f"{'─' * 72}")
        print(f"  {'Puzzle':<12} {'Solved':<8} {'Time (ms)':<12} "
              f"{'Arc Revisions':<16} {'Domain Reductions':<20} {'Backtracks'}")
        print(f"  {'─'*10} {'─'*6} {'─'*10} {'─'*14} {'─'*18} {'─'*10}")

        for filename in puzzles:
            path = os.path.join(folder, filename)
            stats = runner(path)
            name  = filename.replace(".txt", "")
            solved = "✓" if stats["solved"] else "✗"
            print(f"  {name:<12} {solved:<8} {stats['time_ms']:<12.3f} "
                  f"{stats['arc_revisions']:<16} {stats['domain_reductions']:<20} "
                  f"{stats['backtracks']}")

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    benchmark()
