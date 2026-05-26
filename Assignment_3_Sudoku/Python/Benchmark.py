import heapq
import time
import os
from Sudoku import Sudoku
from Game import Game


class BenchmarkedGame(Game):
    """
    Extends Game with counters and a pluggable queue heuristic for AC-3.

    queue_heuristic:
        "none"       — standard FIFO queue, no ordering
        "mrv"        — process arcs where xi has the fewest remaining
                       candidates first (Minimum Remaining Values)
        "degree"     — process arcs where xi has the most unsolved
                       neighbours first
        "finalized"  — prioritise arcs where xj is already finalized;
                       these are guaranteed to cause a reduction if xi
                       still contains xj's value
        "lcv"        — process arcs where xi's domain overlaps most with
                       xj's values first (most likely to cause a reduction)
    """

    def __init__(self, sudoku, queue_heuristic="none"):
        super().__init__(sudoku)
        self.queue_heuristic   = queue_heuristic
        self.arc_revisions     = 0
        self.domain_reductions = 0
        self.backtracks        = 0
        self._counter          = 0   # unique tiebreaker so Field objects are never compared

    # -------------------------------------------------- queue helpers

    def _priority(self, xi, xj):
        """Compute the priority for arc (xi, xj). Lower = processed first."""
        h = self.queue_heuristic

        if h == "mrv":
            return xi.get_domain_size()

        if h == "degree":
            deg = sum(1 for n in xi.get_neighbours() if not n.is_finalized())
            return -deg   # negate so higher degree = lower number = first

        if h == "finalized":
            return 0 if xj.is_finalized() else 1

        if h == "lcv":
            xj_vals = ({xj.get_value()} if xj.is_finalized()
                       else set(xj.get_domain()))
            overlap = len(set(xi.get_domain()) & xj_vals)
            return -overlap  # more overlap = more likely to reduce = first

    def _push(self, queue, xi, xj):
        if self.queue_heuristic == "none":
            queue.append((xi, xj))
        else:
            heapq.heappush(queue, (self._priority(xi, xj), self._counter, xi, xj))
            self._counter += 1

    def _pop(self, queue):
        if self.queue_heuristic == "none":
            return queue.pop(0)
        _, _, xi, xj = heapq.heappop(queue)
        return xi, xj

    # ---------------------------------------------------------------- AC-3

    def _ac3(self) -> bool:
        board = self.sudoku.get_board()
        queue = []
        for i in range(9):
            for j in range(9):
                for xj in board[i][j].get_neighbours():
                    self._push(queue, board[i][j], xj)

        while queue:
            xi, xj = self._pop(queue)
            self.arc_revisions += 1
            if self._revise(xi, xj):
                if xi.get_domain_size() == 0 and not xi.is_finalized():
                    return False
                if xi.is_finalized():
                    for xk in xi.get_neighbours():
                        if xk.is_finalized() and xk.get_value() == xi.get_value():
                            return False
                for xk in xi.get_other_neighbours(xj):
                    self._push(queue, xk, xi)
        return True

    def _revise(self, xi, xj) -> bool:
        if xi.is_finalized():
            return False
        xj_values = [xj.get_value()] if xj.is_finalized() else xj.get_domain()
        revised = False
        for x in list(xi.get_domain()):
            if not any(y != x for y in xj_values):
                xi.remove_from_domain(x)
                self.domain_reductions += 1
                revised = True
        return revised

    # -------------------------------------------------------- backtracking
    # (bonus — uses MRV cell selection, no value ordering)

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

        for value in sorted(best.get_domain()):
            self.backtracks += 1
            state = self._save_state()

            best.set_value(value)
            best.domain = []

            if self._ac3() and self._backtrack():
                return True

            self._restore_state(state)

        return False


# --------------------------------------------------------------- runner

def run(sudoku_path, queue_h):
    g     = BenchmarkedGame(Sudoku(sudoku_path), queue_heuristic=queue_h)
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


# ------------------------------------------------------------ benchmark

STRATEGIES = [
    ("No heuristic",    "none"),
    ("MRV",             "mrv"),
    ("Degree",          "degree"),
    ("Finalized first", "finalized"),
    ("LCV",             "lcv"),
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

        for label, queue_h in STRATEGIES:
            s      = run(path, queue_h)
            solved = "✓" if s["solved"] else "✗"
            print(f"  {label:<18} {solved:<8} {s['time_ms']:<12.3f}"
                  f" {s['arc_revisions']:<16} {s['domain_reductions']:<14}"
                  f" {s['backtracks']}")

    print(f"\n{'=' * 82}")


if __name__ == "__main__":
    benchmark()
