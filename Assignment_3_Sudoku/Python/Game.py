class Game:

    def __init__(self, sudoku, heuristic = None):
        self.sudoku = sudoku
        self.heuristic = heuristic

    def show_sudoku(self):
        print(self.sudoku)

    def solve(self) -> bool:
        if not self._ac3():
            return False
        return self._backtrack()

    ###             HEURISTICS           ###
    '''Called using args into the class'''

    def _mrv(self, xi, xj):
        return xi.get_domain_size()
 
    def _degree(self, xi, xj):
        return -sum(1 for n in xi.get_neighbours() if not n.is_finalized())
 
    def _finalized(self, xi, xj):
        return 0 if xj.is_finalized() else 1
 
    def _lcv(self, xi, xj):
        xj_vals = ({xj.get_value()} if xj.is_finalized()
                   else set(xj.get_domain()))
        return -len(set(xi.get_domain()) & xj_vals)
    


    def _ac3(self) -> bool:
        board = self.sudoku.get_board()
        queue = []
        for i in range(9):
            for j in range(9):
                xi = board[i][j]
                for xj in xi.get_neighbours():
                    queue.append((xi, xj))

        while queue:
            if self.heuristic is None:
                xi, xj = queue.pop(0)
            else:
                fn = getattr(self, self.heuristic)
                best = min(queue, key=lambda arc: fn(*arc))
                queue.remove(best)
                xi, xj = best

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
                revised = True
        return revised

    def _save_state(self):
        board = self.sudoku.get_board()
        return [
            (board[i][j].get_value(), list(board[i][j].get_domain()))
            for i in range(9) for j in range(9)
        ]

    def _restore_state(self, state):
        board = self.sudoku.get_board()
        idx = 0
        for i in range(9):
            for j in range(9):
                board[i][j].set_value(state[idx][0])
                board[i][j].domain = list(state[idx][1])
                idx += 1

    def _backtrack(self) -> bool:
        board = self.sudoku.get_board()

        # Find the unsolved cell with the fewest candidates (MRV)
        best, best_size = None, 10
        for i in range(9):
            for j in range(9):
                cell = board[i][j]
                if not cell.is_finalized() and cell.get_domain_size() < best_size:
                    best, best_size = cell, cell.get_domain_size()

        if best is None:
            return True  # every cell is solved

        for value in list(best.get_domain()):
            state = self._save_state()      # snapshot before the guess

            best.set_value(value)           # commit the guess
            best.domain = []

            if self._ac3() and self._backtrack():   # propagate + recurse
                return True

            self._restore_state(state)      # guess was wrong, undo everything

        return False  # no value worked → contradiction, tell caller to backtrack

    def valid_solution(self) -> bool:
        board = self.sudoku.get_board()
        expected = set(range(1, 10))
        for i in range(9):
            if {board[i][j].get_value() for j in range(9)} != expected:
                return False
        for j in range(9):
            if {board[i][j].get_value() for i in range(9)} != expected:
                return False
        for box_r in range(3):
            for box_c in range(3):
                values = {board[box_r * 3 + r][box_c * 3 + c].get_value()
                          for r in range(3) for c in range(3)}
                if values != expected:
                    return False
        return True
