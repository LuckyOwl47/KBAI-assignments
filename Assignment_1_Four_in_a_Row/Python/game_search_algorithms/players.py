from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING

from numpy.random import random
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board
import copy
import time



class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic
        # added this, love, ingvar
        self.evaluations: int = 0


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass
class Bot(PlayerController):
    """Class for a bot player using a heuristic
    Inherits from Playercontroller
    basically this player just makes random moves
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        super().__init__(player_id, game_n, heuristic)
    
    def __repr__(self) -> str:
        return f"Bot (h={self.heuristic})"


    def make_move(self, board: Board) -> int:
        """
        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        valid_cols = []
        for col in range(board.width):
            if board.is_valid(col):
                valid_cols.append(col)

        if not valid_cols:
            # Handle the case where there are no valid moves.
            # This might mean the game is over or the board is full.
            # You might want to return None, raise an error, or handle it differently.
            return None

        return random.choice(valid_cols)


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth
    
    def __repr__(self) -> str:
        return f"MinMax (d={self.depth}, h={self.heuristic})"


    def make_move(self, board: Board) -> int:
        """
        ### DOCSTRING FOR THE MINIMAX ALGORITHM ###

        This function implements the minimax algorithm through recursion. 

        Args:
            board (Board): the current board
        """

        # TODO: implement minmax algortihm!
        # HINT: use the functions on the 'board' object to produce a new board given a specific move
        # HINT: use the functions on the 'heuristic' object to produce evaluations for the different board states!
        
        # Example:
        # max_value: float = -np.inf # negative infinity
        # max_move: int = 0
        # for col in range(board.width):
        #     if board.is_valid(col):
        #         new_board: Board = board.get_new_board(col, self.player_id)
        #         value: int = self.heuristic.evaluate_board(self.player_id, new_board)
        #         if value > max_value:
        #             max_move = col

        # This returns the same as
        # self.heuristic.get_best_action(self.player_id, board) # Very useful helper function!

        # This is obviously not enough (this is depth 1)
        # Your assignment is to create a data structure (tree) to store the gameboards such that you can evaluate a higher depths.
        # Then, use the minmax algorithm to search through this tree to find the best move/action to take!

        # return max_move

        best_value = -np.inf # start with negative infinity so all values are larger
        best_move = 0
        opponent = 2 if self.player_id == 1 else 1

        for col in range(board.width):
            if not board.is_valid(col):
                continue # skips non valid colums
            new_board = board.get_new_board(col, self.player_id) # simulates the current board and what would happen if that move were chosen
            value = self.minimax(new_board, self.depth -1, maximizing=False, me=self.player_id, opponent=opponent) #Maximizing false because you're the root, and the next turn will start as the minimizer
            if value > best_value:
                best_value, best_move = value, col

        return best_move
    

    def minimax(self, board: Board, depth: int, maximizing: bool, me: int, opponent: int) -> float:
        """
        ### DOCSTRING FOR THE MINIMAX ALGORITHM ###

        This function implements the minimax algorithm through recursion. 
        
        Base-case: if depth is 0, or if the board is in a terminal case, 
                   return the board evaluation of the current board state

        Recursive case:
                     1) Iterate through all valid moves
                     2) Simulate the move to get a new board state for each move
                     3) Recursively call the minimax function on the new board state with depth - 1
                        and switch player
                     4) It will choose the max value of the children if it's maximizing, and the min value if it's minimizing
                     5) Returns the best move with the highest/lowest value for each player

        
        Args:
            board (Board): the current board
        """

        if depth == 0 or self._has_winner(board):
            self.evaluations += 1 # Counter for how many times it's evaluated the score
            return self.heuristic.evaluate_board(me, board)
        
        if maximizing:
            best = -np.inf # initialising value
            for col in range(board.width):
                if board.is_valid(col):
                    new_board = board.get_new_board(col, me)
                    best = max(best, self.minimax(new_board, depth - 1, False, me, opponent))
            return best
        else: # if player is minimizing:
            best = np.inf
            for col in range(board.width):
                if board.is_valid(col):
                    new_board = board.get_new_board(col, opponent)
                    best = min(best, self.minimax(new_board, depth - 1, True, me, opponent))
            return best
        

    def _has_winner(self, board: Board) -> bool:
        from app import winning
        result = winning(board.get_board_state(), self.game_n)
        return result in (1,2) #true if someone won, false if they've lsot



class AlphaBetaPlayer(PlayerController):
    """Minimax player with alpha–beta pruning."""
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth
        self.evaluations: int = 0  # how many leaf evaluations we did
        self.prunes: int = 0       # how many times we pruned

    def __str__(self) -> str:
        return f'AlphaBeta (d={self.depth}, h={self.heuristic})'
    def __repr__(self) -> str:
        return f"AlphaBeta (d={self.depth}, h={self.heuristic})"

    def make_move(self, board: Board) -> int:
        """Choose a column using minimax with alpha–beta pruning."""
        best_value = -np.inf
        best_move = 0
        opponent = 2 if self.player_id == 1 else 1
        alpha = -np.inf
        beta = np.inf

        # Iterate legal moves; (optional) you could do center-first ordering for speed.
        for col in range(board.width):
            if not board.is_valid(col):
                continue
            child = board.get_new_board(col, self.player_id)
            value = self._alphabeta(
                child,
                self.depth - 1,
                alpha,
                beta,
                maximizing=False,
                me=self.player_id,
                opponent=opponent
            )
            if value > best_value:
                best_value, best_move = value, col
            alpha = max(alpha, best_value)
        print(f"[AlphaBetaPlayer] Move chosen: {best_move}, value={best_value:.2f}, "
      f"evaluations={self.evaluations}, prunes={self.prunes}")

        # after you decide best_col0
        return best_move

    def _alphabeta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        me: int,
        opponent: int
    ) -> float:
        # Terminal node or depth limit: evaluate
        if depth == 0 or self._has_winner(board):
            self.evaluations += 1
            return float(self.heuristic.evaluate_board(me, board))

        if maximizing:
            value = -np.inf
            for col in range(board.width):
                if not board.is_valid(col):
                    continue
                child = board.get_new_board(col, me)
                value = max(value, self._alphabeta(child, depth - 1, alpha, beta, False, me, opponent))
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.prunes += 1
                    break
            return value
        else:
            value = np.inf
            for col in range(board.width):
                if not board.is_valid(col):
                    continue
                child = board.get_new_board(col, opponent)
                value = min(value, self._alphabeta(child, depth - 1, alpha, beta, True, me, opponent))
                beta = min(beta, value)
                if alpha >= beta:
                    self.prunes += 1
                    break
            return value

    def _has_winner(self, board: Board) -> bool:
        # Reuse the same helper used by MinMaxPlayer
        from app import winning
        result = winning(board.get_board_state(), self.game_n)
        # Treat win/loss and draw (-1) as terminal; change to (1,2) only if you want to keep exploring draws.
        return result in (1, 2, -1)


class MonteCarloPlayer(PlayerController):
    """
    Monte Carlo Tree Search player for n-in-a-row.

    Parameters
    ----------
    player_id : int
        1 or 2
    game_n : int
        The 'n' you need in a row to win
    rollouts : int
        Number of simulations (budget). Ignored if time_limit_ms is set.
    heuristic : Heuristic
        Not used in basic random-rollout MCTS, but kept for interface parity and
        easy future upgrades (e.g., heuristic-guided rollouts).
    exploration : float
        UCT exploration constant (default: sqrt(2) ~ 1.4142)
    time_limit_ms : int | None
        Optional wall-clock budget in milliseconds. If provided, we stop when time is up.
    max_playout_depth : int
        Safety cap for rollout length to avoid very long games during simulation.
    """
    #def __repr__(self) -> str:
    #    return f"AlphaBeta (d={self.depth}, h={self.heuristic})"
    def __repr__(self) -> str:
    #Debug-friendly label (for logs, game summaries, etc.).
        parts = [f"r={self.rollouts}"]
        if self.time_limit_ms:
            parts.append(f"t={self.time_limit_ms}ms")
        parts.append(f"h={self.heuristic}")
        return f"MonteCarlo ({', '.join(parts)})"

    def __init__(self,
                 player_id: int,
                 game_n: int,
                 rollouts: int,
                 heuristic: 'Heuristic',
                 exploration: float = 1.41421356237,
                 time_limit_ms: int | None = None,
                 max_playout_depth: int = 256) -> None:
        super().__init__(player_id, game_n, heuristic)
        self.rollouts = max(1, int(rollouts))
        self.c = float(exploration)
        self.time_limit_ms = time_limit_ms
        self.max_playout_depth = max_playout_depth

    # --------- Public API ---------
    def make_move(self, board: 'Board') -> int:
        """
        Return a column to play (0-based).
        We pick the root child with the highest visit count (robust child).
        """
        if self._is_terminal(board)[0] != 0:
            # Game is already terminal; just pick the first legal move to be safe.
            for col0 in range(board.width):
                if board.is_valid(col0):
                    return col0 
            return 0  # fallback

        root = _MCTSNode(board=board, to_move=self.player_id)
        self._search(root)

        # Choose the most visited child
        if not root.children:
            # No expansions? pick first legal move
            for col0 in range(board.width):
                if board.is_valid(col0):
                    return col0 
            return 0
        best_child = max(root.children, key=lambda ch: ch.N)
        # --- compute values for printout ---
        total_rollouts = getattr(self, "rollout_count", 0)
        total_nodes = sum(ch.N for ch in root.children)
        avg_value = (best_child.W / best_child.N) if best_child.N > 0 else 0.0  # <--- this line defines avg_value
        print(
            f"[MonteCarloPlayer] Move chosen: {best_child.move_col0}, "
            f"value={avg_value:.2f}, "
            f"rollouts={total_rollouts}, "
            f"child_visits={total_nodes}"
)
        return best_child.move_col0 # convert back to 1-based

    # --------- Core MCTS loop ---------
    def _search(self, root: '_MCTSNode') -> None:
        start = time.perf_counter()
        iterations = 0

        def under_budget() -> bool:
            if self.time_limit_ms is None:
                return iterations < self.rollouts
            return (time.perf_counter() - start) * 1000.0 < self.time_limit_ms

        while under_budget():
            node = root

            # SELECTION: descend by UCT while node is fully expanded and non-terminal
            while not node.is_terminal and node.is_fully_expanded():
                node = self._uct_select(node)

            # EXPANSION: if non-terminal and not fully expanded, expand one child
            if not node.is_terminal and not node.is_fully_expanded():
                node = self._expand(node)
            if node.is_fully_expanded() and not node.children:
                # defensive assertion to detect mismatch
                any_legal = any(node.board.is_valid(c) for c in range(node.board.width))
                if any_legal:
                    print("[MCTS] Mismatch: leaf has no children but board still has legal moves.")
                else:
                    print("[MCTS] Leaf is truly full — marking terminal as draw.")
                node.is_terminal = True
                break  # short-circuit selection for this iteration


            # SIMULATION
            result = self._simulate(node)

            # BACKPROP
            self._backpropagate(node, result)

            iterations += 1
            if hasattr(self, "rollout_count"):
                self.rollout_count += 1


    def _uct_select(self, node: '_MCTSNode') -> '_MCTSNode':
        import math
        # Parent visit count for UCT
        log_parent_N = math.log(max(1, node.N))
        def uct_value(child: '_MCTSNode') -> float:
            if child.N == 0:
                return float('inf')
            exploit = child.W / child.N  # mean value from root perspective
            explore = self.c * ((log_parent_N / child.N) ** 0.5)
            return exploit + explore

        return max(node.children, key=uct_value)

    def _expand(self, node: '_MCTSNode') -> '_MCTSNode':
        # Expand one of the untried legal moves
        untried = node.untried_moves()
        if not untried:
            node.is_terminal = True
            return node

        col0 = untried.pop()
        child_board = self._apply_move(node.board, col0, node.to_move)
        next_to_move = 1 if node.to_move == 2 else 2
        child = _MCTSNode(board=child_board, to_move=next_to_move,
                          parent=node, move_col0=col0)
        node.children.append(child)
        return child

    def _simulate(self, node: '_MCTSNode') -> float:
        """
        Random playout until terminal or max depth.
        Return +1 if ROOT player eventually wins, -1 if ROOT player loses, 0 for draw.
        """
        import random
        board = node.board
        to_move = node.to_move
        root_player = self._root_player(node)

        winner, terminal = self._is_terminal(board)
        if terminal:
            return self._terminal_value(winner, root_player)

        depth = 0
        while depth < self.max_playout_depth:
            # collect legal moves
            legal = [c for c in range(board.width) if board.is_valid(c)]
            if not legal:
                break
            # 1) try winning move
            for c in legal:
                if self._is_terminal(self._apply_move(board, c, to_move))[1]:
                    # Only true if that move finishes the game with to_move winning
                    w, term = self._is_terminal(self._apply_move(board, c, to_move))
                    if term and w == to_move:
                        col0 = c
                        break
            else:
                # 2) try block opponent's immediate win
                opp = 1 if to_move == 2 else 2
                block = None
                for c in legal:
                    nb = self._apply_move(board, c, to_move)
                    # see if opponent can win right after
                    opp_legal = [cc for cc in range(nb.width) if nb.is_valid(cc)]
                    for cc in opp_legal:
                        w2, term2 = self._is_terminal(self._apply_move(nb, cc, opp))
                        if term2 and w2 == opp:
                            block = c
                            break
                    if block is not None:
                        break
                if block is not None:
                    col0 = block
                else:
                    # 3) random fallback
                    col0 = random.choice(legal)

            board = self._apply_move(board, col0, to_move)
            winner, terminal = self._is_terminal(board)
            if terminal:
                return self._terminal_value(winner, root_player)
            to_move = 1 if to_move == 2 else 2
            depth += 1

        # If we hit depth cap or no moves: treat as draw
        return 0.0

    def _backpropagate(self, node: '_MCTSNode', result_for_root: float) -> None:
        # Walk back up to root; each node stores value from ROOT player's perspective
        while node is not None:
            node.N += 1
            node.W += result_for_root
            node = node.parent

    # --------- Helpers / Game hooks ---------
    def _apply_move(self, board: 'Board', col0: int, player_id: int) -> 'Board':
        """
        Clone the board, then apply the move using your Board API:
        """
        # Prefer a real copy/clone if your Board supports it
        if hasattr(board, "copy") and callable(board.copy):
            new_board = board.copy()
        elif hasattr(board, "clone") and callable(board.clone):
            new_board = board.clone()
        else:
            # Safe fallback: deep-copy the Python object
            new_board = copy.deepcopy(board)

        # Apply move on the cloned board (convert 0-based -> 1-based)
        ok = new_board.play(col0, player_id)
        # In MCTS we only expand legal moves, so this should always succeed
        # but keep it robust in case Board validity rules change:
        if not ok:
            # If this ever triggers, we likely mismatched indices.
            raise ValueError(f"MCTS tried illegal move col0={col0} (-> {col0+1}) for player {player_id}")
        return new_board


    def _root_player(self, node: '_MCTSNode') -> int:
        # Climb to root to know who we’re evaluating from
        cur = node
        while cur.parent is not None:
            cur = cur.parent
        return cur.to_move  # the first mover at root

    def _terminal_value(self, winner: int, root_player: int) -> float:
        if winner == 0:   # non-terminal sentinel (shouldn’t happen here)
            return 0.0
        if winner == -1:  # draw
            return 0.0
        return 1.0 if winner == root_player else -1.0

    def _is_terminal(self, board: 'Board') -> tuple[int, bool]:
        """
        Returns (winner, is_terminal)
        winner: 1 or 2 if that player has n-in-a-row,
                -1 if draw, 0 otherwise
        We read the numpy state with shape (width, height).
        """
        state = board.get_board_state()
        n = self.game_n
        w, h = state.shape[0], state.shape[1]

        # vertical
        for x in range(w):
            count, player = 0, 0
            for y in range(h):
                cell = state[x, y]
                if cell != 0 and cell == player:
                    count += 1
                else:
                    player = cell
                    count = 1 if cell != 0 else 0
                if player and count >= n:
                    return player, True

        # horizontal
        for y in range(h):
            count, player = 0, 0
            for x in range(w):
                cell = state[x, y]
                if cell != 0 and cell == player:
                    count += 1
                else:
                    player = cell
                    count = 1 if cell != 0 else 0
                if player and count >= n:
                    return player, True

        # diag down-right
        for x0 in range(w - n + 1):
            for y0 in range(h - n + 1):
                player = state[x0, y0]
                if player == 0:
                    continue
                ok = True
                for k in range(1, n):
                    if state[x0 + k, y0 + k] != player:
                        ok = False
                        break
                if ok:
                    return player, True

        # diag up-right
        for x0 in range(w - n + 1):
            for y0 in range(n - 1, h):
                player = state[x0, y0]
                if player == 0:
                    continue
                ok = True
                for k in range(1, n):
                    if state[x0 + k, y0 - k] != player:
                        ok = False
                        break
                if ok:
                    return player, True

        # any legal moves left?
        any_legal = any(board.is_valid(c) for c in range(board.width))
        if not any_legal:
            return -1, True  # draw
        return 0, False


class _MCTSNode:
    """
    A simple tree node for MCTS.
    Stores statistics from the ROOT player's perspective:
    - N: visit count
    - W: total value (sum of rollout returns); mean = W/N
    """
    __slots__ = ('board', 'to_move', 'parent', 'children', 'move_col0', 'N', 'W', '_untried', 'is_terminal')

    def __init__(self,
                 board: 'Board',
                 to_move: int,
                 parent: '_MCTSNode' | None = None,
                 move_col0: int | None = None) -> None:
        self.board = board
        self.to_move = to_move
        self.parent = parent
        self.children: list[_MCTSNode] = []
        self.move_col0 = move_col0  # column (0-based) used to reach this node
        self.N = 0
        self.W = 0.0
        self._untried = None  # lazily computed set of 0-based legal columns
        # Pre-compute terminal?
        self.is_terminal = False

    def untried_moves(self) -> set[int]:
        if self._untried is None:
            self._untried = {c for c in range(self.board.width) if self.board.is_valid(c)}
        return self._untried

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves()) == 0



class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play (0 based)in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
        