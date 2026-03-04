from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board
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
        # added this, love, Jacob
        np.random.seed(43) # for repeatable results 

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


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth
    
    def __repr__(self) -> str:
        return f"MinMax (d={self.depth}, h={self.heuristic})"


    def make_move(self, board: Board) -> int:
        """This finds the column that has the best move for minimax.

        Args:
            board (Board): the current board
        
        Returns:
            int: The column of the best move
        """
        # For the first move pick randomly for more random starting conditions (otherwise the game is always identical)
        if np.all(board.get_board_state() == 0):
            valid_cols = [col for col in range(board.width) if board.is_valid(col)]
            return np.random.choice(valid_cols)

        best_value = -np.inf # start with negative infinity so all values are larger
        best_move = 0 # column of best move
        opponent = 2 if self.player_id == 1 else 1 # You the opponent is 2 if your Id is 1, otherwise it's 1.

        for col in range(board.width):
            if not board.is_valid(col):
                continue # skips non valid colums
            new_board = board.get_new_board(col, self.player_id) # simulates the current board and what would happen if that move were chosen
            value = self.minimax(new_board, self.depth -1, maximizing=False, me=self.player_id, opponent=opponent) #Maximizing false because you're the root, and the next turn will start as the minimizer
            if value > best_value:
                best_value, best_move = value, col

        print(f"MiniMax returning col {best_move}, valid={board.is_valid(best_move)}") # added to see what column they pick

        return best_move
    

    def minimax(self, board: Board, depth: int, maximizing: bool, me: int, opponent: int) -> float:
        """This function implements the minimax algorithm through recursion. 
        
        Base-case: if depth is 0, or if the board is in a terminal case, 
                   return the board evaluation of the current board state

        Recursive case:
            1) Iterate through all valid moves
            2) Simulate the move to get a new board state for each move
            3) Recursively call the minimax function on the new board state with depth - 1 and switch player
            4) It will choose the max value of the children if it's maximizing, and the min value if it's minimizing
            5) Returns the best move with the highest/lowest value for each player

        Args:
            board (Board): the current board
            depth (int): remaining search depth
            maximizing (bool): True if the current player is maximizing, False if minimizing
            me (int): the maximizing player's id
            opponent (int): the minimizing player's id

        Returns 
            float: minimax's value for the board position
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
    
    """Class for the AlphaBeta player using the AlphaBeta algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth
    
    def __repr__(self) -> str:
        return f"AlphaBeta (d={self.depth}, h={self.heuristic})"


    def make_move(self, board: Board) -> int:
        """This finds the column that has the best move for AlphaBeta.

        Args:
            board (Board): the current board

        Returns:
            int: The column of the best move
        """

        # For the first move pick randomly for more random starting conditions (otherwise the game is always identical)
        if np.all(board.get_board_state() == 0):
            valid_cols = [col for col in range(board.width) if board.is_valid(col)]
            return np.random.choice(valid_cols)

        alpha = -np.inf
        beta = np.inf
        best_value = -np.inf # start with negative infinity so all values are larger
        best_move = next(col for col in range(board.width) if board.is_valid(col)) # Finds the first valid column. If I use 0 here like in minimax I get some weird error when n_game is 5 or larger, where it tries to use col 0 over and over even though it's invalid
        opponent = 2 if self.player_id == 1 else 1 # You the opponent is 2 if your Id is 1, otherwise it's 1.

        for col in range(board.width):
            if not board.is_valid(col):
                continue # skips non valid colums
            new_board = board.get_new_board(col, self.player_id) # simulates the current board and what would happen if that move were chosen
            value = self.alpha_beta(new_board, self.depth -1, maximizing=False, me=self.player_id, opponent=opponent, alpha=alpha, beta=beta) #Maximizing false because you're the root, and the next turn will start as the minimizer
            if value > best_value:
                best_value, best_move = value, col
            alpha = max(alpha, best_value)

        print(f"AlphaBeta returning col {best_move}, valid={board.is_valid(best_move)}") # added to see what column they pick
        
        return best_move
    

    def alpha_beta(self, board: Board, depth: int, maximizing: bool, me: int, opponent: int, alpha, beta) -> float:
        """
        This function implements the alpha-beta pruning algorithm through recursion.
        Base-case: if depth is 0, or if the board is in a terminal state,
                return the board evaluation of the current board state
        Recursive case:
            1) Iterate through all valid moves
            2) Simulate the move to get a new board state for each move
            3) Recursively call the minimax function on the new board state with depth - 1 and switch player
            4) It will choose the max value of the children if it's maximizing, and the min value if it's minimizing
            5) Returns the best move with the highest/lowest value for each player
            6) If alpha >= beta, prune the remaining branches, they won't affect the final decision
        Args:
            board (Board): the current board
            depth (int): remaining search depth
            maximizing (bool): True if the current player is maximizing, False if minimizing
            me (int): the maximizing player's id
            opponent (int): the minimizing player's id
            alpha: the best value the maximizer can guarantee so far
            beta: the best value the minimizer can guarantee so far

        Returns: 
            float: AlphaBeta's value for the board position
        """

        if depth == 0 or self._has_winner(board):
            self.evaluations += 1 # Counter for how many times it's evaluated the score
            return self.heuristic.evaluate_board(me, board)
        
        if maximizing:
            best = -np.inf # initialising value
            for col in range(board.width):
                if board.is_valid(col):
                    new_board = board.get_new_board(col, me)
                    best = max(best, self.alpha_beta(new_board, depth - 1, False, me, opponent, alpha, beta))
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        break
            return best
        else: # if player is minimizing:
            best = np.inf
            for col in range(board.width):
                if board.is_valid(col):
                    new_board = board.get_new_board(col, opponent)
                    best = min(best, self.alpha_beta(new_board, depth - 1, True, me, opponent, alpha, beta))
                    beta = min(beta, best)
                    if alpha >= beta:
                        break
            return best
        

    def _has_winner(self, board: Board) -> bool:
        from app import winning
        result = winning(board.get_board_state(), self.game_n)
        return result in (1,2) #true if someone won, false if they've lsot

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
        