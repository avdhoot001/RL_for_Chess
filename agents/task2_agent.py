





# from minichess.chess.fastchess import Chess
# from .base_agent import BaseAgent
# import random
# from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves, inv_color

# class Task2Agent(BaseAgent):
#     def __init__(self, name="OptimizedMiniMaxAgent"):
#         super().__init__(name)
        
#         # Piece values with refined weights
#         self.piece_values = {
#             0: 100,   # Pawn
#             1: 320,   # Knight
#             2: 330,   # Bishop
#             3: 500,   # Rook
#             4: 900,   # Queen
#             5: 20000  # King
#         }
        
#         # Position bonus tables for 5x4 board (encourage center control)
#         self.pawn_table = [
#             [0,  0,  0,  0],
#             [50, 50, 50, 50],
#             [10, 20, 20, 10],
#             [5,  10, 10,  5],
#             [0,  0,  0,  0]
#         ]
        
#         self.piece_table = [
#             [0,  5,  5,  0],
#             [5, 10, 10,  5],
#             [5, 15, 15,  5],
#             [5, 10, 10,  5],
#             [0,  5,  5,  0]
#         ]
        
#         self.max_depth = 4
#         self.nodes_searched = 0
    
#     # ---------------------------------------------------------------------
#     # Evaluation Function (fixed to be from the agent’s perspective)
#     # ---------------------------------------------------------------------
#     def evaluate_board(self, chess_obj: Chess, root_color: int):
#         """Enhanced board evaluation with consistent agent perspective."""
        
#         result = chess_obj.game_result()
#         if result is not None:
#             if result == 0:
#                 return 0
#             if result == 1:  # White wins
#                 return 100000 if root_color == 1 else -100000
#             if result == -1:  # Black wins
#                 return -100000 if root_color == 1 else 100000
        
#         score = 0
#         my_turn = chess_obj.turn
#         opp_turn = inv_color(my_turn)
        
#         # 1. Material + position bonuses
#         for i in range(chess_obj.dims[0]):
#             for j in range(chess_obj.dims[1]):
#                 # My pieces
#                 my_piece = chess_obj.piece_at(i, j, my_turn)
#                 if my_piece != -1:
#                     score += self.piece_values[my_piece]
#                     if my_piece == 0:  # Pawn
#                         row = i if my_turn == 1 else (4 - i)
#                         score += self.pawn_table[row][j]
#                     elif my_piece != 5:
#                         score += self.piece_table[i][j]
                
#                 # Opponent pieces
#                 opp_piece = chess_obj.piece_at(i, j, opp_turn)
#                 if opp_piece != -1:
#                     score -= self.piece_values[opp_piece]
#                     if opp_piece == 0:
#                         row = i if opp_turn == 1 else (4 - i)
#                         score -= self.pawn_table[row][j]
#                     elif opp_piece != 5:
#                         score -= self.piece_table[i][j]
        
#         # 2. Mobility
#         moves, proms = chess_obj.legal_moves()
#         mobility = sum(1 for _ in piece_matrix_to_legal_moves(moves, proms))
#         score += mobility * 10
        
#         # 3. King safety
#         if chess_obj.any_checkers:
#             score -= 50
        
#         # 4. Pawn advancement
#         for i in range(chess_obj.dims[0]):
#             for j in range(chess_obj.dims[1]):
#                 pawn = chess_obj.piece_at(i, j, my_turn)
#                 if pawn == 0:
#                     advancement = (4 - i) if my_turn == 1 else i
#                     score += advancement * 15
        
#         # Flip perspective so that positive = good for root_color
#         if chess_obj.turn != root_color:
#             score = -score
        
#         return score

#     # ---------------------------------------------------------------------
#     # Move ordering for better alpha-beta pruning
#     # ---------------------------------------------------------------------
#     def order_moves(self, chess_obj: Chess, legal_moves):
#         scored_moves = []
#         for move in legal_moves:
#             (i, j), (dx, dy), promo = move
#             target_i, target_j = i + dx, j + dy
#             move_score = 0
            
#             # Capture bonus
#             captured_piece, _ = chess_obj.any_piece_at(target_i, target_j)
#             if captured_piece != -1:
#                 attacker = chess_obj.piece_at(i, j, chess_obj.turn)
#                 move_score += 10000 + self.piece_values[captured_piece] - self.piece_values.get(attacker, 0)
            
#             # Promotion bonus
#             if promo != -1:
#                 move_score += 5000 + self.piece_values[promo]
            
#             # Center control bonus
#             move_score += self.piece_table[target_i][target_j] * 10
#             scored_moves.append((move_score, move))
        
#         scored_moves.sort(reverse=True, key=lambda x: x[0])
#         return [m for _, m in scored_moves]

#     # ---------------------------------------------------------------------
#     # Minimax with Alpha-Beta Pruning (root_color added)
#     # ---------------------------------------------------------------------
#     def minimax(self, chess_obj: Chess, depth: int, alpha: float, beta: float, maximizing: bool, root_color: int):
#         self.nodes_searched += 1
        
#         result = chess_obj.game_result()
#         if result is not None:
#             if result == 0:
#                 return 0
#             if result == 1:  # White wins
#                 return 100000 - depth if root_color == 1 else -100000 + depth
#             if result == -1:  # Black wins
#                 return -100000 + depth if root_color == 1 else 100000 - depth
        
#         if depth == 0:
#             return self.evaluate_board(chess_obj, root_color)
        
#         moves, proms = chess_obj.legal_moves()
#         legal_moves = piece_matrix_to_legal_moves(moves, proms)
#         if not legal_moves:
#             return self.evaluate_board(chess_obj, root_color)
        
#         ordered_moves = self.order_moves(chess_obj, legal_moves)
        
#         if maximizing:
#             max_eval = float('-inf')
#             for move in ordered_moves:
#                 (i, j), (dx, dy), promo = move
#                 sim_board = chess_obj.copy()
#                 sim_board.make_move(i, j, dx, dy, promo)
#                 eval_score = self.minimax(sim_board, depth - 1, alpha, beta, False, root_color)
#                 max_eval = max(max_eval, eval_score)
#                 alpha = max(alpha, eval_score)
#                 if beta <= alpha:
#                     break
#             return max_eval
#         else:
#             min_eval = float('inf')
#             for move in ordered_moves:
#                 (i, j), (dx, dy), promo = move
#                 sim_board = chess_obj.copy()
#                 sim_board.make_move(i, j, dx, dy, promo)
#                 eval_score = self.minimax(sim_board, depth - 1, alpha, beta, True, root_color)
#                 min_eval = min(min_eval, eval_score)
#                 beta = min(beta, eval_score)
#                 if beta <= alpha:
#                     break
#             return min_eval

#     # ---------------------------------------------------------------------
#     # Root move selection
#     # ---------------------------------------------------------------------
#     def move(self, chess_obj: Chess):
#         moves, proms = chess_obj.legal_moves()
#         legal_moves = piece_matrix_to_legal_moves(moves, proms)
#         if not legal_moves:
#             return None
#         if len(legal_moves) == 1:
#             return legal_moves[0]
        
#         self.nodes_searched = 0
#         best_move = None
#         best_score = float('-inf')
#         alpha, beta = float('-inf'), float('inf')
        
#         ordered_moves = self.order_moves(chess_obj, legal_moves)
#         root_color = chess_obj.turn
        
#         for move in ordered_moves:
#             (i, j), (dx, dy), promo = move
#             sim_board = chess_obj.copy()
#             sim_board.make_move(i, j, dx, dy, promo)
#             score = self.minimax(sim_board, self.max_depth - 1, alpha, beta, False, root_color)
            
#             if score > best_score:
#                 best_score = score
#                 best_move = move
            
#             alpha = max(alpha, score)
        
#         return best_move if best_move else random.choice(legal_moves)

#     # ---------------------------------------------------------------------
#     def reset(self):
#         self.nodes_searched = 0




















from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves, inv_color

class Task2Agent(BaseAgent):
    def __init__(self, name="OptimizedMiniMaxAgent"):
        super().__init__(name)
        
        # Piece values with refined weights
        self.piece_values = {
            0: 100,   # Pawn
            1: 320,   # Knight
            2: 330,   # Bishop
            3: 500,   # Rook
            4: 900,   # Queen
            5: 20000  # King
        }
        
        # Position bonus tables for 5x4 board (encourage center control)
        self.pawn_table = [
            [0,  0,  0,  0],
            [50, 50, 50, 50],
            [10, 20, 20, 10],
            [5,  10, 10,  5],
            [0,  0,  0,  0]
        ]
        
        self.piece_table = [
            [0,  5,  5,  0],
            [5, 10, 10,  5],
            [5, 15, 15,  5],
            [5, 10, 10,  5],
            [0,  5,  5,  0]
        ]
        
        self.max_depth = 4
        self.nodes_searched = 0
    
    # ---------------------------------------------------------------------
    # Evaluation Function (fixed to be from the agent’s perspective)
    # ---------------------------------------------------------------------
    def evaluate_board(self, chess_obj: Chess, root_color: int):
        """Enhanced board evaluation with consistent agent perspective."""
        
        result = chess_obj.game_result()
        if result is not None:
            if result == 0:
                return 0
            if result == 1:  # White wins
                return 100000 if root_color == 1 else -100000
            if result == -1:  # Black wins
                return -100000 if root_color == 1 else 100000
        
        score = 0
        my_turn = chess_obj.turn
        opp_turn = inv_color(my_turn)
        
        # 1. Material + position bonuses
        for i in range(chess_obj.dims[0]):
            for j in range(chess_obj.dims[1]):
                # My pieces
                my_piece = chess_obj.piece_at(i, j, my_turn)
                if my_piece != -1:
                    score += self.piece_values[my_piece]
                    if my_piece == 0:  # Pawn
                        row = i if my_turn == 1 else (4 - i)
                        score += self.pawn_table[row][j]
                    elif my_piece != 5:
                        score += self.piece_table[i][j]
                
                # Opponent pieces
                opp_piece = chess_obj.piece_at(i, j, opp_turn)
                if opp_piece != -1:
                    score -= self.piece_values[opp_piece]
                    if opp_piece == 0:
                        row = i if opp_turn == 1 else (4 - i)
                        score -= self.pawn_table[row][j]
                    elif opp_piece != 5:
                        score -= self.piece_table[i][j]
        
        # 2. Mobility
        moves, proms = chess_obj.legal_moves()
        mobility = sum(1 for _ in piece_matrix_to_legal_moves(moves, proms))
        score += mobility * 10
        
        # 3. King safety
        if chess_obj.any_checkers:
            score -= 50
        
        # 4. Pawn advancement
        for i in range(chess_obj.dims[0]):
            for j in range(chess_obj.dims[1]):
                pawn = chess_obj.piece_at(i, j, my_turn)
                if pawn == 0:
                    advancement = (4 - i) if my_turn == 1 else i
                    score += advancement * 15
        
        # Flip perspective so that positive = good for root_color
        if chess_obj.turn != root_color:
            score = -score
        
        return score

    # ---------------------------------------------------------------------
    # Move ordering for better alpha-beta pruning
    # ---------------------------------------------------------------------
    def order_moves(self, chess_obj: Chess, legal_moves):
        scored_moves = []
        for move in legal_moves:
            (i, j), (dx, dy), promo = move
            target_i, target_j = i + dx, j + dy
            move_score = 0
            
            # Capture bonus
            captured_piece, _ = chess_obj.any_piece_at(target_i, target_j)
            if captured_piece != -1:
                attacker = chess_obj.piece_at(i, j, chess_obj.turn)
                move_score += 10000 + self.piece_values[captured_piece] - self.piece_values.get(attacker, 0)
            
            # Promotion bonus
            if promo != -1:
                move_score += 5000 + self.piece_values[promo]
            
            # Center control bonus
            move_score += self.piece_table[target_i][target_j] * 10
            scored_moves.append((move_score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored_moves]

    # ---------------------------------------------------------------------
    # Minimax with Alpha-Beta Pruning (root_color added)
    # ---------------------------------------------------------------------
    def minimax(self, chess_obj: Chess, depth: int, alpha: float, beta: float, maximizing: bool, root_color: int):
        self.nodes_searched += 1
        
        result = chess_obj.game_result()
        if result is not None:
            if result == 0:
                return 0
            if result == 1:  # White wins
                return 100000 - depth if root_color == 1 else -100000 + depth
            if result == -1:  # Black wins
                return -100000 + depth if root_color == 1 else 100000 - depth
        
        if depth == 0:
            return self.evaluate_board(chess_obj, root_color)
        
        moves, proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        if not legal_moves:
            return self.evaluate_board(chess_obj, root_color)
        
        ordered_moves = self.order_moves(chess_obj, legal_moves)
        
        if maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                (i, j), (dx, dy), promo = move
                sim_board = chess_obj.copy()
                sim_board.make_move(i, j, dx, dy, promo)
                eval_score = self.minimax(sim_board, depth - 1, alpha, beta, False, root_color)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                (i, j), (dx, dy), promo = move
                sim_board = chess_obj.copy()
                sim_board.make_move(i, j, dx, dy, promo)
                eval_score = self.minimax(sim_board, depth - 1, alpha, beta, True, root_color)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    # ---------------------------------------------------------------------
    # Root move selection
    # ---------------------------------------------------------------------
    def move(self, chess_obj: Chess):
        moves, proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        self.nodes_searched = 0
        best_move = None
        best_score = float('-inf')
        alpha, beta = float('-inf'), float('inf')
        
        ordered_moves = self.order_moves(chess_obj, legal_moves)
        root_color = chess_obj.turn
        
        for move in ordered_moves:
            (i, j), (dx, dy), promo = move
            sim_board = chess_obj.copy()
            sim_board.make_move(i, j, dx, dy, promo)
            score = self.minimax(sim_board, self.max_depth - 1, alpha, beta, False, root_color)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_move if best_move else random.choice(legal_moves)

    # ---------------------------------------------------------------------
    def reset(self):
        self.nodes_searched = 0