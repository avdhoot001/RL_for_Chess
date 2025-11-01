
from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
import numpy as np
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves

class Task1Agent(BaseAgent):
    def __init__(self, name="MiniMaxAgent"):
        super().__init__(name)

    def evaluate_board(self, chess_obj: Chess):
        """Simplified static board evaluation using piece_at()"""
        piece_values = {0: 100, 1: 320, 2: 330, 3: 500, 4: 900, 5: 20000}
        score = 0
        center = (chess_obj.dims[0] / 2, chess_obj.dims[1] / 2)

        for color in [0, 1]:
            sign = 1 if color == chess_obj.turn else -1  # we evaluate from current player's POV
            for i in range(chess_obj.dims[0]):
                for j in range(chess_obj.dims[1]):
                    piece = chess_obj.piece_at(i, j, color)
                    if piece != -1:
                        dist = abs(i - center[0]) + abs(j - center[1])
                        score += sign * (piece_values.get(piece, 0) + (5 - dist))
        return score

    def move(self, chess_obj: Chess):
        moves, proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)

        if not legal_moves:
            return None

        best_move = None
        best_score = -float("inf")

        for origin, deltas, promo in legal_moves:
            i, j = origin
            dx, dy = deltas

            # simulate our move
            sim_board = chess_obj.copy()
            sim_board.make_move(i, j, dx, dy, promo)

            # opponent's replies
            opp_moves, opp_proms = sim_board.legal_moves()
            opp_legal = piece_matrix_to_legal_moves(opp_moves, opp_proms)

            if not opp_legal:
                # opponent has no moves (likely checkmate)
                score = 999999 if sim_board.turn != chess_obj.turn else -999999
            else:
                # assume opponent picks the move that minimizes our evaluation
                worst_reply_score = float("inf")
                for o_origin, o_deltas, o_promo in random.sample(opp_legal, min(10, len(opp_legal))):
                    oi, oj = o_origin
                    odx, ody = o_deltas
                    opp_board = sim_board.copy()
                    opp_board.make_move(oi, oj, odx, ody, o_promo)
                    eval_score = self.evaluate_board(opp_board)
                    worst_reply_score = min(worst_reply_score, eval_score)
                score = worst_reply_score

            if score > best_score:
                best_score = score
                best_move = ((i, j), (dx, dy), promo)

        # fallback
        if best_move is None:
            best_move = random.choice(legal_moves)

        return best_move

    def reset(self):
        pass









































# from minichess.chess.fastchess import Chess
# from .base_agent import BaseAgent
# import random
# from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves, inv_color

# class Task1Agent(BaseAgent):
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
#         # Rows 0-4, Cols 0-3
#         self.pawn_table = [
#             [0,  0,  0,  0],   # Row 0 (promotion row)
#             [50, 50, 50, 50],  # Row 1
#             [10, 20, 20, 10],  # Row 2
#             [5,  10, 10,  5],  # Row 3
#             [0,  0,  0,  0]    # Row 4 (back row)
#         ]
        
#         self.piece_table = [
#             [0,  5,  5,  0],   # Row 0
#             [5, 10, 10,  5],   # Row 1
#             [5, 15, 15,  5],   # Row 2 (center rows more valuable)
#             [5, 10, 10,  5],   # Row 3
#             [0,  5,  5,  0]    # Row 4
#         ]
        
#         self.max_depth = 4
#         self.nodes_searched = 0
    
#     def evaluate_board(self, chess_obj: Chess):
#         """Enhanced board evaluation with multiple factors"""
        
#         # Check for terminal states first
#         result = chess_obj.game_result()
#         if result is not None:
#             if result == 0:  # Draw
#                 return 0
#             # Win/loss from current player's perspective
#             if (result == 1 and chess_obj.turn == 1) or (result == -1 and chess_obj.turn == 0):
#                 return -100000  # We lost (opponent won)
#             else:
#                 return 100000   # We won
        
#         score = 0
#         my_turn = chess_obj.turn
#         opp_turn = inv_color(my_turn)
        
#         # 1. Material count with position bonuses
#         for i in range(chess_obj.dims[0]):
#             for j in range(chess_obj.dims[1]):
#                 # Check my pieces
#                 my_piece = chess_obj.piece_at(i, j, my_turn)
#                 if my_piece != -1:
#                     score += self.piece_values[my_piece]
#                     # Add positional bonus
#                     if my_piece == 0:  # Pawn
#                         # For black (turn=0), flip the table vertically
#                         row = i if my_turn == 1 else (4 - i)
#                         score += self.pawn_table[row][j]
#                     elif my_piece != 5:  # Not king
#                         score += self.piece_table[i][j]
                
#                 # Check opponent pieces
#                 opp_piece = chess_obj.piece_at(i, j, opp_turn)
#                 if opp_piece != -1:
#                     score -= self.piece_values[opp_piece]
#                     # Subtract opponent's positional bonus
#                     if opp_piece == 0:  # Pawn
#                         row = i if opp_turn == 1 else (4 - i)
#                         score -= self.pawn_table[row][j]
#                     elif opp_piece != 5:  # Not king
#                         score -= self.piece_table[i][j]
        
#         # 2. Mobility (number of legal moves)
#         moves, proms = chess_obj.legal_moves()
#         my_mobility = sum(1 for m in piece_matrix_to_legal_moves(moves, proms))
#         score += my_mobility * 10
        
#         # 3. King safety - penalize if in check
#         if chess_obj.any_checkers:
#             score -= 50
        
#         # 4. Pawn advancement bonus (closer to promotion)
#         for i in range(chess_obj.dims[0]):
#             for j in range(chess_obj.dims[1]):
#                 my_pawn = chess_obj.piece_at(i, j, my_turn)
#                 if my_pawn == 0:
#                     # Distance to promotion row
#                     if my_turn == 1:  # White pawns move up (towards row 0)
#                         advancement = 4 - i
#                     else:  # Black pawns move down (towards row 4)
#                         advancement = i
#                     score += advancement * 15
        
#         return score
    
#     def order_moves(self, chess_obj: Chess, legal_moves):
#         """
#         Move ordering heuristic for better alpha-beta pruning.
#         Prioritize: captures > promotions > center moves > others
#         """
#         scored_moves = []
        
#         for move in legal_moves:
#             (i, j), (dx, dy), promo = move
#             target_i, target_j = i + dx, j + dy
#             move_score = 0
            
#             # 1. Prioritize captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
#             captured_piece, _ = chess_obj.any_piece_at(target_i, target_j)
#             if captured_piece != -1:
#                 attacker = chess_obj.piece_at(i, j, chess_obj.turn)
#                 move_score += 10000 + self.piece_values[captured_piece] - self.piece_values.get(attacker, 0)
            
#             # 2. Prioritize promotions
#             if promo != -1:
#                 move_score += 5000 + self.piece_values[promo]
            
#             # 3. Prioritize center control
#             center_bonus = self.piece_table[target_i][target_j]
#             move_score += center_bonus * 10
            
#             scored_moves.append((move_score, move))
        
#         # Sort by score (descending)
#         scored_moves.sort(reverse=True, key=lambda x: x[0])
#         return [move for _, move in scored_moves]
    
#     def minimax(self, chess_obj: Chess, depth: int, alpha: float, beta: float, maximizing: bool):
#         """
#         Minimax with alpha-beta pruning
#         """
#         self.nodes_searched += 1
        
#         # Terminal conditions
#         result = chess_obj.game_result()
#         if result is not None:
#             if result == 0:
#                 return 0
#             # From maximizing player's perspective
#             if (result == 1 and maximizing) or (result == -1 and not maximizing):
#                 return 100000 - depth  # Prefer quicker wins
#             else:
#                 return -100000 + depth  # Prefer delayed losses
        
#         # Depth limit reached
#         if depth == 0:
#             return self.evaluate_board(chess_obj)
        
#         # Get and order legal moves
#         moves, proms = chess_obj.legal_moves()
#         legal_moves = piece_matrix_to_legal_moves(moves, proms)
        
#         if not legal_moves:
#             # No legal moves but game not over (shouldn't happen, but safe check)
#             return self.evaluate_board(chess_obj)
        
#         # Order moves for better pruning
#         ordered_moves = self.order_moves(chess_obj, legal_moves)
        
#         if maximizing:
#             max_eval = float('-inf')
#             for move in ordered_moves:
#                 (i, j), (dx, dy), promo = move
                
#                 # Make move
#                 sim_board = chess_obj.copy()
#                 sim_board.make_move(i, j, dx, dy, promo)
                
#                 # Recursive call
#                 eval_score = self.minimax(sim_board, depth - 1, alpha, beta, False)
#                 max_eval = max(max_eval, eval_score)
#                 alpha = max(alpha, eval_score)
                
#                 # Alpha-beta pruning
#                 if beta <= alpha:
#                     break  # Beta cutoff
            
#             return max_eval
#         else:
#             min_eval = float('inf')
#             for move in ordered_moves:
#                 (i, j), (dx, dy), promo = move
                
#                 # Make move
#                 sim_board = chess_obj.copy()
#                 sim_board.make_move(i, j, dx, dy, promo)
                
#                 # Recursive call
#                 eval_score = self.minimax(sim_board, depth - 1, alpha, beta, True)
#                 min_eval = min(min_eval, eval_score)
#                 beta = min(beta, eval_score)
                
#                 # Alpha-beta pruning
#                 if beta <= alpha:
#                     break  # Alpha cutoff
            
#             return min_eval
    
#     def move(self, chess_obj: Chess):
#         """Select best move using minimax with alpha-beta pruning"""
#         moves, proms = chess_obj.legal_moves()
#         legal_moves = piece_matrix_to_legal_moves(moves, proms)
        
#         if not legal_moves:
#             return None
        
#         if len(legal_moves) == 1:
#             return legal_moves[0]
        
#         # Reset search counter
#         self.nodes_searched = 0
        
#         best_move = None
#         best_score = float('-inf')
#         alpha = float('-inf')
#         beta = float('inf')
        
#         # Order moves for better pruning at root
#         ordered_moves = self.order_moves(chess_obj, legal_moves)
        
#         # Evaluate each move
#         for move in ordered_moves:
#             (i, j), (dx, dy), promo = move
            
#             # Make move
#             sim_board = chess_obj.copy()
#             sim_board.make_move(i, j, dx, dy, promo)
            
#             # Use minimax to evaluate (opponent's turn, so minimizing)
#             score = self.minimax(sim_board, self.max_depth - 1, alpha, beta, False)
            
#             if score > best_score:
#                 best_score = score
#                 best_move = move
            
#             alpha = max(alpha, score)
        
#         # Fallback to random if something goes wrong
#         return best_move if best_move else random.choice(legal_moves)
    
#     def reset(self):
#         """Reset internal state"""
#         self.nodes_searched = 0





































































































# from minichess.chess.fastchess import Chess
# from .base_agent import BaseAgent
# import random
# import numpy as np
# from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves

# class Task1Agent(BaseAgent):
#     def _init_(self, name="MiniMaxAgent"):
#         super()._init_(name)

#     def evaluate_board(self, chess_obj: Chess):
#         """Simplified static board evaluation using piece_at()"""
#         piece_values = {0: 100, 1: 320, 2: 330, 3: 500, 4: 900, 5: 20000}
#         score = 0
#         center = (chess_obj.dims[0] / 2, chess_obj.dims[1] / 2)

#         for color in [0, 1]:
#             sign = 1 if color == chess_obj.turn else -1  # we evaluate from current player's POV
#             for i in range(chess_obj.dims[0]):
#                 for j in range(chess_obj.dims[1]):
#                     piece = chess_obj.piece_at(i, j, color)
#                     if piece != -1:
#                         dist = abs(i - center[0]) + abs(j - center[1])
#                         score += sign * (piece_values.get(piece, 0) + (5 - dist))
#         return score

#     def move(self, chess_obj: Chess):
#         moves, proms = chess_obj.legal_moves()
#         legal_moves = piece_matrix_to_legal_moves(moves, proms)

#         if not legal_moves:
#             return None

#         best_move = None
#         best_score = -float("inf")

#         for origin, deltas, promo in legal_moves:
#             i, j = origin
#             dx, dy = deltas

#             # simulate our move
#             sim_board = chess_obj.copy()
#             sim_board.make_move(i, j, dx, dy, promo)

#             # opponent's replies
#             opp_moves, opp_proms = sim_board.legal_moves()
#             opp_legal = piece_matrix_to_legal_moves(opp_moves, opp_proms)

#             if not opp_legal:
#                 # opponent has no moves (likely checkmate)
#                 score = 999999 if sim_board.turn != chess_obj.turn else -999999
#             else:
#                 # assume opponent picks the move that minimizes our evaluation
#                 worst_reply_score = float("inf")
#                 for o_origin, o_deltas, o_promo in random.sample(opp_legal, min(10, len(opp_legal))):
#                     oi, oj = o_origin
#                     odx, ody = o_deltas
#                     opp_board = sim_board.copy()
#                     opp_board.make_move(oi, oj, odx, ody, o_promo)
#                     eval_score = self.evaluate_board(opp_board)
#                     worst_reply_score = min(worst_reply_score, eval_score)
#                 score = worst_reply_score

#             if score > best_score:
#                 best_score = score
#                 best_move = ((i, j), (dx, dy), promo)

#         # fallback
#         if best_move is None:
#             best_move = random.choice(legal_moves)

#         return best_move

#     def reset(self):
#         pass
