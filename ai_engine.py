import random

class ChessAI:
    def __init__(self):
        self.level = 1
        # Điểm quân cờ cơ bản
        self.piece_values = {'Pawn': 100, 'Knight': 320, 'Bishop': 330, 'Rook': 500, 'Queen': 900, 'King': 20000}
        
        # Bảng vị trí (PST) - Giữ nguyên logic cũ của bạn
        self.pst = {
            'Pawn': [ 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5,-10, 0, 0,-10, -5, 5, 5, 10, 10,-20,-20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0 ],
            'Knight': [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20, 0, 0, 0, 0,-20,-40, -30, 0, 10, 15, 15, 10, 0,-30, -30, 5, 15, 20, 20, 15, 5,-30, -30, 0, 15, 20, 20, 15, 0,-30, -30, 5, 10, 15, 15, 10, 5,-30, -40,-20, 0, 5, 5, 0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50]
        }

    def set_difficulty(self, level):
        self.level = level

    def get_best_move(self, game, color):
        # Lấy tất cả nước đi
        all_moves = self.get_all_valid_moves(game, color)
        if not all_moves: return None
        
        # Level 1: Random (Dễ)
        if self.level == 1: return random.choice(all_moves)
        
        # Level 2: Ăn tham (Trung bình)
        if self.level == 2: return self.get_greedy_move(game, all_moves)

        # Level 3 (Hard): Depth = 2
        # Level 4 (Expert): Depth = 3 (Depth 4 Python sẽ bị lag)
        depth = 2 if self.level == 3 else 3
        
        return self.minimax_root(game, depth, True)

    def get_greedy_move(self, game, moves):
        random.shuffle(moves)
        best_move = moves[0]
        max_val = -99999
        for piece, r, c in moves:
            target = game.logic_board[r][c]
            val = self.piece_values.get(type(target).__name__, 0) if target else 0
            if val > max_val:
                max_val = val
                best_move = (piece, r, c)
        return best_move

    # --- CẢI TIẾN 1: SẮP XẾP NƯỚC ĐI (MOVE ORDERING) ---
    def order_moves(self, game, moves):
        """
        Sắp xếp các nước đi: Ưu tiên nước ĂN QUÂN trước.
        Giúp Alpha-Beta cắt tỉa nhanh hơn gấp nhiều lần.
        """
        scored_moves = []
        for move in moves:
            piece, r, c = move
            score = 0
            
            # Nếu nước đi này ăn quân địch
            target = game.logic_board[r][c]
            if target:
                # Ăn quân giá trị càng cao càng ưu tiên (MVV-LVA logic)
                victim_val = self.piece_values.get(type(target).__name__, 0)
                attacker_val = self.piece_values.get(type(piece).__name__, 0)
                # Công thức: Lấy quân giá trị cao - quân mình dùng để ăn
                score = 10 * victim_val - attacker_val
            
            # Ưu tiên phong cấp (Promotion) - Nếu là Tốt đi xuống cuối bàn
            if type(piece).__name__ == 'Pawn':
                if (piece.color == 'white' and r == 0) or (piece.color == 'black' and r == 7):
                    score += 900 # Ưu tiên bằng việc ăn Hậu
            
            scored_moves.append((score, move))
            
        # Sắp xếp giảm dần (Điểm cao xếp trước)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Trả về danh sách nước đi đã sắp xếp
        return [m[1] for m in scored_moves]

    def evaluate_board(self, board, turn_color):
        white_score = 0
        black_score = 0
        
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p:
                    # 1. Điểm chất (Material)
                    val = self.piece_values.get(type(p).__name__, 0)
                    
                    # 2. Điểm vị trí (Position)
                    pst_val = 0
                    if type(p).__name__ in self.pst:
                        idx = (7-r)*8 + c if p.color == 'white' else r*8 + c
                        pst_val = self.pst[type(p).__name__][idx]
                    
                    # --- CẢI TIẾN 2: MOBILITY (ĐƠN GIẢN HÓA) ---
                    # Cộng điểm nhỏ nếu quân nằm ở trung tâm (kiểm soát tốt hơn)
                    # (Tính mobility thật sự rất chậm, nên ta dùng mẹo vị trí)
                    mobility = 0
                    if 2 <= r <= 5 and 2 <= c <= 5: # Vùng trung tâm
                        mobility = 10 

                    total = val + pst_val + mobility
                    
                    if p.color == 'white': white_score += total
                    else: black_score += total
        
        # Trả về điểm lợi thế cho phe đang tính toán
        eval_score = black_score - white_score
        # Nếu đang là lượt mình đi, mình muốn điểm cao. Lượt địch đi, địch muốn điểm thấp.
        return eval_score

    def minimax_root(self, game, depth, is_maximizing):
        best_move = None
        best_value = -99999
        
        moves = self.get_all_valid_moves(game, 'black')
        
        # 🔥 ÁP DỤNG MOVE ORDERING
        moves = self.order_moves(game, moves)
        
        alpha = -100000
        beta = 100000
        
        for piece, r, c in moves:
            original_target = game.logic_board[r][c]
            old_r, old_c = piece.row, piece.col
            
            # Make move
            game.logic_board[old_r][old_c] = None
            game.logic_board[r][c] = piece
            piece.row, piece.col = r, c
            
            value = self.minimax(game, depth - 1, alpha, beta, False)
            
            # Undo move
            piece.row, piece.col = old_r, old_c
            game.logic_board[old_r][old_c] = piece
            game.logic_board[r][c] = original_target
            
            if value > best_value:
                best_value = value
                best_move = (piece, r, c)
            
            alpha = max(alpha, best_value)
            if beta <= alpha: break
                
        # Fallback nếu không tìm được nước nào (hiếm gặp)
        if not best_move and moves: best_move = moves[0]
        return best_move

    def minimax(self, game, depth, alpha, beta, is_maximizing):
        if depth == 0:
            return self.evaluate_board(game.logic_board, 'black')

        if is_maximizing:
            max_eval = -99999
            moves = self.get_all_valid_moves(game, 'black')
            # 🔥 Sắp xếp nước đi để cắt tỉa nhanh hơn
            moves = self.order_moves(game, moves) 
            
            for piece, r, c in moves:
                original_target = game.logic_board[r][c]
                old_r, old_c = piece.row, piece.col
                
                game.logic_board[old_r][old_c] = None
                game.logic_board[r][c] = piece
                piece.row, piece.col = r, c
                
                eval = self.minimax(game, depth - 1, alpha, beta, False)
                
                piece.row, piece.col = old_r, old_c
                game.logic_board[old_r][old_c] = piece
                game.logic_board[r][c] = original_target
                
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = 99999
            moves = self.get_all_valid_moves(game, 'white')
            moves = self.order_moves(game, moves) # 🔥 Sắp xếp
            
            for piece, r, c in moves:
                original_target = game.logic_board[r][c]
                old_r, old_c = piece.row, piece.col
                
                game.logic_board[old_r][old_c] = None
                game.logic_board[r][c] = piece
                piece.row, piece.col = r, c
                
                eval = self.minimax(game, depth - 1, alpha, beta, True)
                
                piece.row, piece.col = old_r, old_c
                game.logic_board[old_r][old_c] = piece
                game.logic_board[r][c] = original_target
                
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

    def get_all_valid_moves(self, game, color):
        moves = []
        for r in range(8):
            for c in range(8):
                p = game.logic_board[r][c]
                if p and p.color == color:
                    valid_rcs = game.get_valid_moves(p)
                    for tr, tc in valid_rcs:
                        moves.append((p, tr, tc))
        return moves