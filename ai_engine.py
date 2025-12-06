import random
import numpy as np
from numba import jit, int8, int32

# ==========================================
# PHẦN 1: CẤU HÌNH & DỮ LIỆU TĨNH
# ==========================================
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

# Material Values
PIECE_VALUES = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int32)

# Buffer limit
MAX_MOVES = 300

# Khởi tạo PST
PST = np.zeros((7, 8, 8), dtype=np.int32)

# --- NẠP DỮ LIỆU PST ---
# Đảm bảo mỗi hàng có đúng 8 số, tổng cộng 64 số
pawn_pst_raw = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]
PST[PAWN] = np.array(pawn_pst_raw, dtype=np.int32).reshape(8, 8)

# Fix lỗi thiếu phần tử ở đây (đã kiểm tra kỹ 64 số)
knight_pst_raw = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]
PST[KNIGHT] = np.array(knight_pst_raw, dtype=np.int32).reshape(8, 8)

center_bias = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5, 10, 20, 20, 10,  5,-10,
    -10,  5, 10, 20, 20, 10,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
], dtype=np.int32).reshape(8, 8)

# --- SỬA LẠI PST CHO VUA (KING SAFETY) ---
# Dữ liệu gốc của bạn ưu tiên Hàng 0 (đầu mảng).
# Nhưng Vua Trắng đứng ở Hàng 7 (cuối mảng).
# => Cần lật ngược bảng này (np.flip) để Hàng 7 nhận điểm cao.
king_safety_pst = np.array([
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30
], dtype=np.int32).reshape(8, 8)

PST[BISHOP] = center_bias
PST[ROOK] = center_bias // 2
PST[QUEEN] = center_bias // 2
# 🔥 FIX QUAN TRỌNG: Lật ngược bảng điểm Vua
# Bây giờ Hàng 7 (nhà của Trắng) sẽ ứng với dòng đầu tiên (20, 30...) -> Điểm cao
PST[KING] = np.flip(king_safety_pst, axis=0) 


# ==========================================
# PHẦN 2: LÕI NUMBA (ENGINE)
# ==========================================

@jit(nopython=True, cache=False)
def is_on_board(r, c):
    return 0 <= r < 8 and 0 <= c < 8

@jit(nopython=True, cache=False)
def is_square_attacked_numba(board, r, c, attacker_color):
    """Kiểm tra ô (r, c) có bị phe attacker_color tấn công không."""
    # 1. Pawn
    pawn_dir = -1 if attacker_color == 1 else 1
    check_r = r - pawn_dir
    if 0 <= check_r < 8:
        for dc in [-1, 1]:
            check_c = c + dc
            if 0 <= check_c < 8:
                p = board[check_r, check_c]
                if p != 0 and abs(p) == PAWN and (p * attacker_color > 0): return True

    # 2. Knight
    knight_offsets = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
    for dr, dc in knight_offsets:
        nr, nc = r + dr, c + dc
        if is_on_board(nr, nc):
            p = board[nr, nc]
            if p != 0 and abs(p) == KNIGHT and (p * attacker_color > 0): return True

    # 3. King
    king_offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for dr, dc in king_offsets:
        nr, nc = r + dr, c + dc
        if is_on_board(nr, nc):
            p = board[nr, nc]
            if p != 0 and abs(p) == KING and (p * attacker_color > 0): return True
                
    # 4. Rook/Queen
    straight_dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    for dr, dc in straight_dirs:
        nr, nc = r + dr, c + dc
        while is_on_board(nr, nc):
            p = board[nr, nc]
            if p != 0:
                if (p * attacker_color > 0) and (abs(p) == ROOK or abs(p) == QUEEN): return True
                break
            nr += dr
            nc += dc
            
    # 5. Bishop/Queen
    diag_dirs = [(-1,-1), (-1,1), (1,-1), (1,1)]
    for dr, dc in diag_dirs:
        nr, nc = r + dr, c + dc
        while is_on_board(nr, nc):
            p = board[nr, nc]
            if p != 0:
                if (p * attacker_color > 0) and (abs(p) == BISHOP or abs(p) == QUEEN): return True
                break
            nr += dr
            nc += dc

    return False

@jit(nopython=True, cache=False)
def get_legal_moves_numba(board, color, en_passant_col, can_castle, only_captures):
    """
    Sinh nước đi hợp lệ.
    """
    moves = np.zeros((MAX_MOVES, 5), dtype=np.int8) 
    count = 0
    
    # Tìm Vua
    my_king_r, my_king_c = -1, -1
    for r in range(8):
        for c in range(8):
            if board[r, c] == KING * color:
                my_king_r, my_king_c = r, c
                break
        if my_king_r != -1: break
    if my_king_r == -1: return moves[:0]

    opponent_color = -1 if color == 1 else 1

    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece == 0 or (piece * color <= 0): continue
            p_type = abs(piece)
            
            targets = np.zeros((28, 3), dtype=np.int8)
            t_count = 0
            
            # --- PAWN ---
            if p_type == PAWN:
                direction = -1 if color == 1 else 1
                start_row = 6 if color == 1 else 1
                
                # Di chuyển thẳng
                if not only_captures:
                    nr, nc = r + direction, c
                    if is_on_board(nr, nc) and board[nr, nc] == 0:
                        # KIỂM TRA PHONG CẤP (Promotion)
                        is_promo = (color == 1 and nr == 0) or (color == -1 and nr == 7)
                        if is_promo:
                            # Sinh 4 nước đi cho 4 loại quân
                            targets[t_count] = [nr, nc, 3]; t_count += 1 # 3 = Queen
                            targets[t_count] = [nr, nc, 4]; t_count += 1 # 4 = Knight
                            targets[t_count] = [nr, nc, 5]; t_count += 1 # 5 = Rook
                            targets[t_count] = [nr, nc, 6]; t_count += 1 # 6 = Bishop
                        else:
                            targets[t_count] = [nr, nc, 0]; t_count += 1
                            
                        # Nước đi đôi (Double push)
                        nnr = r + 2 * direction
                        if r == start_row and is_on_board(nnr, nc) and board[nnr, nc] == 0:
                            targets[t_count] = [nnr, nc, 0]; t_count += 1
                
                # Ăn chéo
                for dc in [-1, 1]:
                    nr, nc = r + direction, c + dc
                    if is_on_board(nr, nc):
                        target_p = board[nr, nc]
                        # Ăn thường
                        if target_p != 0 and (target_p * color < 0):
                            # KIỂM TRA PHONG CẤP KHI ĂN
                            is_promo = (color == 1 and nr == 0) or (color == -1 and nr == 7)
                            if is_promo:
                                targets[t_count] = [nr, nc, 3]; t_count += 1
                                targets[t_count] = [nr, nc, 4]; t_count += 1
                                targets[t_count] = [nr, nc, 5]; t_count += 1
                                targets[t_count] = [nr, nc, 6]; t_count += 1
                            else:
                                targets[t_count] = [nr, nc, 0]; t_count += 1
                        
                        # En Passant (Giữ nguyên logic cũ)
                        en_passant_rank = 3 if color == 1 else 4
                        if r == en_passant_rank and nc == en_passant_col and target_p == 0:
                            targets[t_count] = [nr, nc, 1]; t_count += 1

            # --- KNIGHT/KING/SLIDING ---
            else:
                offsets = [] 
                is_sliding = False
                
                if p_type == KNIGHT:
                    offsets = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
                elif p_type == KING:
                    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                else: # Sliding
                    is_sliding = True
                    if p_type == BISHOP or p_type == QUEEN: offsets.extend([(-1,-1), (-1,1), (1,-1), (1,1)])
                    if p_type == ROOK or p_type == QUEEN: offsets.extend([(-1,0), (1,0), (0,-1), (0,1)])

                if not is_sliding: # Knight, King
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        if is_on_board(nr, nc):
                            tgt = board[nr, nc]
                            is_capture = (tgt != 0 and tgt * color < 0)
                            if is_capture:
                                targets[t_count] = [nr, nc, 0]; t_count += 1
                            elif not only_captures and tgt == 0:
                                targets[t_count] = [nr, nc, 0]; t_count += 1
                else: # Sliding
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        while is_on_board(nr, nc):
                            tgt = board[nr, nc]
                            if tgt == 0:
                                if not only_captures:
                                    targets[t_count] = [nr, nc, 0]; t_count += 1
                            elif tgt * color < 0: # Capture
                                targets[t_count] = [nr, nc, 0]; t_count += 1
                                break
                            else: break 
                            nr += dr
                            nc += dc

                # Castling
                if p_type == KING and not only_captures:
                    idx_offset = 0 if color == 1 else 2
                    row_idx = 7 if color == 1 else 0
                    
                    if can_castle[idx_offset] and board[row_idx, 5] == 0 and board[row_idx, 6] == 0:
                        if not is_square_attacked_numba(board, row_idx, 4, opponent_color) and \
                           not is_square_attacked_numba(board, row_idx, 5, opponent_color) and \
                           not is_square_attacked_numba(board, row_idx, 6, opponent_color):
                            targets[t_count] = [row_idx, 6, 2]; t_count += 1
                    
                    if can_castle[idx_offset + 1] and board[row_idx, 1] == 0 and board[row_idx, 2] == 0 and board[row_idx, 3] == 0:
                        if not is_square_attacked_numba(board, row_idx, 4, opponent_color) and \
                           not is_square_attacked_numba(board, row_idx, 3, opponent_color) and \
                           not is_square_attacked_numba(board, row_idx, 2, opponent_color):
                            targets[t_count] = [row_idx, 2, 2]; t_count += 1

            # --- MAKE-CHECK-UNMAKE ---
            for i in range(t_count):
                tr, tc, flag = targets[i]
                
                captured = board[tr, tc]
                old_piece = board[r, c]
                ep_captured_r, ep_captured_c = -1, -1
                ep_captured_val = 0
                
                board[tr, tc] = old_piece
                board[r, c] = 0
                
                if flag == 1: 
                    board[r, tc] = 0
                    ep_captured_r, ep_captured_c = r, tc
                    ep_captured_val = PAWN * opponent_color

                cur_king_r, cur_king_c = my_king_r, my_king_c
                if p_type == KING: cur_king_r, cur_king_c = tr, tc
                
                if not is_square_attacked_numba(board, cur_king_r, cur_king_c, opponent_color):
                    if count < MAX_MOVES:
                        moves[count] = [r, c, tr, tc, flag]
                        count += 1
                
                board[r, c] = old_piece
                board[tr, tc] = captured
                if flag == 1: board[ep_captured_r, ep_captured_c] = ep_captured_val

    return moves[:count]

@jit(nopython=True, cache=False)
def evaluate_board_numba(board):
    score = 0
    for r in range(8):
        for c in range(8):
            p = board[r, c]
            if p == 0: continue
            p_type = abs(p)
            sign = 1 if p > 0 else -1
            material = PIECE_VALUES[p_type]
            pst_r = r if sign == 1 else 7 - r
            position = PST[p_type, pst_r, c]
            score += sign * (material + position)
    return score

# ==========================================
# QUIESCENCE SEARCH (TÌM KIẾM YÊN TĨNH)
# ==========================================
@jit(nopython=True, cache=False)
def q_search_negamax(board, alpha, beta, color, node_count, node_limit):
    # --- CHECK NODE LIMIT ---
    node_count[0] += 1
    if node_count[0] >= node_limit:
        return evaluate_board_numba(board) * color # Trả về điểm ngay lập tức

    stand_pat = evaluate_board_numba(board) * color
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
        
    dummy_castle = np.array([False, False, False, False], dtype=np.bool_)
    moves = get_legal_moves_numba(board, color, -1, dummy_castle, True)
    
    move_scores = np.zeros(len(moves), dtype=np.int32)
    for i in range(len(moves)):
        target = board[moves[i, 2], moves[i, 3]]
        if target != 0:
            attacker = board[moves[i, 0], moves[i, 1]]
            move_scores[i] = 10 * PIECE_VALUES[abs(target)] - PIECE_VALUES[abs(attacker)]
    sorted_indices = np.argsort(move_scores)[::-1]
    
    for i in sorted_indices:
        move = moves[i]
        r1, c1, r2, c2, flag = move
        captured = board[r2, c2]
        moving_piece = board[r1, c1]
        
        board[r2, c2] = moving_piece
        board[r1, c1] = 0
        
        # --- LOGIC PHONG CẤP MỚI ---
        promoted = False
        if flag >= 3:
            promoted = True
            new_type = QUEEN
            if flag == 4: new_type = KNIGHT
            elif flag == 5: new_type = ROOK
            elif flag == 6: new_type = BISHOP
            board[r2, c2] = new_type * color
        # ----------------------------
            
        score = -q_search_negamax(board, -beta, -alpha, -color, node_count, node_limit)
        
        # Undo move
        board[r1, c1] = moving_piece
        board[r2, c2] = captured
        if promoted: board[r1, c1] = PAWN * color # Trả lại quân Tốt gốc
        
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
            
    return alpha

# ==========================================
# MINIMAX ENGINE (MAIN)
# ==========================================
@jit(nopython=True, cache=False)
def minimax_numba_engine(board, depth, alpha, beta, is_maximizing, en_passant_col, can_castle, node_count, node_limit):
    # --- CHECK NODE LIMIT ---
    node_count[0] += 1
    if node_count[0] >= node_limit:
        return evaluate_board_numba(board) * (1 if is_maximizing else -1), np.zeros(5, dtype=np.int8)

    color = 1 if is_maximizing else -1
    
    if depth == 0:
        val = q_search_negamax(board, alpha, beta, color, node_count, node_limit)
        return val if is_maximizing else -val, np.zeros(5, dtype=np.int8)

    moves = get_legal_moves_numba(board, color, en_passant_col, can_castle, False)
    
    if len(moves) == 0:
        king_r, king_c = -1, -1
        for r in range(8):
            for c in range(8):
                if board[r, c] == KING * color:
                    king_r, king_c = r, c
                    break
        in_check = False
        if king_r != -1:
            in_check = is_square_attacked_numba(board, king_r, king_c, -color)
            
        if in_check: return (-999999 - depth) if is_maximizing else (999999 + depth), np.zeros(5, dtype=np.int8)
        else: return 0, np.zeros(5, dtype=np.int8)

    move_scores = np.zeros(len(moves), dtype=np.int32)
    for i in range(len(moves)):
        target = board[moves[i, 2], moves[i, 3]]
        if target != 0:
            attacker = board[moves[i, 0], moves[i, 1]]
            move_scores[i] = 10 * PIECE_VALUES[abs(target)] - PIECE_VALUES[abs(attacker)]
    sorted_indices = np.argsort(move_scores)[::-1]
    
    best_move = np.zeros(5, dtype=np.int8)
    
    if is_maximizing: # WHITE
        max_eval = -999999999
        for i in sorted_indices:
            move = moves[i]
            r1, c1, r2, c2, flag = move
            
            captured = board[r2, c2]
            moving_piece = board[r1, c1]
            ep_backup_val = 0
            board[r2, c2] = moving_piece
            board[r1, c1] = 0
            
            # --- LOGIC PHONG CẤP MỚI (WHITE) ---
            promoted = False
            if flag >= 3:
                promoted = True
                new_type = QUEEN
                if flag == 4: new_type = KNIGHT
                elif flag == 5: new_type = ROOK
                elif flag == 6: new_type = BISHOP
                board[r2, c2] = new_type * color # color là 1
            # -----------------------------------
            
            if flag == 1:
                ep_backup_val = board[r1, c2]
                board[r1, c2] = 0
                
            castle_rook_backup = 0
            if flag == 2:
                if c2 == 6: 
                    castle_rook_backup = board[7, 7]; board[7, 5] = castle_rook_backup; board[7, 7] = 0
                elif c2 == 2:
                    castle_rook_backup = board[7, 0]; board[7, 3] = castle_rook_backup; board[7, 0] = 0

            eval_val, _ = minimax_numba_engine(board, depth - 1, alpha, beta, False, -1, can_castle, node_count, node_limit)
            
            board[r1, c1] = moving_piece
            board[r2, c2] = captured
            if promoted: board[r1, c1] = PAWN * color
            if flag == 1: board[r1, c2] = ep_backup_val
            if flag == 2:
                if c2 == 6: board[7, 7] = castle_rook_backup; board[7, 5] = 0
                elif c2 == 2: board[7, 0] = castle_rook_backup; board[7, 3] = 0

            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha: break
        return max_eval, best_move
        
    else: # BLACK
        min_eval = 999999999
        for i in sorted_indices:
            move = moves[i]
            r1, c1, r2, c2, flag = move
            
            captured = board[r2, c2]
            moving_piece = board[r1, c1]
            ep_backup_val = 0
            board[r2, c2] = moving_piece
            board[r1, c1] = 0
            
            # --- LOGIC PHONG CẤP MỚI (BLACK) ---
            promoted = False
            if flag >= 3:
                promoted = True
                new_type = QUEEN
                if flag == 4: new_type = KNIGHT
                elif flag == 5: new_type = ROOK
                elif flag == 6: new_type = BISHOP
                board[r2, c2] = new_type * color # color là -1
            # -----------------------------------
                
            if flag == 1:
                ep_backup_val = board[r1, c2]
                board[r1, c2] = 0
                
            castle_rook_backup = 0
            if flag == 2:
                if c2 == 6: 
                    castle_rook_backup = board[0, 7]; board[0, 5] = castle_rook_backup; board[0, 7] = 0
                elif c2 == 2: 
                    castle_rook_backup = board[0, 0]; board[0, 3] = castle_rook_backup; board[0, 0] = 0

            eval_val, _ = minimax_numba_engine(board, depth - 1, alpha, beta, True, -1, can_castle, node_count, node_limit)
            
            board[r1, c1] = moving_piece
            board[r2, c2] = captured
            if promoted: board[r1, c1] = PAWN * color
            if flag == 1: board[r1, c2] = ep_backup_val
            if flag == 2:
                if c2 == 6: board[0, 7] = castle_rook_backup; board[0, 5] = 0
                elif c2 == 2: board[0, 0] = castle_rook_backup; board[0, 3] = 0
            
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha: break
        return min_eval, best_move

# ==========================================
# PHẦN 3: LỚP VỎ OOP
# ==========================================
class ChessAI:
    def __init__(self):
        self.level = 1
        self.type_map = {'Pawn': PAWN, 'Knight': KNIGHT, 'Bishop': BISHOP, 'Rook': ROOK, 'Queen': QUEEN, 'King': KING}
        
        # Cấu hình giới hạn Node theo Level
        self.node_limits = {
            1: 1000,      # Dễ
            2: 10000,     # Trung bình
            3: 100000,    # Khó
            4: 1000000,   # Rất khó
            5: 10000000   # Chuyên gia
        }

    def set_difficulty(self, level):
        self.level = level

    def get_best_move(self, game, color):
        np_board = np.zeros((8, 8), dtype=np.int8)
        can_castle = np.array([False, False, False, False], dtype=np.bool_)
        
        try:
            wk = game.logic_board[7][4]
            if wk and type(wk).__name__ == 'King' and not getattr(wk, 'has_moved', True):
                if game.logic_board[7][7] and type(game.logic_board[7][7]).__name__ == 'Rook' and not getattr(game.logic_board[7][7], 'has_moved', True): can_castle[0] = True
                if game.logic_board[7][0] and type(game.logic_board[7][0]).__name__ == 'Rook' and not getattr(game.logic_board[7][0], 'has_moved', True): can_castle[1] = True
            
            bk = game.logic_board[0][4]
            if bk and type(bk).__name__ == 'King' and not getattr(bk, 'has_moved', True):
                if game.logic_board[0][7] and type(game.logic_board[0][7]).__name__ == 'Rook' and not getattr(game.logic_board[0][7], 'has_moved', True): can_castle[2] = True
                if game.logic_board[0][0] and type(game.logic_board[0][0]).__name__ == 'Rook' and not getattr(game.logic_board[0][0], 'has_moved', True): can_castle[3] = True
        except: pass

        en_passant_col = -1
        try:
            if hasattr(game, 'en_passant_target') and game.en_passant_target:
                en_passant_col = game.en_passant_target[1]
        except: pass

        for r in range(8):
            for c in range(8):
                piece = game.logic_board[r][c]
                if piece:
                    p_val = self.type_map.get(type(piece).__name__, 0)
                    if piece.color == 'black': p_val *= -1
                    np_board[r, c] = p_val
        
        depth = 2
        if self.level == 2: depth = 3
        elif self.level == 3: depth = 4
        elif self.level >= 4: depth = 5
        
        is_maximizing = (color == 'white')
        node_count = np.array([0], dtype=np.int32)
        node_limit = self.node_limits.get(self.level, 5000000)
        
        try:
            best_val, move_arr = minimax_numba_engine(
                np_board, depth, -1000000000, 1000000000, is_maximizing, 
                int8(en_passant_col), can_castle, node_count, node_limit
            )
            
            r1, c1, r2, c2, flag = move_arr
            if r1 == 0 and c1 == 0 and r2 == 0 and c2 == 0:
                print(f"AI ({color}): No moves found.")
                return None

            # Xác định loại quân phong cấp dựa vào flag
            promotion_type = None
            if flag == 3: promotion_type = 'Queen'
            elif flag == 4: promotion_type = 'Knight'
            elif flag == 5: promotion_type = 'Rook'
            elif flag == 6: promotion_type = 'Bishop'

            piece_obj = game.logic_board[r1][c1]
            
            # In ra log để kiểm tra
            promo_text = f" -> Promote to {promotion_type}" if promotion_type else ""
            print(f"AI ({color}, Lv{self.level}) move: {type(piece_obj).__name__} ({r1},{c1}) -> ({r2},{c2}){promo_text} | Nodes: {node_count[0]}")
            
            # Trả về thêm promotion_type (None nếu không phong cấp)
            return piece_obj, r2, c2, promotion_type
            
        except Exception as e:
            print(f"AI Critical Error: {e}")
            import traceback
            traceback.print_exc()
            return None
