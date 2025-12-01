import sys
import time
import traceback
import numpy as np
import sdl2
import sdl2.ext
import os
import json
from datetime import datetime
from collections import defaultdict
import sdl2.sdlmixer as sdlmixer
import pickle
import threading # Để chạy đa luồng

from ai_engine import ChessAI # Import file vừa tạo
from renderer import Renderer
from compiler import compile_renderer
from loader.mesh_loader import load_scene, load_mesh
from chess_pieces import *

class Chess3D:
    def __init__(self):
        print("🚀 Đang khởi tạo Chess 3D (Full Fix: Win Logic + History Text)...")

        self.base_path = os.path.dirname(os.path.abspath(__file__)) # Lấy đường dẫn thư mục hiện tại
        
        self.width = 800
        self.height = 600
        self.renderer = Renderer(self.width, self.height, "3D Chess")
        
        # Camera Init
        self.mouse_sensitivity = 0.002
        self.mouse_captured = True
        sdl2.SDL_SetRelativeMouseMode(True)
        
        self.yaw = 0.0
        self.pitch = -0.8 
        self.renderer.set_camera_position((0, 20, 5))
        self.renderer.set_camera_target((0, 10, 0))
        self.renderer.rotate_camera(self.yaw, self.pitch)

        # Game State
        self.logic_board = [[None for _ in range(8)] for _ in range(8)]
        self.turn = 'white'
        self.selected_piece = None
        self.valid_moves = []
        self.en_passant_target = None
        self.game_over = False
        self.winner = None
        self.promotion_pending = None
        
        # --- TÍNH NĂNG ---
        self.time_limit = 600.0 # 10 phút
        self.white_time = self.time_limit
        self.black_time = self.time_limit
        self.last_time = time.time()
        self.last_title_update = time.time()
        
        # --- THÊM BIẾN ĐỂ TÍNH FPS ---
        self.frame_count = 0 
        self.fps = 0
        
        # History Vars
        self.undo_stack = []
        self.draw_offered_by = None
        self.half_move_clock = 0
        self.board_history = defaultdict(int)
        
        # --- FILE SAVING SETUP ---
        # Đường dẫn tuyệt đối như bạn yêu cầu
        self.save_folder = os.path.join(self.base_path, "histories")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        self.current_log_file = ""
        self._start_new_log_file()
        
        # Mappings
        self.mesh_to_piece = {}
        self.piece_to_mesh = {}
        self.tile_to_mesh = {}
        self.mesh_id_to_tile = {}
        self.mesh_names = {}
        self.initial_mesh_positions = {}
        
        self.load_and_map_resources()

        # --- LIGHTING ---
        self.renderer.add_light((0, 40, 0), (0, 0, 0))    
        
        self._record_board_state()
        self.animations = [] # Danh sách chứa các quân cờ đang di chuyển

        # --- AI SETUP ---
        self.game_mode = 'pvp' # Mặc định
        
        try:
            # Import class AI
            from ai_engine import ChessAI
            
            # 👇 SỬA DÒNG NÀY: Bỏ tham số 'depth=2' đi
            self.ai = ChessAI() 
            
            self.ai_thinking = False
            self.ai_move_result = None
            print("✅ Đã khởi tạo AI Engine thành công.")
            
        except Exception as e:
            print(f"⚠️ Lỗi khởi tạo AI: {e}")
            self.ai = None
            
        self.ai_thinking = False
        self.ai_move_result = None

        # 1. Khởi động hệ thống âm thanh
        try:
            if sdl2.sdlmixer.Mix_OpenAudio(44100, sdl2.sdlmixer.MIX_DEFAULT_FORMAT, 2, 2048) < 0:
                print("⚠️ Không thể khởi tạo Audio. Game sẽ chạy không có tiếng.")
                self.audio_enabled = False
            else:
                self.audio_enabled = True
        except Exception as e:
            print(f"⚠️ Lỗi Audio: {e}")
            self.audio_enabled = False

        # 2. Tải file âm thanh (ĐỊNH DẠNG .WAV hoặc .OGG là tốt nhất)
        self.sounds = {}
        if self.audio_enabled:
            # 👇👇👇 BẠN ĐỔI ĐƯỜNG DẪN FILE TẠI ĐÂY NHA 👇👇👇
            self.sounds = {}
        if self.audio_enabled:
            # 👇👇👇 TỰ ĐỘNG TẠO ĐƯỜNG DẪN CHO SOUND 👇👇👇
            sound_dir = os.path.join(self.base_path, "res", "sounds")
            
            sound_files = {
                "move": os.path.join(sound_dir, "Move.wav"),
                "capture": os.path.join(sound_dir, "Capture.wav"),
                "notify": os.path.join(sound_dir, "Notify.wav"),
                "victory": os.path.join(sound_dir, "Victory.wav"),
                "check": os.path.join(sound_dir, "Check.wav"),
                "draw": os.path.join(sound_dir, "Draw.wav"),
            }
            # 👆👆👆 ------------------------------------------ 👆👆👆

            for name, path in sound_files.items():
                if os.path.exists(path):
                    self.sounds[name] = sdl2.sdlmixer.Mix_LoadWAV(path.encode('utf-8'))
                else:
                    print(f"⚠️ Không tìm thấy file âm thanh: {path}")

        print("✅ Khởi tạo hoàn tất! Bắt đầu game.")

    def run_ai_logic(self):
        """Hàm này chạy ngầm trong Thread riêng"""
        
        # 👇 THÊM DÒNG NÀY: Đợi 1 giây trước khi bắt đầu tính
        # Giúp game ổn định animation cũ và tạo cảm giác AI đang "nhìn" bàn cờ
        time.sleep(1.0) 
        
        # Gọi hàm tính toán bên file ai_engine.py
        move = self.ai.get_best_move(self, 'black')
        
        # Lưu kết quả để Main Loop thực hiện
        self.ai_move_result = move 
        self.ai_thinking = False
    
    def play_sound(self, name):
        if self.audio_enabled and name in self.sounds and self.sounds[name]:
            # Phát âm thanh trên channel còn trống (-1)
            sdl2.sdlmixer.Mix_PlayChannel(-1, self.sounds[name], 0)

    def _start_new_log_file(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_log_file = os.path.join(self.save_folder, f"game_{timestamp}.json")
        initial_data = {
            "start_time": timestamp,
            "moves": [],
            "result": "ongoing"
        }
        try:
            with open(self.current_log_file, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=4)
        except Exception as e:
            print(f"⚠️ Không thể tạo file log: {e}")

    def _save_history_to_disk(self):
        # --- 1. LƯU DẠNG JSON (Giữ nguyên để load game nếu cần) ---
        json_moves = []
        cols = ['a','b','c','d','e','f','g','h']
        
        # Danh sách tạm để xử lý ghép cặp nước đi cho file Text
        temp_moves_text = [] 
        
        for i, move in enumerate(self.undo_stack):
            piece = move['piece']
            start = move['start_pos']
            end = move['end_pos']
            is_capture = True if move['captured'] else False
            
            # Record cho JSON
            record = {
                "move_number": i + 1,
                "color": move['piece'].color,
                "piece": type(move['piece']).__name__,
                "start": start,
                "end": end,
                "is_capture": is_capture,
                "is_castling": move['is_castling'],
                "is_promotion": move['promotion']
            }
            json_moves.append(record)
            
            # --- XỬ LÝ FORMAT TEXT CHUẨN CỜ VUA ---
            # Chuyển tọa độ: row 7 -> 1, row 0 -> 8
            s_sq = f"{cols[start[1]]}{8 - start[0]}" 
            e_sq = f"{cols[end[1]]}{8 - end[0]}"
            
            # Ký hiệu quân cờ (Viết tắt tiếng Anh: P, N, B, R, Q, K)
            piece_map = {'Pawn':'', 'Knight':'N', 'Bishop':'B', 'Rook':'R', 'Queen':'Q', 'King':'K'}
            p_char = piece_map.get(type(piece).__name__, '')
            
            # Ký hiệu ăn quân
            cap_char = "x" if is_capture else ""
            
            # Tạo chuỗi nước đi: VD "Nxe5" hoặc "e4"
            # Nếu là tốt (Pawn) ăn quân thì phải thêm cột xuất phát: VD "exd5"
            if type(piece).__name__ == 'Pawn' and is_capture:
                 move_str = f"{cols[start[1]]}x{e_sq}"
            else:
                 move_str = f"{p_char}{cap_char}{e_sq}"
                 
            # Xử lý Nhập thành
            if move['is_castling']:
                move_str = "O-O" if end[1] > start[1] else "O-O-O"
                
            temp_moves_text.append(move_str)
            
        data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "finished" if self.game_over else "ongoing",
            "winner": self.winner if self.winner else None,
            "moves": json_moves
        }
        
        # Ghi file JSON
        try:
            with open(self.current_log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception: pass

        # --- 2. LƯU DẠNG TEXT (SCORE SHEET - DỄ ĐỌC) ---
        txt_file = self.current_log_file.replace(".json", ".txt")
        try:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"CHESS 3D - GAME HISTORY\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("==========================================\n")
                f.write(f"{'No.':<5} {'White':<15} {'Black':<15}\n")
                f.write("------------------------------------------\n")
                
                # Ghép cặp nước đi: Trắng - Đen
                # Ví dụ: 1. e4 e5
                num_moves = len(temp_moves_text)
                for i in range(0, num_moves, 2):
                    move_num = (i // 2) + 1
                    white_move = temp_moves_text[i]
                    black_move = temp_moves_text[i+1] if (i+1) < num_moves else ""
                    
                    f.write(f"{move_num:<5}. {white_move:<15} {black_move:<15}\n")
                
                if self.game_over:
                    f.write("==========================================\n")
                    f.write(f"RESULT: {self.winner.upper()} WON\n")
        except Exception: pass

    def load_and_map_resources(self):
        print("📂 Đang tải scene...")
        # ==============================================================
        path_to_model = os.path.join(self.base_path, "res", "chess_pieces.glb")
        path_to_cache = os.path.join(self.base_path, "res", "chess_pieces.pkl")
        # ==============================================================
        meshes, texs, uvs, nodes, bounds, pos_list, rot_list = [], [], [], [], [], [], []
        loaded_from_cache = False

        # 1. Thử load từ Cache (nhanh siêu tốc)
        if os.path.exists(path_to_cache):
            try:
                print("⚡ Tìm thấy cache, đang load nhanh...")
                with open(path_to_cache, 'rb') as f:
                    data = pickle.load(f)
                    meshes, texs, uvs, nodes, bounds, pos_list, rot_list = data
                loaded_from_cache = True
                print("✅ Load từ cache thành công!")
            except Exception as e:
                print(f"⚠️ Cache lỗi, sẽ load lại file gốc: {e}")

        # 2. Nếu không có cache hoặc lỗi, load từ file GLB (chậm)
        if not loaded_from_cache:
            try:
                # Load scene từ file gốc
                meshes, texs, uvs, nodes, bounds, pos_list, rot_list = load_scene(path_to_model, True)
                
                # Lưu vào cache cho lần sau
                with open(path_to_cache, 'wb') as f:
                    pickle.dump([meshes, texs, uvs, nodes, bounds, pos_list, rot_list], f)
                print("💾 Đã tạo file cache cho lần mở sau.")
            except Exception as e:
                print(f"\n❌ LỖI: Không thể tải file mô hình: {e}")
                sys.exit(1)

        # 3. Thêm các mesh vào renderer và map với logic
        for i in range(len(meshes)):
            idx = self.renderer.add_mesh(
                position=pos_list[i],
                pivot=(0, 0, 0),
                rotation=rot_list[i],
                loaded_meshes=meshes[i],
                loaded_texs=texs[i],
                loaded_uvs=uvs[i],
                aabb_data=bounds[i]
            )
            self.mesh_names[idx] = nodes[i]
            self.initial_mesh_positions[idx] = pos_list[i]
            self._parse_node(nodes[i], idx)

    def _parse_node(self, name, mesh_idx):
        parts = name.split('_')
        if parts[0] == 'board':
            try:
                col = int(parts[1])
                row = int(parts[2])
                self.mesh_id_to_tile[mesh_idx] = (col, row)
                self.tile_to_mesh[(col, row)] = mesh_idx
            except: pass
            return
        if len(parts) >= 3:
            color_code = parts[0]
            kind = parts[1]
            try: idx_in_name = int(parts[2])
            except: return
            color = 'white' if color_code == 'w' else 'black'
            row, col = self._get_initial_pos(color, kind, idx_in_name)
            if row != -1 and col != -1:
                piece = self._create_piece_logic(kind, color, row, col)
                self.logic_board[row][col] = piece
                self.mesh_to_piece[mesh_idx] = piece
                self.piece_to_mesh[piece] = mesh_idx

    def _get_initial_pos(self, color, kind, idx):
        is_white = (color == 'white')
        if kind == 'pawn':
            row = 6 if is_white else 1
            col = idx - 1
        else:
            row = 7 if is_white else 0
            back_rank = {('rook', 1): 0, ('knight', 1): 1, ('bishop', 1): 2, ('queen', 1): 3, ('king', 1): 4, ('bishop', 2): 5, ('knight', 2): 6, ('rook', 2): 7}
            col = back_rank.get((kind, idx), -1)
        if 0 <= col < 8: return (row, col)
        return (-1, -1)

    def _create_piece_logic(self, kind, color, row, col):
        mapping = {'pawn': Pawn, 'rook': Rook, 'knight': Knight, 'bishop': Bishop, 'queen': Queen, 'king': King}
        return mapping.get(kind, Piece)(color, row, col)

    # ==========================================
    # LOGIC HỖ TRỢ
    # ==========================================
    def _get_castling_rights_str(self):
        res = ""
        w_king = self.logic_board[7][4]
        if isinstance(w_king, King) and w_king.color == 'white' and not getattr(w_king, 'has_moved', False):
            if isinstance(self.logic_board[7][7], Rook) and self.logic_board[7][7].color == 'white' and not getattr(self.logic_board[7][7], 'has_moved', False): res += "K"
            if isinstance(self.logic_board[7][0], Rook) and self.logic_board[7][0].color == 'white' and not getattr(self.logic_board[7][0], 'has_moved', False): res += "Q"
        b_king = self.logic_board[0][4]
        if isinstance(b_king, King) and b_king.color == 'black' and not getattr(b_king, 'has_moved', False):
            if isinstance(self.logic_board[0][7], Rook) and self.logic_board[0][7].color == 'black' and not getattr(self.logic_board[0][7], 'has_moved', False): res += "k"
            if isinstance(self.logic_board[0][0], Rook) and self.logic_board[0][0].color == 'black' and not getattr(self.logic_board[0][0], 'has_moved', False): res += "q"
        return res if res else "-"

    def _generate_fen_key(self):
        board_str = ""
        for r in range(8):
            empty = 0
            for c in range(8):
                p = self.logic_board[r][c]
                if p:
                    if empty > 0: board_str += str(empty)
                    empty = 0
                    char = p.symbol if p.color == 'white' else p.symbol.lower()
                    board_str += char
                else:
                    empty += 1
            if empty > 0: board_str += str(empty)
            board_str += "/"
        ep_str = "-"
        if self.en_passant_target: ep_str = f"{self.en_passant_target[0]},{self.en_passant_target[1]}"
        castling_str = self._get_castling_rights_str()
        return f"{board_str} {self.turn} {castling_str} {ep_str}"

    def _record_board_state(self):
        key = self._generate_fen_key()
        self.board_history[key] += 1

    def _check_insufficient_material(self):
        pieces = [p for row in self.logic_board for p in row if p]
        
        # 1. Nếu còn Tốt (Pawn), Xe (Rook), Hậu (Queen) -> Chắc chắn KHÔNG hòa
        for p in pieces:
            if isinstance(p, (Pawn, Rook, Queen)):
                return False
        
        # --- ĐẾN ĐÂY THÌ TRÊN BÀN CHỈ CÒN VUA, MÃ, TƯỢNG ---

        # 2. Ít hơn 4 quân (tức là 2 hoặc 3 quân) -> HÒA CHẮC CHẮN
        # (K vs K) hoặc (K vs K+N) hoặc (K vs K+B)
        if len(pieces) < 4:
            return True

        # 3. Trường hợp 4 quân (K+N vs K+N, K+B vs K+B...)
        # Logic đơn giản cho game 3D: Nếu không còn quân nặng (Xe/Hậu/Tốt), coi như hòa cho đỡ mất thời gian.
        # (Luật FIDE đoạn này rất phức tạp về màu ô của Tượng, nhưng với game giải trí thì check thế này là đủ tốt)
        if len(pieces) == 4:
            return True # K+N vs K+N hoặc K+B vs K+B -> Cho hòa luôn
            
        return False

    # ==========================================
    # RESET & UNDO
    # ==========================================
    def reset_game(self):
        print("\n🔄 ĐANG KHỞI ĐỘNG LẠI GAME...")
        self.turn = 'white'
        self.game_over = False
        self.winner = None
        self.selected_piece = None
        self.valid_moves = []
        self.en_passant_target = None
        self.promotion_pending = None
        self.half_move_clock = 0
        self.board_history.clear()
        self.white_time = self.time_limit
        self.black_time = self.time_limit
        self.undo_stack.clear()
        self.draw_offered_by = None
        self.last_time = time.time()
        
        self._start_new_log_file()
        
        self.logic_board = [[None for _ in range(8)] for _ in range(8)]
        self.mesh_to_piece.clear()
        self.piece_to_mesh.clear()
        
        # --- FIX RESET GEOMETRY ---
        w_pawn_donor = None
        b_pawn_donor = None
        for idx, name in self.mesh_names.items():
            if 'w_pawn' in name and w_pawn_donor is None: w_pawn_donor = idx
            if 'b_pawn' in name and b_pawn_donor is None: b_pawn_donor = idx
        
        for mesh_idx, start_pos in self.initial_mesh_positions.items():
            self.renderer.set_mesh_transform(mesh_idx, position=start_pos)
            self.renderer.set_mesh_visible_flag(mesh_idx, True)
            
            name = self.mesh_names.get(mesh_idx, "")
            if 'pawn' in name:
                donor = w_pawn_donor if name.startswith('w') else b_pawn_donor
                if donor is not None:
                    self._apply_geometry_from_id(mesh_idx, donor)
            if not name.startswith('board'): 
                self._parse_node(name, mesh_idx)

        self._record_board_state()
        self.renderer.update_light()

    def undo_last_move(self):
        if not self.undo_stack or self.game_over:
            print("⚠️ Không thể Undo!")
            return

        print("↩️ Đang Undo nước đi...")
        state = self.undo_stack.pop()
        
        piece = state['piece']
        start_r, start_c = state['start_pos']
        end_r, end_c = state['end_pos']
        
        # 1. Di chuyển hình 3D về trước
        self._move_piece_3d(piece, start_r, start_c)
        
        # 2. Cập nhật logic
        self.logic_board[end_r][end_c] = None
        self.logic_board[start_r][start_c] = piece
        piece.row, piece.col = start_r, start_c
        piece.has_moved = state['has_moved']
        
        # 3. Khôi phục quân bị ăn
        captured = state['captured']
        if captured:
            cap_r, cap_c = state['captured_pos']
            self.logic_board[cap_r][cap_c] = captured
            if 'captured_mesh_id' in state:
                mid = state['captured_mesh_id']
                self.renderer.set_mesh_visible_flag(mid, True)
                self.mesh_to_piece[mid] = captured
                self.piece_to_mesh[captured] = mid
                
        # 4. Khôi phục các biến trạng thái
        self.en_passant_target = state['en_passant_target']
        self.half_move_clock = state['half_move_clock']
        self.white_time = state['white_time']
        self.black_time = state['black_time']
        
        if 'turn_before_move' in state:
            self.turn = state['turn_before_move']
        else:
            self.turn = 'white' if self.turn == 'black' else 'black'
        
        if state['is_castling']:
            rook = state['castling_rook']
            r_start = state['rook_start']
            r_end = state['rook_end']
            self.logic_board[r_end[0]][r_end[1]] = None
            self.logic_board[r_start[0]][r_start[1]] = rook
            self._move_piece_3d(rook, r_start[0], r_start[1])
            rook.row, rook.col = r_start
            rook.has_moved = False

        if state['promotion']:
            piece.__class__ = Pawn
            self._swapping_geometry(piece, Pawn)
        
        self.valid_moves = []
        self.selected_piece = None
        self.draw_offered_by = None
        
        self._save_history_to_disk()
        self.renderer.update_light()
        print(f"✅ Đã quay lại lượt: {self.turn.upper()}")

    # ==========================================
    # LOGIC CỜ
    # ==========================================
    def get_king_pos(self, color):
        for r in range(8):
            for c in range(8):
                p = self.logic_board[r][c]
                if p and isinstance(p, King) and p.color == color: return (r, c)
        return None

    def is_square_attacked(self, r, c, ally_color):
        enemy_color = 'black' if ally_color == 'white' else 'white'
        pawn_dir = -1 if enemy_color == 'white' else 1
        for dc in [-1, 1]:
            pr, pc = r - pawn_dir, c + dc
            if 0 <= pr < 8 and 0 <= pc < 8:
                p = self.logic_board[pr][pc]
                if isinstance(p, Pawn) and p.color == enemy_color: return True
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            if 0 <= r+dr < 8 and 0 <= c+dc < 8:
                p = self.logic_board[r+dr][c+dc]
                if isinstance(p, Knight) and p.color == enemy_color: return True
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                if 0 <= r+dr < 8 and 0 <= c+dc < 8:
                    p = self.logic_board[r+dr][c+dc]
                    if isinstance(p, King) and p.color == enemy_color: return True
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in dirs:
            cur_r, cur_c = r + dr, c + dc
            while 0 <= cur_r < 8 and 0 <= cur_c < 8:
                p = self.logic_board[cur_r][cur_c]
                if p:
                    if p.color == enemy_color:
                        is_straight = (dr == 0 or dc == 0)
                        if isinstance(p, Queen): return True
                        if is_straight and isinstance(p, Rook): return True
                        if not is_straight and isinstance(p, Bishop): return True
                    break
                cur_r += dr
                cur_c += dc
        return False

    def is_in_check(self, color):
        k_pos = self.get_king_pos(color)
        return False if k_pos is None else self.is_square_attacked(k_pos[0], k_pos[1], color)

    def get_valid_moves(self, piece):
        if isinstance(piece, Pawn): pseudo = piece.get_pseudo_moves(self.logic_board, self.en_passant_target)
        elif isinstance(piece, King): pseudo = piece.get_pseudo_moves(self.logic_board, self.is_in_check)
        else: pseudo = piece.get_pseudo_moves(self.logic_board)
        
        valid = []
        for r, c in pseudo:
            if isinstance(piece, King) and abs(c - piece.col) > 1:
                if self.is_in_check(piece.color): continue
                mid_col = (piece.col + c) // 2
                if self.is_square_attacked(piece.row, mid_col, piece.color): continue
            if self.is_move_legal(piece, r, c): valid.append((r, c))
        return valid

    def is_move_legal(self, piece, r_dest, c_dest):
        r_start, c_start = piece.row, piece.col
        captured = self.logic_board[r_dest][c_dest]
        self.logic_board[r_dest][c_dest] = piece
        self.logic_board[r_start][c_start] = None
        piece.row, piece.col = r_dest, c_dest
        in_check = self.is_in_check(piece.color)
        self.logic_board[r_start][c_start] = piece
        self.logic_board[r_dest][c_dest] = captured
        piece.row, piece.col = r_start, c_start
        return not in_check

    def check_game_over(self):
        if self.white_time <= 0: return "timeout_white"
        if self.black_time <= 0: return "timeout_black"
        can_move = False
        pieces = [p for row in self.logic_board for p in row if p and p.color == self.turn]
        for p in pieces:
            if self.get_valid_moves(p): 
                can_move = True; break
        if not can_move: return "checkmate" if self.is_in_check(self.turn) else "stalemate"
        if self.half_move_clock >= 100: return "50-move"
        current_key = self._generate_fen_key()
        if self.board_history[current_key] >= 3: return "3-fold"
        if self._check_insufficient_material(): return "insufficient"
        return None

    # ==========================================
    # INPUT & UPDATE
    # ==========================================
    def update_clock(self):
        if not self.game_over and not self.promotion_pending:
            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            if self.turn == 'white': self.white_time -= dt
            else: self.black_time -= dt
        else:
            self.last_time = time.time() 

    def handle_input(self):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT: return False
            if self.game_over:
                if event.type == sdl2.SDL_KEYDOWN:
                    if event.key.keysym.sym == sdl2.SDLK_SPACE: self.reset_game()
                    elif event.key.keysym.sym == sdl2.SDLK_ESCAPE: return False
                if event.type == sdl2.SDL_MOUSEMOTION and self.mouse_captured:
                    self.yaw += event.motion.xrel * self.mouse_sensitivity
                    self.pitch += event.motion.yrel * self.mouse_sensitivity
                    self.pitch = max(-1.5, min(1.5, self.pitch))
                    self.renderer.rotate_camera(self.yaw, self.pitch)
                if event.type == sdl2.SDL_MOUSEBUTTONDOWN and event.button.button == sdl2.SDL_BUTTON_RIGHT:
                    self.mouse_captured = True; sdl2.SDL_SetRelativeMouseMode(True)
                continue 

            if event.type == sdl2.SDL_MOUSEMOTION:
                if self.mouse_captured:
                    self.yaw += event.motion.xrel * self.mouse_sensitivity
                    self.pitch += event.motion.yrel * self.mouse_sensitivity
                    self.pitch = max(-1.5, min(1.5, self.pitch))
                    self.renderer.rotate_camera(self.yaw, self.pitch)
            elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                if event.button.button == sdl2.SDL_BUTTON_RIGHT:
                    self.mouse_captured = True
                    sdl2.SDL_SetRelativeMouseMode(True)
                elif event.button.button == sdl2.SDL_BUTTON_LEFT:
                    if not self.mouse_captured: self._process_click(event.button.x, event.button.y)
            elif event.type == sdl2.SDL_KEYDOWN:
                key = event.key.keysym.sym
                if key == sdl2.SDLK_ESCAPE:
                    self.mouse_captured = False; sdl2.SDL_SetRelativeMouseMode(False)
                elif self.promotion_pending:
                    if key == sdl2.SDLK_q: self.complete_promotion(Queen)
                    elif key == sdl2.SDLK_r: self.complete_promotion(Rook)
                    elif key == sdl2.SDLK_b: self.complete_promotion(Bishop)
                    elif key == sdl2.SDLK_n: self.complete_promotion(Knight)
                else:
                    if key == sdl2.SDLK_z: self.undo_last_move() 
                    elif key == sdl2.SDLK_f: self._resign() 
                    elif key == sdl2.SDLK_h: self._offer_draw() 
                    elif key == sdl2.SDLK_y: self._accept_draw()
        
        if self.mouse_captured:
            keystate = sdl2.SDL_GetKeyboardState(None)
            move_speed = 2.0
            if keystate[sdl2.SDL_SCANCODE_W]: self.renderer.move_camera(forward=move_speed)
            if keystate[sdl2.SDL_SCANCODE_S]: self.renderer.move_camera(forward=-move_speed)
            if keystate[sdl2.SDL_SCANCODE_A]: self.renderer.move_camera(right=-move_speed)
            if keystate[sdl2.SDL_SCANCODE_D]: self.renderer.move_camera(right=move_speed)
            if keystate[sdl2.SDL_SCANCODE_SPACE] and not self.game_over: self.renderer.move_camera(up=move_speed)
            if keystate[sdl2.SDL_SCANCODE_LCTRL]: self.renderer.move_camera(up=-move_speed)
        return True

    def _resign(self):
        print(f"🏳️ {self.turn.upper()} ĐẦU HÀNG!")
        self.winner = 'black' if self.turn == 'white' else 'white'
        self.game_over = True
        self._show_game_over_msg(f"{self.winner.upper()} THẮNG (ĐỐI THỦ ĐẦU HÀNG)")
        self._save_history_to_disk()

    def _offer_draw(self):
        if self.draw_offered_by is None:
            self.draw_offered_by = self.turn
            print(f"🤝 {self.turn.upper()} ĐỀ NGHỊ HÒA! (Đối thủ nhấn Y để đồng ý, hoặc đi quân để từ chối)")
        else: 
            if self.draw_offered_by == self.turn:
                print("⚠️ Bạn đã gửi lời mời hòa rồi!")
            else:
                print("⚠️ Đối thủ đang mời hòa! Nhấn Y để đồng ý.")

    def _accept_draw(self):
        if self.draw_offered_by is not None:
            if self.draw_offered_by != self.turn:
                self.winner = 'draw'
                self.game_over = True
                self.play_sound("draw")
                self._show_game_over_msg("HÒA CỜ (THỎA THUẬN)")
                self._save_history_to_disk()
            else:
                print("⚠️ Không thể tự chấp nhận lời mời của chính mình!")
        else: 
            print("⚠️ Không có lời mời hòa nào!")

    def _process_click(self, mx, my):
        if self.game_over or not (0 <= my < self.height and 0 <= mx < self.width): return
        hit_id = int(self.renderer.object_buffer[my, mx])
        if hit_id == 255 or hit_id < 0: return
        mesh_idx = hit_id
        if mesh_idx in self.mesh_to_piece:
            piece = self.mesh_to_piece[mesh_idx]
            if piece.color == self.turn:
                self.selected_piece = piece
                self.valid_moves = self.get_valid_moves(piece)
                print(f"🔹 Chọn: {piece.symbol} tại ({piece.row}, {piece.col})")
            elif self.selected_piece and (piece.row, piece.col) in self.valid_moves:
                self._execute_move(self.selected_piece, piece.row, piece.col)
        elif mesh_idx in self.mesh_id_to_tile:
            if self.selected_piece:
                col, row = self.mesh_id_to_tile[mesh_idx]
                if (row, col) in self.valid_moves:
                    self._execute_move(self.selected_piece, row, col)

    def _execute_move(self, piece, end_row, end_col):
        if self.draw_offered_by is not None and self.draw_offered_by == piece.color:
            pass
        elif self.draw_offered_by is not None and self.draw_offered_by != piece.color:
            print(f"❌ {piece.color.title()} đã đi quân -> Từ chối hòa.")
            self.draw_offered_by = None
            
        start_row, start_col = piece.row, piece.col
        target = self.logic_board[end_row][end_col]
        captured_pos = (end_row, end_col) if target else None
        captured_mesh_id = self.piece_to_mesh.get(target) if target else None
        
        move_info = {
            'piece': piece,
            'start_pos': (start_row, start_col),
            'end_pos': (end_row, end_col),
            'has_moved': piece.has_moved,
            'captured': target,
            'captured_pos': captured_pos,
            'captured_mesh_id': captured_mesh_id,
            'en_passant_target': self.en_passant_target,
            'half_move_clock': self.half_move_clock,
            'white_time': self.white_time,
            'black_time': self.black_time,
            'turn_before_move': self.turn, 
            'is_castling': False,
            'promotion': False
        }

        is_capture = (target is not None)
        if target:
            mid = self.piece_to_mesh[target]
            self.renderer.set_mesh_visible_flag(mid, False)

        if isinstance(piece, Pawn) and (end_row, end_col) == self.en_passant_target:
            cap_row = start_row
            cap_p = self.logic_board[cap_row][end_col]
            if cap_p:
                move_info['captured'] = cap_p
                move_info['captured_pos'] = (cap_row, end_col)
                move_info['captured_mesh_id'] = self.piece_to_mesh.get(cap_p)
                mid = self.piece_to_mesh[cap_p]
                self.renderer.set_mesh_visible_flag(mid, False)
                self.logic_board[cap_row][end_col] = None
                is_capture = True

        if isinstance(piece, Pawn) or is_capture: self.half_move_clock = 0
        else: self.half_move_clock += 1
            
        if isinstance(piece, King) and abs(end_col - start_col) > 1:
            move_info['is_castling'] = True
            rook_col = 7 if end_col == 6 else 0
            new_rook_col = 5 if end_col == 6 else 3
            rook = self.logic_board[start_row][rook_col]
            move_info['castling_rook'] = rook
            move_info['rook_start'] = (start_row, rook_col)
            move_info['rook_end'] = (start_row, new_rook_col)
            self._move_piece_3d(rook, start_row, new_rook_col)
            self.logic_board[start_row][new_rook_col] = rook
            self.logic_board[start_row][rook_col] = None
            rook.move(start_row, new_rook_col)
            rook.has_moved = True

        if isinstance(piece, Pawn) and abs(end_row - start_row) == 2:
            self.en_passant_target = ((start_row + end_row)//2, start_col)
        else:
            self.en_passant_target = None
            
        self.logic_board[end_row][end_col] = piece
        self.logic_board[start_row][start_col] = None
        self._move_piece_3d(piece, end_row, end_col)
        piece.move(end_row, end_col)
        piece.has_moved = True
        
        self.undo_stack.append(move_info)
        self._save_history_to_disk()
        
        if is_capture:
            self.play_sound("capture") # Có ăn quân -> tiếng Capture
        else:
            self.play_sound("move")    # Đi thường -> tiếng Move
        self._print_move_to_console(piece, (start_row, start_col), (end_row, end_col), is_capture)

        if isinstance(piece, Pawn) and (end_row == 0 or end_row == 7):
            self.promotion_pending = (piece, end_row, end_col)
            move_info['promotion'] = True
            print("👑 Phong cấp! Nhấn Q/R/B/N")
            self.play_sound("notify") # Phong cấp -> Tiếng thông báo
        else:
            self.selected_piece = None
            self.valid_moves = []
            self.switch_turn()

    def _print_move_to_console(self, piece, start, end, is_capture):
        cols = ['a','b','c','d','e','f','g','h']
        def to_chess_coord(r, c):
            rank = 8 - r 
            file = cols[c]
            return f"{file}{rank}"
        
        s_txt = to_chess_coord(start[0], start[1])
        e_txt = to_chess_coord(end[0], end[1])
        
        move_count = (len(self.undo_stack) + 1) // 2
        prefix = "🔹" if piece.color == 'white' else "🔸"
        cap_str = "⚔️ ĂN" if is_capture else "➝"
        
        print(f"{prefix} Move {move_count}: {piece.color.title()} {piece.symbol} ({s_txt} {cap_str} {e_txt})")

    def _move_piece_3d(self, piece, r, c):
        mid = self.piece_to_mesh[piece]

        # Vị trí hiện tại (Điểm xuất phát)
        start_pos = np.copy(self.renderer.pos_list[mid])

        # Tính vị trí đích (Điểm đến)
        # Lưu ý: Logic tính toán này dựa trên code cũ của bạn
        d_c, d_r = float(c - piece.col), float(r - piece.row)

        # Code cũ của bạn cộng dồn vào pos hiện tại, nhưng vì start_pos lấy từ renderer
        # nên ta cần tính đích đến tuyệt đối dựa trên ô cờ (row, col) mới để chính xác hơn.
        # TUY NHIÊN, để an toàn với logic cũ, ta tính delta dựa trên sự chênh lệch:
        target_pos = np.array([start_pos[0] + d_c, start_pos[1], start_pos[2] + d_r], dtype=np.float32)

        # Tạo Animation Info
        anim_data = {
            "mesh_id": mid,
            "start": start_pos,
            "end": target_pos,
            "start_time": time.time(),
            "duration": 0.3  # <--- THỜI GIAN DI CHUYỂN (giây). Sửa số này để nhanh/chậm
        }

        # Xóa các animation cũ của cùng 1 mesh (nếu user click quá nhanh)
        self.animations = [a for a in self.animations if a["mesh_id"] != mid]
        self.animations.append(anim_data)

    def _apply_geometry_from_id(self, target_id, donor_id):
        if target_id is not None and donor_id is not None:
            mesh = self.renderer.mesh_list[donor_id]
            tex = self.renderer.tex_list[donor_id]
            uv = self.renderer.uv_list[donor_id]
            self.renderer.set_mesh_geometry(target_id, mesh=mesh, uvs=uv, tex=tex)

    def _swapping_geometry(self, piece_obj, target_class):
        curr_id = self.piece_to_mesh.get(piece_obj)
        type_map = {Queen:'queen', Rook:'rook', Bishop:'bishop', Knight:'knight', Pawn:'pawn'}
        target_str = type_map.get(target_class, 'pawn')
        
        donor_id = None
        for p, mid in self.piece_to_mesh.items():
            if type(p).__name__.lower() == target_str and p.color == piece_obj.color:
                donor_id = mid; break
        if donor_id is None:
             for p, mid in self.piece_to_mesh.items():
                if type(p).__name__.lower() == target_str:
                    donor_id = mid; break
        
        if curr_id is not None and donor_id is not None:
            mesh = self.renderer.mesh_list[donor_id]
            tex = self.renderer.tex_list[donor_id]
            uv = self.renderer.uv_list[donor_id]
            self.renderer.set_mesh_geometry(curr_id, mesh=mesh, uvs=uv, tex=tex)
            
            target_y = self.renderer.pos_list[donor_id][1]
            curr_pos = self.renderer.pos_list[curr_id]
            new_pos = np.array([curr_pos[0], target_y, curr_pos[2]], dtype=np.float32)
            self.renderer.set_mesh_transform(curr_id, position=new_pos)

    def complete_promotion(self, cls):
        if not self.promotion_pending: return
        old_p, r, c = self.promotion_pending
        old_p.__class__ = cls
        self._swapping_geometry(old_p, cls)
        self.promotion_pending = None
        self.selected_piece = None
        self.valid_moves = []
        self._save_history_to_disk()
        self.switch_turn()

    def switch_turn(self):
        # Đổi lượt
        self.turn = 'black' if self.turn == 'white' else 'white'
        self._record_board_state()
        print(f"\n🔄 Lượt của: {self.turn.upper()} | Time: W {int(self.white_time)}s - B {int(self.black_time)}s")
        
        result = self.check_game_over()
        if result:
            self.game_over = True
            self.play_sound("victory") # Tiếng chiến thắng
            
            # --- XÁC ĐỊNH NGƯỜI THẮNG ---
            msg = ""
            if result == "checkmate":
                # Nếu đến lượt hiện tại mà bị chiếu hết -> Người đó thua -> Đối thủ thắng
                self.winner = 'black' if self.turn == 'white' else 'white'
                msg = f"{self.winner.upper()} THẮNG (CHIẾU HẾT)"
            elif result == "timeout_white":
                self.winner = 'black'
                msg = "ĐEN THẮNG (TRẮNG HẾT GIỜ)"
            elif result == "timeout_black":
                self.winner = 'white'
                msg = "TRẮNG THẮNG (ĐEN HẾT GIỜ)"
            else:
                self.winner = 'draw'
                self.play_sound("draw")
                msg_map = {
                    "stalemate": "HÒA (HẾT NƯỚC ĐI)",
                    "50-move": "HÒA (50 NƯỚC)",
                    "3-fold": "HÒA (LẶP 3 LẦN)",
                    "insufficient": "HÒA (KHÔNG ĐỦ QUÂN)"
                }
                msg = msg_map.get(result, "HÒA CỜ")

            self._show_game_over_msg(msg)
            self._save_history_to_disk()
        else:
            # 👇👇👇 THÊM ĐOẠN NÀY VÀO 👇👇👇
            # TRƯỜNG HỢP GAME VẪN TIẾP TỤC
            # Kiểm tra xem phe hiện tại (vừa nhận lượt) có đang bị chiếu không?
            if self.is_in_check(self.turn):
                print(f"⚠️ {self.turn.upper()} ĐANG BỊ CHIẾU!")
                self.play_sound("check") # <--- Lệnh phát âm thanh Check
            # 👆👆👆 ----------------------- 👆👆👆

        # 👇👇👇 THÊM ĐOẠN KÍCH HOẠT AI Ở CUỐI HÀM 👇👇👇
        if not self.game_over and self.game_mode == 'pvai' and self.turn == 'black':
            print("🤖 AI đang suy nghĩ...")
            self.ai_thinking = True
            # Gọi thread AI chạy ngầm
            threading.Thread(target=self.run_ai_logic, daemon=True).start()

    def _show_game_over_msg(self, msg):
        print("\n" + "═"*50)
        print(f"{msg.center(50)}")
        print("NHẤN [SPACE] ĐỂ CHƠI LẠI".center(50))
        print("═"*50 + "\n")
        title_str = f"GAME OVER! {msg} - Press SPACE"
        sdl2.SDL_SetWindowTitle(self.renderer.window.window, title_str.encode('utf-8'))
    
    def update_animations(self):
        current_time = time.time()
        completed = []

        for anim in self.animations:
            # Tính toán tiến độ (t đi từ 0.0 đến 1.0)
            elapsed = current_time - anim["start_time"]
            t = elapsed / anim["duration"]

            if t >= 1.0:
                t = 1.0
                completed.append(anim)

            # Công thức nội suy tuyến tính (Lerp): Pos = Start + (End - Start) * t
            # Giúp quân cờ trượt từ A sang B
            new_pos = anim["start"] + (anim["end"] - anim["start"]) * t

            self.renderer.set_mesh_transform(anim["mesh_id"], position=new_pos)

        # Xóa các animation đã hoàn thành khỏi danh sách
        for c in completed:
            self.animations.remove(c)

    def run(self):
        print("\n🎮 CONTROLS:")
        print("   [Chuột Trái]: Chọn/Đi quân")
        print("   [Chuột Phải]: Xoay Camera")
        print("   [Z]: Undo (Đi lại)")
        print("   [F]: Resign (Đầu hàng)")
        print("   [H]: Offer Draw (Cầu hòa) / [Y]: Đồng ý hòa")

        # 👇👇👇 THÊM ĐOẠN NÀY: CHỌN CHẾ ĐỘ CHƠI 👇👇👇
        print("\n========================================")
        print("   CHOOSE GAME MODE (Chọn chế độ):")
        print("   [1] Player vs Player (Hai người chơi)")
        print("   [2] Player vs AI (Đấu với Máy)")
        print("========================================")
 
        while True:
            choice = input("👉 Nhập (1 hoặc 2): ").strip()
            if choice == '1':
                self.game_mode = 'pvp'
                print("✅ Mode: PvP")
                break
            elif choice == '2':
                if getattr(self, 'ai', None):
                    self.game_mode = 'pvai'
                    print("\n--- CHỌN ĐỘ KHÓ AI ---")
                    print("   [1] Easy (Dễ - Đi lung tung)")
                    print("   [2] Medium (Vừa - Biết ăn quân)")
                    print("   [3] Hard (Khó - Tính trước 1 nước)")
                    print("   [4] Expert (Siêu khó - Tính trước 2 nước)")
                    
                    while True:
                        lvl = input("👉 Chọn độ khó (1-4): ").strip()
                        if lvl in ['1', '2', '3', '4']:
                            self.ai.set_difficulty(int(lvl))
                            print(f"✅ Đã chọn Level {lvl}. Bạn cầm quân Trắng.")
                            break
                        print("⚠️ Vui lòng nhập từ 1 đến 4.")
                else:
                    print("❌ Lỗi: Không có AI Engine. Về PvP.")
                    self.game_mode = 'pvp'
                break
            else:
                print("⚠️ Nhập sai.")

        running = True
        self.renderer.show()
        self.renderer.update_light()
        try:
            self.last_time = time.time()
            self.frame_count = 0
            self.fps = 0
            while running:
                self.frame_count += 1
                self.update_clock()
                self.update_animations()
                running = self.handle_input()

                # 👇 XỬ LÝ AI ĐI QUÂN 👇
                if self.ai_move_result:
                    p, r, c = self.ai_move_result
                    print(f"🤖 AI (Lv{self.ai.level}) đi: {p.symbol} tới ({r}, {c})")
                    self._execute_move(p, r, c)
                    self.ai_move_result = None

                self.renderer.render_meshes()
                self.renderer.render_lights()
                
                current_time = time.time()
                if current_time - self.last_title_update > 1.0:
                    self.fps = self.frame_count / (current_time - self.last_title_update)
                    self.frame_count = 0
                    self.last_title_update = current_time

                    if not self.game_over:
                        w_m, w_s = divmod(int(self.white_time), 60)
                        b_m, b_s = divmod(int(self.black_time), 60)
                        turn_str = "White" if self.turn == 'white' else "Black"
                        title = f"3D Chess | FPS: {int(self.fps)} | Turn: {turn_str} | W: {w_m:02}:{w_s:02} | B: {b_m:02}:{b_s:02}"
                        sdl2.SDL_SetWindowTitle(self.renderer.window.window, title.encode('utf-8'))
                    else:
                        win_txt = self.winner.upper() if self.winner else "GAME OVER"
                        title = f"GAME OVER | FPS: {int(self.fps)} | {win_txt} WON"
                        sdl2.SDL_SetWindowTitle(self.renderer.window.window, title.encode('utf-8'))

                if not self.game_over:
                    if self.selected_piece and not self.promotion_pending:
                        mid = self.piece_to_mesh.get(self.selected_piece)
                        if mid is not None: self.renderer.render_bounding_box(mid, 1, (255, 0, 0, 255))
                        for r, c in self.valid_moves:
                            t_mid = self.tile_to_mesh.get((c, r))
                            if t_mid: self.renderer.render_bounding_box(t_mid, 2, (0, 100, 255, 255))
                    if self.promotion_pending:
                        mid = self.piece_to_mesh.get(self.promotion_pending[0])
                        if mid: self.renderer.render_bounding_box(mid, 1, (255, 215, 0, 255))
                
                self.renderer.update_light()
                self.renderer.present()
                sdl2.SDL_Delay(1)  # Giới hạn FPS để tránh sử dụng CPU quá mức
        except KeyboardInterrupt:
            print("\n⚠️ Stop.")
        except Exception as e:
            print(f"\n❌ CRASH: {e}")
            traceback.print_exc()
        finally:
            self.renderer.cleanup()
            print("✅ Exit.")

if __name__ == "__main__":
    if not compile_renderer():
        sys.exit(1)
    game = Chess3D()
    game.run()