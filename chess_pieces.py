class Piece:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.symbol = " "
        self.has_moved = False

    def move(self, row, col):
        self.row = row
        self.col = col
        self.has_moved = True

    def is_enemy(self, other):
        return other is not None and other.color != self.color

class SliderPiece(Piece):
    def get_sliding_moves(self, board, directions):
        moves = []
        for dr, dc in directions:
            r, c = self.row + dr, self.col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                target = board[r][c]
                if target is None:
                    moves.append((r, c))
                elif self.is_enemy(target):
                    moves.append((r, c))
                    break
                else:
                    break
                r += dr
                c += dc
        return moves

class Rook(SliderPiece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = "R" if color == "white" else "r"
    def get_pseudo_moves(self, board):
        return self.get_sliding_moves(board, [(-1, 0), (1, 0), (0, -1), (0, 1)])

class Bishop(SliderPiece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = "B" if color == "white" else "b"
    def get_pseudo_moves(self, board):
        return self.get_sliding_moves(board, [(-1, -1), (-1, 1), (1, -1), (1, 1)])

class Queen(SliderPiece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = "Q" if color == "white" else "q"
    def get_pseudo_moves(self, board):
        orthogonal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return self.get_sliding_moves(board, orthogonal + diagonal)

class Knight(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = "N" if color == "white" else "n"
    def get_pseudo_moves(self, board):
        moves = []
        offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                   (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, dc in offsets:
            r, c = self.row + dr, self.col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                target = board[r][c]
                if target is None or self.is_enemy(target):
                    moves.append((r, c))
        return moves

class King(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = "K" if color == "white" else "k"
    
    def get_pseudo_moves(self, board, check_callback=None):
        moves = []
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in offsets:
            r, c = self.row + dr, self.col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                target = board[r][c]
                if target is None or self.is_enemy(target):
                    moves.append((r, c))
        
        # Castling logic
        if not self.has_moved and check_callback and not check_callback(self.color):
            row = self.row
            # Kingside
            ks_rook = board[row][7]
            if isinstance(ks_rook, Rook) and not ks_rook.has_moved:
                if board[row][5] is None and board[row][6] is None:
                    # Check if squares are not attacked (need callback)
                    moves.append((row, 6))
            # Queenside
            qs_rook = board[row][0]
            if isinstance(qs_rook, Rook) and not qs_rook.has_moved:
                if board[row][1] is None and board[row][2] is None and board[row][3] is None:
                    moves.append((row, 2))
        return moves

class Pawn(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = "P" if color == "white" else "p"
        self.direction = -1 if color == "white" else 1
        self.start_row = 6 if color == "white" else 1

    def get_pseudo_moves(self, board, en_passant_target=None):
        moves = []
        r1 = self.row + self.direction
        if 0 <= r1 < 8 and board[r1][self.col] is None:
            moves.append((r1, self.col))
            r2 = self.row + 2 * self.direction
            if self.row == self.start_row and board[r2][self.col] is None:
                moves.append((r2, self.col))
        
        for dc in [-1, 1]:
            r_diag, c_diag = self.row + self.direction, self.col + dc
            if 0 <= r_diag < 8 and 0 <= c_diag < 8:
                target = board[r_diag][c_diag]
                if target is not None and self.is_enemy(target):
                    moves.append((r_diag, c_diag))
                elif en_passant_target and (r_diag, c_diag) == en_passant_target:
                    moves.append((r_diag, c_diag))
        return moves