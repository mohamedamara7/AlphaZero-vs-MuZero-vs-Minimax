import numpy as np
from .base import Game

class Connect4(Game):
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.current_player_id = 1
        self.done = False
        self.winner = None
    
    def reset(self):
        self.__init__(self.rows, self.cols)
        return self
    
    def canonical_observation(self):
        obs = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        if self.current_player_id == 1:
            obs[0] = (self.board == 1).astype(np.float32)
            obs[1] = (self.board == 2).astype(np.float32)
            obs[2] = 1.0
        else:
            obs[0] = (self.board == 2).astype(np.float32)
            obs[1] = (self.board == 1).astype(np.float32)
            obs[2] = -1.0
        return obs
    
    def observation_shape(self):
        return (3, self.rows, self.cols)
    
    def action_size(self):
        return self.cols
    
    def legal_actions(self):
        if self.done:
            return []
        return [c for c in range(self.cols) if self.board[0, c] == 0]
    
    def current_player(self):
        return 1 if self.current_player_id == 1 else -1
    
    def step(self, action):
        if self.done:
            return self.canonical_observation(), 0.0, True, {}
        
        if action < 0 or action >= self.cols or self.board[0, action] != 0:
            self.done = True
            self.winner = 3 - self.current_player_id
            reward = -1.0
            return self.canonical_observation(), reward, True, {}
        
        # Find lowest empty row in the chosen column
        row = self.rows - 1
        while row >= 0 and self.board[row, action] != 0:
            row -= 1
        
        self.board[row, action] = self.current_player_id
        
        if self._check_win(row, action):
            self.done = True
            self.winner = self.current_player_id
            reward = 1.0
            return self.canonical_observation(), reward, True, {}
        
        if np.all(self.board[0, :] != 0):
            self.done = True
            self.winner = 0
            reward = 0.0
            return self.canonical_observation(), reward, True, {}
        
        self.current_player_id = 3 - self.current_player_id
        return self.canonical_observation(), 0.0, False, {}
    
    def _check_win(self, row, col):
        player = self.board[row, col]
        
        # Horizontal
        count = 0
        for c in range(self.cols):
            if self.board[row, c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
        
        # Vertical
        count = 0
        for r in range(self.rows):
            if self.board[r, col] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
        
        # Diagonal (top-left to bottom-right)
        count = 0
        start = min(row, col)
        r, c = row - start, col - start
        while r < self.rows and c < self.cols:
            if self.board[r, c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
            r += 1
            c += 1
        
        # Anti-diagonal (top-right to bottom-left)
        count = 0
        start = min(row, self.cols - 1 - col)
        r, c = row - start, col + start
        while r < self.rows and c >= 0:
            if self.board[r, c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
            r += 1
            c -= 1
        
        return False
    
    def result(self):
        if not self.done:
            return None
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == 1 else -1.0
    
    def clone(self):
        g = Connect4(self.rows, self.cols)
        g.board = self.board.copy()
        g.current_player_id = self.current_player_id
        g.done = self.done
        g.winner = self.winner
        return g
    
    def render(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        for r in range(self.rows):
            lines.append(' '.join(symbols[v] for v in self.board[r]))
        lines.append(' '.join(str(c) for c in range(self.cols)))
        return '\n'.join(lines)
    
    def hash_state(self):
        return (self.board.tobytes(), self.current_player_id)