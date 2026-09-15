# games/tictactoe.py
import numpy as np
from .base import Game

class TicTacToe(Game):
    def __init__(self, n=3):
        self.n = n
        self.board = np.zeros((n, n), dtype=np.int8)
        self.current_player_id = 1
        self.done = False
        self.winner = None
    
    def reset(self):
        self.__init__()
        return self

    def canonical_observation(self):
        obs = np.zeros((3, self.n, self.n), dtype=np.float32)
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
        return (3, self.n, self.n)

    def action_size(self):
        return self.n * self.n

    def legal_actions(self):
        if self.done:
            return []
        return [i for i in range(self.n * self.n) if self.board[i // self.n, i % self.n] == 0]

    def current_player(self):
        return 1 if self.current_player_id == 1 else -1

    def step(self, action):
        if self.done:
            return self.canonical_observation(), 0.0, True, {}
            
        row, col = action // self.n, action % self.n
        if self.board[row, col] != 0:
            self.done = True
            self.winner = 3 - self.current_player_id
            reward = -1.0
            return self.canonical_observation(), reward, True, {}
            
        self.board[row, col] = self.current_player_id
        
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player_id
            reward = 1.0
            return self.canonical_observation(), reward, True, {}
            
        if np.all(self.board != 0):
            self.done = True
            self.winner = 0
            reward = 0.0
            return self.canonical_observation(), reward, True, {}
            
        self.current_player_id = 3 - self.current_player_id
        return self.canonical_observation(), 0.0, False, {}

    def _check_win(self, row, col):
        player = self.board[row, col]
        if np.all(self.board[row, :] == player): return True
        if np.all(self.board[:, col] == player): return True
        if row == col and np.all(np.diag(self.board) == player): return True
        if row + col == self.n - 1 and np.all(np.diag(np.fliplr(self.board)) == player): return True
        return False

    def result(self):
        if not self.done: return None
        if self.winner == 0: return 0.0
        return 1.0 if self.winner == 1 else -1.0  # Always from Player 1's perspective!

    def clone(self):
        g = TicTacToe(self.n)
        g.board = self.board.copy()
        g.current_player_id = self.current_player_id
        g.done = self.done
        g.winner = self.winner
        return g
    
    def render(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        for i in range(self.n):
            lines.append(' '.join(symbols[v] for v in self.board[i]))
        return '\n'.join(lines)
    
    def hash_state(self):
        return (self.board.tobytes(), self.current_player_id)