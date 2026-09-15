import numpy as np
import time
from collections import OrderedDict

class MinimaxAgent:
    def __init__(self, game_class, max_depth=20, use_tt=True, tt_size=11e6):
        self.game_class = game_class
        self.max_depth = max_depth
        self.use_tt = use_tt
        self.tt = OrderedDict() if use_tt else None
        self.tt_size = int(tt_size)
        self.nodes_searched = 0
        self.tt_hits = 0
        
        dummy = game_class()
        self.action_size = dummy.action_size()
        self.board_size = getattr(dummy, 'n', 3)
    
    def select_action(self, game, temperature=0.0, return_root=False):
        action = self.search(game)
        if action is None:
            legal = game.legal_actions()
            action = legal[0] if legal else None
        
        probs = np.zeros(self.action_size)
        if action is not None:
            probs[action] = 1.0
        
        if return_root:
            return probs, None
        return probs
    
    def search(self, game, time_limit=None):
        self.nodes_searched = 0
        self.tt_hits = 0
        
        cp = getattr(game, 'current_player_id', None)
        self.root_player = cp() if callable(cp) else cp
        
        if self.root_player == 0:
            return None
            
        start_time = time.time()
        legal = game.legal_actions()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]
        
        n = getattr(game, 'n', self.board_size)
        def action_priority(a):
            row, col = a // n, a % n
            return abs(row - (n-1)/2) + abs(col - (n-1)/2)
        legal.sort(key=action_priority)
        
        best_action = legal[0]
        best_value = -float('inf')
        
        for action in legal:
            g = game.clone()
            g.step(action)
            value = self._min_value(g, self.max_depth - 1, -float('inf'), float('inf'), start_time, time_limit)
            if time_limit and (time.time() - start_time) > time_limit:
                break
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _eval(self, game):
        board = game.board
        player = getattr(game, 'current_player_id', None)
        player = player() if callable(player) else player
        opponent = 3 - player
        score = 0
        lines = []
        n = getattr(game, 'n', self.board_size)
        for r in range(n):
            lines.append(board[r, :])
        for c in range(n):
            lines.append(board[:, c])
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))
        for line in lines:
            p_count = np.sum(line == player)
            o_count = np.sum(line == opponent)
            if p_count > 0 and o_count == 0:
                score += 10 ** (p_count - 1)
            if o_count > 0 and p_count == 0:
                score -= 10 ** (o_count - 1)
        return np.tanh(score / 100.0)
    
    def _tt_lookup(self, key, depth, alpha, beta):
        if not self.use_tt or key not in self.tt:
            return None
        self.tt.move_to_end(key)
        self.tt_hits += 1
        value, stored_depth, flag = self.tt[key]
        if stored_depth < depth:
            return None
        if flag == 'exact':
            return value
        return None
    
    def _tt_store(self, key, value, depth, flag):
        if not self.use_tt:
            return
        if len(self.tt) >= self.tt_size:
            self.tt.popitem(last=False)
        self.tt[key] = (value, depth, flag)
    
    def _hash(self, game):
        player = getattr(game, 'current_player_id', None)
        player = player() if callable(player) else player
        return (game.board.tobytes(), player, self.root_player)
    
    def _min_value(self, game, depth, alpha, beta, start_time, time_limit):
        self.nodes_searched += 1
        if time_limit and (time.time() - start_time) > time_limit:
            return 0
        result = game.result()
        if result is not None:
            return result if self.root_player == 1 else -result
        if depth == 0:
            return self._eval(game)
        key = self._hash(game)
        tt_entry = self._tt_lookup(key, depth, alpha, beta)
        if tt_entry is not None:
            return tt_entry
        value = float('inf')
        legal = game.legal_actions()
        n = getattr(game, 'n', self.board_size)
        legal.sort(key=lambda a: abs((a // n) - (n-1)/2) + abs((a % n) - (n-1)/2))
        for action in legal:
            g = game.clone()
            g.step(action)
            value = min(value, self._max_value(g, depth - 1, alpha, beta, start_time, time_limit))
            if value <= alpha:
                self._tt_store(key, value, depth, 'upper')
                return value
            beta = min(beta, value)
        self._tt_store(key, value, depth, 'exact')
        return value
    
    def _max_value(self, game, depth, alpha, beta, start_time, time_limit):
        self.nodes_searched += 1
        if time_limit and (time.time() - start_time) > time_limit:
            return 0
        result = game.result()
        if result is not None:
            return result if self.root_player == 1 else -result
        if depth == 0:
            return self._eval(game)
        key = self._hash(game)
        tt_entry = self._tt_lookup(key, depth, alpha, beta)
        if tt_entry is not None:
            return tt_entry
        value = -float('inf')
        legal = game.legal_actions()
        n = getattr(game, 'n', self.board_size)
        legal.sort(key=lambda a: abs((a // n) - (n-1)/2) + abs((a % n) - (n-1)/2))
        for action in legal:
            g = game.clone()
            g.step(action)
            value = max(value, self._min_value(g, depth - 1, alpha, beta, start_time, time_limit))
            if value >= beta:
                self._tt_store(key, value, depth, 'lower')
                return value
            alpha = max(alpha, value)
        self._tt_store(key, value, depth, 'exact')
        return value
    
    def save(self, path):
        import pickle
        data = {
            'max_depth': self.max_depth,
            'use_tt': self.use_tt,
            'tt_size': self.tt_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.max_depth = data.get('max_depth', self.max_depth)
        self.use_tt = data.get('use_tt', self.use_tt)
        self.tt_size = data.get('tt_size', self.tt_size)