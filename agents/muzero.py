import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

class MinMaxStats:
    def __init__(self, known_bounds=None):
        self.maximum = -float('inf')
        self.minimum = float('inf')
        if known_bounds is not None:
            self.maximum = known_bounds['max']
            self.minimum = known_bounds['min']
    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class MuZeroNet(nn.Module):
    def __init__(self, obs_shape, action_size, latent_dim=64, num_blocks=3):
        super().__init__()
        self.action_size = action_size
        self.latent_dim = latent_dim
        in_channels = obs_shape[0]
        h, w = obs_shape[1], obs_shape[2]
        
        self.repr_conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.repr_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.repr_fc = nn.Linear(32 * h * w, latent_dim)
        
        self.dynamics_action_embed = nn.Embedding(action_size, latent_dim)
        self.dynamics_fc1 = nn.Linear(latent_dim * 2, latent_dim)
        self.dynamics_fc2 = nn.Linear(latent_dim, latent_dim)
        self.dynamics_reward = nn.Linear(latent_dim, 1)
        
        self.pred_policy = nn.Linear(latent_dim, action_size)
        self.pred_value = nn.Linear(latent_dim, 1)

    def representation(self, obs):
        x = F.relu(self.repr_conv1(obs))
        x = F.relu(self.repr_conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.repr_fc(x))
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x

    def dynamics(self, hidden_state, action):
        action_embed = self.dynamics_action_embed(action)
        x = torch.cat([hidden_state, action_embed], dim=1)
        x = F.relu(self.dynamics_fc1(x))
        x = F.relu(self.dynamics_fc2(x))
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        reward = torch.tanh(self.dynamics_reward(x))
        return x, reward.squeeze(-1)

    def prediction(self, hidden_state):
        policy_logits = self.pred_policy(hidden_state)
        value = torch.tanh(self.pred_value(hidden_state))
        return policy_logits, value.squeeze(-1)

    def initial_inference(self, obs):
        hidden_state = self.representation(obs)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value

class MuZeroMCTSNode:
    def __init__(self, prior=0.0, hidden_state=None, reward=0.0, game_state=None):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
        self.hidden_state = hidden_state
        self.reward = reward
        self.is_expanded = False
        self.game_state = game_state  

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MuZeroMCTS:
    def __init__(self, network, num_simulations=50, c_init=1.25, c_base=19652,
                 discount=1.0, dirichlet_alpha=0.3, noise_frac=0.25, device='cpu'):
        self.network = network
        self.num_simulations = num_simulations
        self.c_init = c_init
        self.c_base = c_base
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self.device = device

    def search(self, game, temperature=1.0, add_noise=True):
        self.min_max_stats = MinMaxStats(known_bounds={'min': -1, 'max': 1})
        
        self.network.eval()
        
        obs = torch.tensor(game.canonical_observation(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hidden_state, policy_logits, value = self.network.initial_inference(obs)
            
        root = MuZeroMCTSNode(hidden_state=hidden_state, game_state=game.clone())
        
        logits = policy_logits.squeeze(0).cpu().numpy()
        legal = game.legal_actions()
        mask = np.full_like(logits, -np.inf)
        mask[legal] = 0
        logits += mask
        policy = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).numpy()
        
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            for i, action in enumerate(legal):
                prior = policy[action] * (1 - self.noise_frac) + noise[i] * self.noise_frac
                root.children[action] = MuZeroMCTSNode(prior=prior)
        else:
            for action in legal:
                root.children[action] = MuZeroMCTSNode(prior=policy[action])
        root.is_expanded = True
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            while node.is_expanded and len(node.children) > 0:
                action, node = self._select_child(node)
                search_path.append(node)
                
            parent = search_path[-2]
            action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                next_hidden_state, reward, policy_logits, value = self.network.recurrent_inference(
                    parent.hidden_state, action_tensor)
                    
            node.hidden_state = next_hidden_state
            node.reward = reward.item()
            
            if parent.game_state is not None:
                g = parent.game_state.clone()
                g.step(action)
                node.game_state = g
                
                if g.result() is not None:
                    if g.winner == 0:
                        terminal_value = 0.0
                    else:
                        # See note in alphazero.py: current_player_id is not toggled on a
                        # terminal step, so it still equals the winner. Value must be expressed
                        # from the (hypothetical) next-mover's perspective for consistency with
                        # the alternating sign convention used elsewhere in the tree.
                        terminal_value = -1.0 if g.winner == g.current_player_id else 1.0
                    
                    self._backup(search_path, terminal_value)
                    continue
                
                legal = g.legal_actions()
            else:
                legal = list(range(self.network.action_size))
            
            logits = policy_logits.squeeze(0).cpu().numpy()
            mask = np.full_like(logits, -np.inf)
            if len(legal) > 0:
                mask[legal] = 0
            logits += mask
            policy = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).numpy()
            
            for a in legal:
                node.children[a] = MuZeroMCTSNode(prior=policy[a])
            node.is_expanded = True
            
            value = value.item()
            self._backup(search_path, value)
            
        visits = np.zeros(game.action_size())
        for action, child in root.children.items():
            visits[action] = child.visit_count
            
        if temperature == 0:
            action = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[action] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)
            
        return probs, root

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in node.children.items():
            q = -child.value()
            q_norm = self.min_max_stats.normalize(q)
            pb_c = np.log((node.visit_count + self.c_base + 1) / self.c_base) + self.c_init
            pb_c *= np.sqrt(node.visit_count) / (child.visit_count + 1)
            u = pb_c * child.prior
            score = q_norm + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _backup(self, search_path, value):
        for i in range(len(search_path) - 1, -1, -1):
            node = search_path[i]
            node.value_sum += value
            node.visit_count += 1
            self.min_max_stats.update(node.value())
            if i > 0:
                value = node.reward - self.discount * value

class MuZeroReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add_game(self, states, actions, policies, rewards, root_values):
        self.buffer.append({
            'states': states,
            'actions': actions,
            'policies': policies,
            'rewards': rewards,
            'root_values': root_values
        })

    def sample_batch(self, batch_size, num_unroll_steps=5, td_steps=10, discount=1.0):
        games = random.choices(self.buffer, k=batch_size)
        batch = []
        
        for game in games:
            max_pos = len(game['states']) - 1
            if max_pos < 0:
                continue
                
            pos = random.randint(0, max_pos)
            state = game['states'][pos]
            actions = game['actions'][pos:pos + num_unroll_steps]
            while len(actions) < num_unroll_steps:
                actions.append(0)
                
            targets = []
            for k in range(num_unroll_steps + 1):
                current_idx = pos + k
                bootstrap_idx = current_idx + td_steps
                
                if bootstrap_idx < len(game['root_values']):
                    bootstrap_value = game['root_values'][bootstrap_idx]
                    if td_steps % 2 == 1:
                        bootstrap_value = -bootstrap_value
                    value = bootstrap_value * (discount ** td_steps)
                else:
                    value = 0.0
                    
                for i, reward in enumerate(game['rewards'][current_idx:bootstrap_idx]):
                    r = reward if i % 2 == 0 else -reward
                    value += r * (discount ** i)
                    
                if k == 0:
                    reward_target = 0.0
                else:
                    reward_target = game['rewards'][current_idx - 1] if (current_idx - 1) < len(game['rewards']) else 0.0
                    
                if current_idx < len(game['policies']):
                    policy_target = game['policies'][current_idx]
                else:
                    policy_target = np.zeros_like(game['policies'][0]) if game['policies'] else np.zeros(9)
                    
                targets.append((value, reward_target, policy_target))
                
            batch.append((state, actions, targets))
            
        return batch

    def __len__(self):
        return len(self.buffer)

class MuZeroAgent:
    def __init__(self, game_class, num_simulations=50,
                 lr=0.001, batch_size=32, buffer_size=100000,
                 latent_dim=64, num_blocks=3, num_unroll_steps=5, td_steps=10, discount=1.0):
        self.game_class = game_class
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.discount = discount
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dummy = game_class()
        obs_shape = dummy.observation_shape()
        action_size = dummy.action_size()
        
        self.network = MuZeroNet(obs_shape, action_size, latent_dim, num_blocks).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.mcts = MuZeroMCTS(self.network, num_simulations, discount=discount, device=self.device)
        self.replay_buffer = MuZeroReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.action_size = action_size

    def select_action(self, game, temperature=1.0, return_root=False):
        add_noise = (temperature > 0)
        probs, root = self.mcts.search(game, temperature, add_noise=add_noise)
        if return_root:
            return probs, root
        return probs

    def self_play(self, num_games=10, temperature=1.0):
        for _ in range(num_games):
            game = self.game_class()
            states = []
            actions = []
            policies = []
            rewards = []
            root_values = []
            
            while game.result() is None:
                probs, root = self.select_action(game, temperature, return_root=True)
                states.append(game.canonical_observation().copy())
                policies.append(probs)
                root_values.append(root.value())
                action = np.random.choice(len(probs), p=probs)
                actions.append(action)
                
                obs, reward, done, info = game.step(action)
                rewards.append(reward)
                
            self.replay_buffer.add_game(states, actions, policies, rewards, root_values)

    def train(self, num_epochs=5, games_per_epoch=10, steps_per_epoch=100, print_interval=1):
        print(f"Training MuZero for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.self_play(games_per_epoch)
            
            if len(self.replay_buffer) < 2:
                print(f"Epoch {epoch+1}: Buffer too small ({len(self.replay_buffer)}), skipping.")
                continue
            
            epoch_losses = []
            epoch_value_losses = []
            epoch_reward_losses = []
            epoch_policy_losses = []
            
            for step in range(steps_per_epoch):
                batch = self.replay_buffer.sample_batch(
                    self.batch_size, self.num_unroll_steps, self.td_steps, self.discount)
                
                if len(batch) < self.batch_size // 2:
                    break
                
                self.network.train()
                self.optimizer.zero_grad()
                
                total_loss = 0
                total_value_loss = 0
                total_reward_loss = 0
                total_policy_loss = 0
                
                for state, actions, targets in batch:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
                    
                    hidden_state, policy_logits, value = self.network.initial_inference(state_tensor)
                    
                    predictions = [(1.0, value, torch.zeros(1).to(self.device), policy_logits)]
                    
                    for i, action in enumerate(actions_tensor):
                        hidden_state = ScaleGradient.apply(hidden_state, 0.5)
                        hidden_state, reward, policy_logits, value = self.network.recurrent_inference(
                            hidden_state, action.unsqueeze(0))
                        predictions.append((1.0 / len(actions), value, reward, policy_logits))
                        
                    loss = 0
                    for i, (scale, pred_value, pred_reward, pred_policy) in enumerate(predictions):
                        target_value, target_reward, target_policy = targets[i]
                        
                        target_value_t = torch.tensor(target_value, dtype=torch.float32).to(self.device)
                        target_reward_t = torch.tensor(target_reward, dtype=torch.float32).to(self.device)
                        target_policy_t = torch.tensor(target_policy, dtype=torch.float32).unsqueeze(0).to(self.device)
                        
                        v_loss = F.mse_loss(pred_value.squeeze(), target_value_t)
                        r_loss = F.mse_loss(pred_reward.squeeze(), target_reward_t)
                        p_loss = -(target_policy_t * F.log_softmax(pred_policy, dim=1)).sum(dim=1).mean()
                        
                        loss += scale * (v_loss + r_loss + p_loss)
                        
                        total_value_loss += v_loss.item()
                        total_reward_loss += r_loss.item()
                        total_policy_loss += p_loss.item()
                        
                    total_loss += loss.item()
                    loss.backward()
                    
                self.optimizer.step()
                
                n = len(batch)
                epoch_losses.append(total_loss / n)
                epoch_value_losses.append(total_value_loss / n)
                epoch_reward_losses.append(total_reward_loss / n)
                epoch_policy_losses.append(total_policy_loss / n)
                
            if (epoch + 1) % print_interval == 0 and len(epoch_losses) > 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                      f"V: {np.mean(epoch_value_losses):.4f} | R: {np.mean(epoch_reward_losses):.4f} | "
                      f"P: {np.mean(epoch_policy_losses):.4f} | Buffer: {len(self.replay_buffer)}")

    def save(self, path):
        torch.save({'network': self.network.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])