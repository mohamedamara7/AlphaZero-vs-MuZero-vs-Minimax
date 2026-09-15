import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class AlphaZeroNet(nn.Module):
    def __init__(self, obs_shape, action_size, num_blocks=3, channels=64):
        super().__init__()
        in_channels = obs_shape[0]
        self.conv_init = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
        
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * obs_shape[1] * obs_shape[2], action_size)
        
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * obs_shape[1] * obs_shape[2], 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.blocks:
            x = block(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value.squeeze(-1)

class MCTSNode:
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0: return 0.0
        return self.value_sum / self.visit_count

class AlphaZeroMCTS:
    def __init__(self, network, num_simulations=50, c_puct=1.0, dirichlet_alpha=0.3, noise_frac=0.25, device='cpu'):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self.device = device

    def search(self, game, temperature=1.0, add_noise=True):
        root = MCTSNode()
        self.network.eval()
        
        obs = torch.tensor(game.canonical_observation(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.network(obs)
            
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
                root.children[action] = MCTSNode(prior=prior)
        else:
            for action in legal:
                root.children[action] = MCTSNode(prior=policy[action])
        root.is_expanded = True
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            g = game.clone()
            
            while node.is_expanded and len(node.children) > 0:
                action, node = self._select_child(node)
                g.step(action)
                search_path.append(node)
                if g.result() is not None:
                    break
                    
            if g.result() is not None:
                # Terminal state reached
                if g.winner == 0:
                    value = 0.0  # Draw
                else:
                    # NOTE: current_player_id is NOT toggled on a terminal step, so it still
                    # equals the player who just moved (the winner). Node values must be kept
                    # in the convention "value for the player about to act at this node" so that
                    # the alternating negation in _backup / _select_child stays consistent.
                    # Since the mover just won, the value from that (non-existent) next-mover's
                    # perspective is -1, not +1.
                    value = -1.0 if g.winner == g.current_player_id else 1.0
            else:
                obs = torch.tensor(g.canonical_observation(), dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy_logits, value_tensor = self.network(obs)
                value = value_tensor.item()
                
                logits = policy_logits.squeeze(0).cpu().numpy()
                legal = g.legal_actions()
                mask = np.full_like(logits, -np.inf)
                mask[legal] = 0
                logits += mask
                policy = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).numpy()
                
                for action in legal:
                    node.children[action] = MCTSNode(prior=policy[action])
                node.is_expanded = True
                
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
            u = self.c_puct * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _backup(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (np.array(states), np.array(policies), np.array(values))

    def __len__(self):
        return len(self.buffer)

class AlphaZeroAgent:
    def __init__(self, game_class, num_simulations=50, c_puct=1.0,
                 lr=0.001, batch_size=32, buffer_size=100000,
                 num_blocks=3, channels=64):
        self.game_class = game_class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dummy = game_class()
        obs_shape = dummy.observation_shape()
        action_size = dummy.action_size()
        
        self.network = AlphaZeroNet(obs_shape, action_size, num_blocks, channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.mcts = AlphaZeroMCTS(self.network, num_simulations, c_puct, device=self.device)
        self.replay_buffer = ReplayBuffer(buffer_size)
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
            policies = []
            current_player = []
            
            while game.result() is None:
                probs, root = self.select_action(game, temperature, return_root=True)
                states.append(game.canonical_observation().copy())
                policies.append(probs)
                current_player.append(game.current_player())
                action = np.random.choice(len(probs), p=probs)
                game.step(action)
                
            outcome = game.result()
            for i, cp in enumerate(current_player):
                value = outcome if cp == 1 else -outcome
                self.replay_buffer.add(states[i], policies[i], value)

    def train(self, num_epochs=5, games_per_epoch=10, steps_per_epoch=100, print_interval=1):
        print(f"Training AlphaZero for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.self_play(games_per_epoch)
            
            if len(self.replay_buffer) < self.batch_size:
                print(f"Epoch {epoch+1}: Buffer too small ({len(self.replay_buffer)}), skipping.")
                continue
            
            self.network.train()
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []
            
            for step in range(steps_per_epoch):
                states, policies, values = self.replay_buffer.sample(self.batch_size)
                states = torch.tensor(states, dtype=torch.float32).to(self.device)
                policies = torch.tensor(policies, dtype=torch.float32).to(self.device)
                values = torch.tensor(values, dtype=torch.float32).to(self.device)
                
                self.optimizer.zero_grad()
                policy_logits, pred_values = self.network(states)
                
                policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_values, values)
                loss = policy_loss + value_loss
                
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
            
            if (epoch + 1) % print_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                    f"Policy: {np.mean(epoch_policy_losses):.4f} | "
                    f"Value: {np.mean(epoch_value_losses):.4f} | "
                    f"Buffer: {len(self.replay_buffer)}")

    def save(self, path):
        torch.save({'network': self.network.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])