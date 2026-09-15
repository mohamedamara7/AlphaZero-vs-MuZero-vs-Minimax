import sys
sys.path.insert(0, '/mnt/agents/output')

import numpy as np
import torch
import random

from games import TicTacToe
from agents import MuZeroAgent, AlphaZeroAgent

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate_vs_random(agent, game_class, num_games=20, agent_first=None):
    """Evaluate agent against random opponent."""
    results = {'win': 0, 'loss': 0, 'draw': 0}
    for _ in range(num_games):
        game = game_class()
        if agent_first is None:
            agent_player = random.choice([1, 2])
        else:
            agent_player = 1 if agent_first else 2
        while game.result() is None:
            if game.current_player_id == agent_player:
                probs = agent.select_action(game, temperature=0)
                action = np.argmax(probs)
            else:
                action = random.choice(game.legal_actions())
            game.step(action)
        r = game.result()
        if (r == 1 and agent_player == 1) or (r == -1 and agent_player == 2):
            results['win'] += 1
        elif r == 0:
            results['draw'] += 1
        else:
            results['loss'] += 1
    return results

def play_game_verbose(agent1, agent2, game_class, temperature=0):
    """Play a game between two agents and print the board."""
    game = game_class()
    move = 0
    while game.result() is None:
        if game.current_player_id == 1:
            probs = agent1.select_action(game, temperature=temperature)
        else:
            probs = agent2.select_action(game, temperature=temperature)
        action = np.argmax(probs)
        game.step(action)
        move += 1
    print(f"Game ended after {move} moves. Result: {game.result()}")
    print("Final board (1=P1, 2=P2):")
    print(game.board)
    return game.result()

# ===========================
# Run Tests
# ===========================
set_seed(42)

print("=" * 70)
print("TESTING MUZERO ON 4x4 TIC-TAC-TOE")
print("=" * 70)

muzero_agent = MuZeroAgent(
    TicTacToe, 
    num_simulations=20, 
    lr=0.001, 
    batch_size=32,
    num_unroll_steps=3, 
    td_steps=5, 
    discount=1.0,
    latent_dim=64,
    num_blocks=2
)

print("\\n--- MuZero: Before Training (vs Random) ---")
print(evaluate_vs_random(muzero_agent, TicTacToe, 20))

print("\\n--- MuZero: Training ---")
muzero_agent.train(num_epochs=5, games_per_epoch=5, print_interval=1)

print("\\n--- MuZero: After Training (vs Random) ---")
print(evaluate_vs_random(muzero_agent, TicTacToe, 20))

print("\\n" + "=" * 70)
print("TESTING ALPHAZERO ON 4x4 TIC-TAC-TOE")
print("=" * 70)

az_agent = AlphaZeroAgent(
    TicTacToe, 
    num_simulations=20, 
    lr=0.001, 
    batch_size=32,
    num_blocks=2,
    channels=64
)

print("\\n--- AlphaZero: Before Training (vs Random) ---")
print(evaluate_vs_random(az_agent, TicTacToe, 20))

print("\\n--- AlphaZero: Training ---")
az_agent.train(num_epochs=5, games_per_epoch=5, print_interval=1)

print("\\n--- AlphaZero: After Training (vs Random) ---")
print(evaluate_vs_random(az_agent, TicTacToe, 20))

print("\\n" + "=" * 70)
print("SAMPLE GAME: MuZero (P1) vs AlphaZero (P2)")
print("=" * 70)
result = play_game_verbose(muzero_agent, az_agent, TicTacToe, temperature=0)