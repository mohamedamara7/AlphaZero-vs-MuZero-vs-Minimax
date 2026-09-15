import numpy as np
import time
import torch
from games import TicTacToe, Connect4
from agents import MinimaxAgent, AlphaZeroAgent, MuZeroAgent

def play_match(agent1, agent2, game_class, num_games=20, temperature=0.0, verbose=False):
    """Play matches. Agents swap sides each game."""
    results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0, 
               'agent1_time': 0, 'agent2_time': 0, 'total_moves': 0}
    
    for game_num in range(num_games):
        game = game_class()
        # Even games: agent1=P1, agent2=P2. Odd games: swapped.
        agents = [agent1, agent2] if game_num % 2 == 0 else [agent2, agent1]
        names = ['Agent1', 'Agent2'] if game_num % 2 == 0 else ['Agent2', 'Agent1']
        
        if verbose:
            print(f"\nGame {game_num + 1}: {names[0]} (P1) vs {names[1]} (P2)")
        
        move_count = 0
        while game.result() is None:
            current = game.current_player() - 1
            agent = agents[current]
            name = names[current]
            
            start_time = time.time()
            if hasattr(agent, 'select_action'):
                probs = agent.select_action(game, temperature=temperature)
                if isinstance(probs, tuple):
                    probs = probs[0]
                action = np.argmax(probs) if temperature == 0 else np.random.choice(len(probs), p=probs)
            else:
                action = agent.search(game)
            elapsed = time.time() - start_time
            
            # Track time by agent identity, not player number
            if name == 'Agent1':
                results['agent1_time'] += elapsed
            else:
                results['agent2_time'] += elapsed
            
            game.step(action)
            move_count += 1
            if verbose:
                print(f"Move {move_count}: {name} plays {action}")
                print(game.render())
        
        results['total_moves'] += move_count
        outcome = game.result()
        if outcome == 0:
            results['draws'] += 1
        elif outcome == 1:
            # P1 won
            if game_num % 2 == 0:
                results['agent1_wins'] += 1
            else:
                results['agent2_wins'] += 1
        else:
            # P2 won
            if game_num % 2 == 0:
                results['agent2_wins'] += 1
            else:
                results['agent1_wins'] += 1
    
    return results

def evaluate_all_agents(game_class=TicTacToe, num_games=10, minimax_depth=4):
    print("=" * 60)
    print(f"Evaluating agents on {game_class.__name__}")
    print("=" * 60)
    
    print("\nInitializing agents...")
    minimax = MinimaxAgent(max_depth=minimax_depth, use_tt=True)
    alphazero = AlphaZeroAgent(game_class, num_simulations=50, num_blocks=2, channels=32, batch_size=32)
    muzero = MuZeroAgent(game_class, num_simulations=50, latent_dim=64, num_blocks=2, batch_size=32)
    
    print("\n--- Training AlphaZero ---")
    alphazero.train(num_epochs=3, games_per_epoch=5, print_interval=1)
    
    print("\n--- Training MuZero ---")
    muzero.train(num_epochs=3, games_per_epoch=5, print_interval=1)
    
    print("\n--- Training Complete ---")
    
    agents = {'Minimax': minimax, 'AlphaZero': alphazero, 'MuZero': muzero}
    
    print("\n" + "=" * 60)
    print("MATCH RESULTS")
    print("=" * 60)
    
    for name1, agent1 in agents.items():
        for name2, agent2 in agents.items():
            if name1 == name2:
                continue
            print(f"\n{name1} vs {name2} ({num_games} games)...")
            res = play_match(agent1, agent2, game_class, num_games=num_games, temperature=0.0)
            print(f"  {name1} wins: {res['agent1_wins']}")
            print(f"  {name2} wins: {res['agent2_wins']}")
            print(f"  Draws: {res['draws']}")
            avg_moves = max(1, res['total_moves'] // num_games)
            print(f"  {name1} avg time/move: {res['agent1_time']/max(1, res['total_moves']//2):.4f}s")
            print(f"  {name2} avg time/move: {res['agent2_time']/max(1, res['total_moves']//2):.4f}s")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    test_game = game_class()
    print("\nSingle move thinking time (avg of 5 calls):")
    for name, agent in agents.items():
        times = []
        for _ in range(5):
            g = game_class()
            start = time.time()
            if hasattr(agent, 'select_action'):
                agent.select_action(g, temperature=0.0)
            else:
                agent.search(g)
            times.append(time.time() - start)
        print(f"  {name}: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
    
    print(f"\nMinimax statistics (last search):")
    print(f"  Nodes searched: {minimax.nodes_searched}")
    print(f"  TT hits: {minimax.tt_hits}")
    print(f"  TT size: {len(minimax.tt) if minimax.tt else 0}")

if __name__ == '__main__':
    evaluate_all_agents(game_class=TicTacToe, num_games=10, minimax_depth=4)