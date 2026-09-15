import argparse
from games import TicTacToe, Connect4
from agents import AlphaZeroAgent, MuZeroAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='tictactoe', choices=['tictactoe', 'connect4'])
    parser.add_argument('--agent', type=str, default='alphazero', choices=['alphazero', 'muzero'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--games_per_epoch', type=int, default=10)
    parser.add_argument('--simulations', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default='checkpoint.pt')
    args = parser.parse_args()
    
    game_class = TicTacToe if args.game == 'tictactoe' else Connect4
    
    if args.agent == 'alphazero':
        agent = AlphaZeroAgent(game_class, num_simulations=args.simulations, lr=args.lr, batch_size=32)
    else:
        agent = MuZeroAgent(game_class, num_simulations=args.simulations, lr=args.lr, batch_size=32)
    
    agent.train(num_epochs=args.epochs, games_per_epoch=args.games_per_epoch, print_interval=1)
    agent.save(args.save_path)
    print(f"Saved to {args.save_path}")

if __name__ == '__main__':
    main()