# 🧠 RL Game Agents: Minimax, AlphaZero, and MuZero

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

A comprehensive, modular Reinforcement Learning framework implementing **Minimax**, **AlphaZero**, and **MuZero** agents for zero-sum board games, with a primary benchmark environment of **Tic-Tac-Toe**. 

This project serves as both an educational resource and a rigorous benchmark for comparing classical adversarial search methods with state-of-the-art model-free and model-based deep reinforcement learning algorithms.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Algorithm Architecture](#algorithm-architecture)
- [Architecture Details](#architecture-details) 
- [Usage & Examples](#usage)
- [Expected Results](#-expected-results)
- [License](#-license)

---

## 📖 Overview

The evolution of game-playing AI has shifted from handcrafted evaluation functions to self-taught deep reinforcement learning. This repository implements three distinct paradigms:
1. **Minimax**: The classical baseline, utilizing alpha-beta pruning and transposition tables for optimal, brute-force adversarial search.
2. **AlphaZero**: A model-free approach that learns a policy and value network purely through self-play and Monte Carlo Tree Search (MCTS), requiring no human domain knowledge.
3. **MuZero**: A cutting-edge model-based approach that learns the environment's dynamics (transition and reward models) in a latent space, allowing it to plan and achieve superhuman performance *without* knowing the underlying rules of the game.

---

## ✨ Features

- **Environments (`games/`)**:
  - Tic-Tac-Toe
  - Connect 4
  - Extensible base game class for custom implementations.
- **Agents (`agents/`)**:
  - **AlphaZero**: Uses Monte Carlo Tree Search (MCTS) combined with a ResNet-based neural network for policy and value prediction.
  - **MuZero**: Learns the game dynamics implicitly through representation, dynamics, and prediction networks without needing the exact game rules during planning.
  - **Minimax**: A classic adversarial search agent featuring alpha-beta pruning and transposition tables (memoization) for perfect play evaluation.
---

##  Algorithm Architecture

### 1. Minimax Agent
- **Search**: Depth-limited minimax with alpha-beta pruning.
- **Optimization**: Transposition tables (hash maps) to cache previously evaluated board states, drastically reducing redundant computations.
- **Role**: Serves as the unbeatable theoretical baseline for perfect-information games.

### 2. AlphaZero Agent
- **Network**: A Convolutional Neural Network (CNN) with residual blocks outputting a policy vector (action probabilities) and a scalar value (expected game outcome).
- **Planning**: Monte Carlo Tree Search (MCTS) guided by the neural network's priors.
- **Training**: Self-play generation followed by supervised learning on (state, MCTS-policy, game-outcome) tuples.

### 3. MuZero Agent
- **Network**: Three distinct learned functions:
  - *Representation*: Maps the observable state to a latent hidden state.
  - *Dynamics*: Predicts the next latent state and immediate reward given a latent state and action.
  - *Prediction*: Outputs policy and value from a latent state.
- **Planning**: MCTS is performed entirely in the learned latent space, unrolling `num_unroll_steps` into the future.

---

## Architecture Details

- **AlphaZero Net**: Uses a shared representation base with ResNet blocks before splitting into a Policy head and a Value head.
- **MuZero Net**: Operates fully in latent space. Features a representation network to encode observations, a dynamics network to predict state transitions and rewards, and a prediction network to output policies and values.
---

##  Usage

### Training an Agent

You can train an AlphaZero or MuZero agent using `train.py`. The script accepts arguments to configure the game, agent, and hyperparameters.

```bash
# Train AlphaZero on Tic-Tac-Toe
python train.py --game tictactoe --agent alphazero --epochs 5 --games_per_epoch 10

# Train MuZero on Connect 4
python train.py --game connect4 --agent muzero --epochs 10 --lr 0.001 --save_path muzero_c4.pt
```

**Training Arguments:**
- `--game`: `tictactoe` (default) or `connect4`
- `--agent`: `alphazero` (default) or `muzero`
- `--epochs`: Number of training epochs (default: 5)
- `--games_per_epoch`: Games played per epoch (default: 10)
- `--simulations`: MCTS simulations per move (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--save_path`: Path to save the model checkpoint

### Evaluating Agents

Run `evaluate.py` to play agents against each other and calculate win rates, draws, and processing time per move.

```bash
python evaluate.py
```

### Testing against Random Baselines

Use `test.py` to evaluate your trained agents against a random opponent or watch agents play out a game with visual board states.

```bash
python test.py
```
---


## 📊 Expected Results
When benchmarked on Tic-Tac-Toe, the agents exhibit distinct learning curves:
- **Minimax:** Achieves a 100% non-loss rate (wins or draws) immediately, serving as the ceiling for performance.
- **AlphaZero:** Starts at random-chance performance (~33% win rate) but rapidly converges to near-perfect play within 30–50 epochs of self-play.
- **MuZero:** Initially struggles as it must simultaneously learn the environment's transition dynamics and the optimal policy. However, after sufficient training, it matches AlphaZero's performance, proving its ability to plan effectively in latent space without explicit game rules.
--- 
## 📄 License
Distributed under the MIT License. See LICENSE for more information.
