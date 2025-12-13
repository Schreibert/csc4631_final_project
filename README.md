# Balatro RL Simulator

## Contributors
- **Tyler Schreiber**
- **Alec Nartatez**

## Project Overview

This project implements a high-performance Balatro simulator designed for reinforcement learning research. The simulator features a two-layer architecture: a C++ core providing fast game logic, and a Python wrapper for RL work.

The goal is to train RL agents to play a simplified single-blind balatro where agents must score chips by playing poker hands from an 8-card hand. Agents have 4 plays and 3 discards per episode to reach a target score (default 300 chips). The scoring formula follows Balatro's system: `(base chips + card score) × base mult`.

---

## Compilation Instructions

### Prerequisites
- CMake 3.14+
- C++17 compiler (MSVC 2022, GCC 7+, or Clang 6+)
- Python 3.8+

### Building C++

```bash
# From project root
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target _balatro_core
```

The compiled Python module (`_balatro_core.pyd` on Windows) is automatically copied to `python/balatro_env/` after building.

### Installing Python Package

```bash
cd python
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

---

## Running the Code

### Running C++ Tests

```bash
# From project root (Windows)
cmake --build build --config Release --target all_tests
ctest --test-dir build/cpp/tests -C Release --verbose
```

### Running Python Tests

```bash
cd python
python -m pytest tests/ -v
```

---

## Running the Agents

All agents are located in `python/examples/`. Run from that directory or use full paths.

### Random Strategy Agent

Baseline agent that randomly selects from 5 high-level strategies.

```bash
cd python/examples

# Run 100 episodes with default settings
python random_strategy_agent.py
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `run` | `run` for batch stats, `eval` for visualized single episode |
| `--episodes` | `100` | Number of episodes to run |
| `--seed` | `42` | Starting random seed |
| `--target-score` | `300` | Chips needed to win |
| `--visualize` | off | Show card-level decision details |
| `--viz-mode` | `full` | `full` for detailed view, `compact` for one-line summaries |
| `--verbose` | off | Print raw state information |
| `--reward-config` | default | Path to custom reward YAML file |

---

### Basic Heuristic Agent

Rule-based agent using greedy play with intelligent discard decisions.

```bash
cd python/examples

# Run 100 episodes with default settings
python basic_heuristic.py
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `run` | `run` for batch stats, `eval` for visualized evaluation |
| `--episodes` | `100` | Number of episodes to run |
| `--seed` | `42` | Starting random seed |
| `--target-score` | `300` | Chips needed to win |
| `--visualize` | off | Show card-level decision details |
| `--viz-mode` | `full` | `full` for detailed view, `compact` for one-line summaries |
| `--verbose` | off | Print raw state information |
| `--reward-config` | default | Path to custom reward YAML file |

**Typical Performance:** ~65-75% win rate at target=300

---

### Hierarchical Q-Learning Agent

Two-level Q-learning agent that learns when to play vs discard, then which strategy to use.

```bash
cd python/examples

# Train for default episodes (from config)
python hierarchical_q_learning_agent.py --mode train

# Evaluate a trained model
python hierarchical_q_learning_agent.py --mode eval --load-model ../models/hql_final.pkl
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train` to learn, `eval` to test |
| `--config` | `q_learning_config.yaml` | Path to hyperparameter config file |
| `--load-model` | none | Path to saved Q-table checkpoint |
| `--episodes` | from config | Override number of episodes |
| `--lr` | from config | Override learning rate |
| `--epsilon-start` | from config | Override initial exploration rate |
| `--seed` | `42` | Random seed for reproducibility |
| `--target-score` | `300` | Chips needed to win |
| `--plot` | off | Show training curves after completion |
| `--visualize` | off | Show card-level decision details |
| `--viz-mode` | `full` | `full` for detailed view, `compact` for one-line summaries |
---

## Results and Outputs

| Output Type | Location |
|-------------|----------|
| Trained Q-tables | `models/*.pkl` |
| Training curves | Displayed via matplotlib (use `--plot`) |
| Episode statistics | Console output (win rate, avg reward, avg steps) |
| Decision visualization | Console output (use `--visualize`) |

---

## Directory Structure

```
.
├── CMakeLists.txt              - Root CMake configuration
├── CLAUDE.md                   - Development guidelines for Claude Code
├── README.md                   - Project documentation
│
├── cpp/                        - C++ simulation core (C++17)
│   ├── CMakeLists.txt          - Build configuration for core library
│   ├── include/balatro/        - Public API headers
│   │   ├── card.hpp            - Card encoding (0-51), Deck class
│   │   ├── hand_eval.hpp       - Poker hand evaluation, HandType enum
│   │   ├── scoring.hpp         - Balatro scoring formula
│   │   ├── blind_state.hpp     - Episode state, Observation/Action structs
│   │   └── simulator.hpp       - Top-level simulator for Python bindings
│   ├── src/                    - Core implementation
│   │   ├── card.cpp            - Deck management, card dealing
│   │   ├── hand_eval.cpp       - Hand evaluation with straight/flush detection
│   │   ├── scoring.cpp         - Score calculation with base values
│   │   ├── blind_state.cpp     - Game loop, action validation, RL helpers
│   │   └── simulator.cpp       - Batch action execution
│   └── tests/                  - Google Test unit tests (44 tests, all passing)
│
├── bindings/                   - pybind11 Python bindings
│   ├── CMakeLists.txt          - pybind11 module configuration
│   └── pybind/bindings.cpp     - Exposes Simulator, Observation, Action classes
│
├── python/
│   ├── balatro_env/            - Gymnasium RL environment
│   │   ├── __init__.py         - Package exports BalatroBatchedSimEnv
│   │   ├── env.py              - Main environment with Dict spaces
│   │   ├── reward_shaper.py    - YAML-configurable reward shaping
│   │   └── rewards_config.yaml - Default reward configuration
│   │
│   ├── examples/               - Agent implementations
│   │   ├── random_strategy_agent.py        - Random baseline
│   │   ├── basic_heuristic.py              - Rule-based agent
│   │   ├── hierarchical_q_learning_agent.py - Two-level Q-learning
│   │   ├── strategy_action_encoder.py      - 5 strategies to card masks
│   │   ├── q_learning_utils.py             - Checkpointing and plotting
│   │   └── agent_visualizer.py             - Decision visualization
│   │
│   └── tests/                  - Python integration tests (23 tests, all passing)
│       ├── test_enhanced_features.py   - Best hand and score prediction tests
│       └── test_reward_shaper.py       - Reward configuration tests
│
├── models/                     - Saved Q-tables and checkpoints
├── docs/                       - Project proposal and architecture docs
└── tools/seed_replay/          - Determinism validation utility
```
