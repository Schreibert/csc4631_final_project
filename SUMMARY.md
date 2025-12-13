# Balatro-Style Poker RL Simulator

## Contributors
- **Tyler Schreiber**
- **Alec Nartatez**

## Project Overview

This project implements a high-performance Balatro-style poker game simulator designed for reinforcement learning research. The simulator features a two-layer architecture: a C++ core providing ultra-fast, deterministic poker game logic, and a Python wrapper offering a Gymnasium-compatible RL interface.

The goal is to train RL agents to play a simplified single-blind poker game where agents must score chips by playing poker hands from an 8-card hand. Agents have 4 plays and 3 discards per episode to reach a target score (default 300 chips). The scoring formula follows Balatro's system: `(base_chips + rank_sum) × base_mult`.

Key features include:
- Deterministic gameplay (same seed produces identical trajectories)
- Configurable reward shaping via YAML
- Multiple baseline agents (random, heuristic, Q-learning)
- Real-time decision visualization

---

## Directory Structure

### Root Files
| File | Description |
|------|-------------|
| `CMakeLists.txt` | Root CMake configuration for building C++ core and Python bindings |
| `CLAUDE.md` | Development guidelines for Claude Code AI assistant |
| `README.md` | User-facing documentation with build instructions and game rules |
| `SUMMARY.md` | This file - comprehensive project documentation |

### C++ Core (`cpp/`)
High-performance poker simulation engine written in C++17.

| Path | Description |
|------|-------------|
| `cpp/CMakeLists.txt` | CMake build configuration for core library and tests |
| `cpp/include/balatro/` | Public API headers defining the simulation interface |
| `cpp/include/balatro/card.hpp` | Card encoding (0-51), Deck class with deterministic shuffling |
| `cpp/include/balatro/hand_eval.hpp` | Poker hand evaluation algorithms, HandType enum (9 types) |
| `cpp/include/balatro/scoring.hpp` | Balatro scoring formula implementation |
| `cpp/include/balatro/blind_state.hpp` | Episode state management, Observation/Action structs |
| `cpp/include/balatro/simulator.hpp` | Top-level simulator class for Python bindings |
| `cpp/src/` | Implementation files for all headers |
| `cpp/src/card.cpp` | Deck management, card dealing, replacement |
| `cpp/src/hand_eval.cpp` | Hand evaluation logic including straight/flush detection |
| `cpp/src/scoring.cpp` | Score calculation with base values table |
| `cpp/src/blind_state.cpp` | Game loop, action validation, RL helper methods |
| `cpp/src/simulator.cpp` | Batch action execution, Python interface |
| `cpp/tests/` | Google Test unit tests (41 tests) |

### Python Bindings (`bindings/`)
pybind11 wrapper exposing C++ simulator to Python.

| Path | Description |
|------|-------------|
| `bindings/CMakeLists.txt` | CMake configuration for pybind11 module |
| `bindings/pybind/bindings.cpp` | Python bindings exposing Simulator, Observation, Action classes |

### Python Environment (`python/balatro_env/`)
Gymnasium-compatible RL environment wrapper.

| Path | Description |
|------|-------------|
| `python/balatro_env/__init__.py` | Package initialization, exports BalatroBatchedSimEnv |
| `python/balatro_env/env.py` | Main Gymnasium environment class with Dict observation/action spaces |
| `python/balatro_env/reward_shaper.py` | YAML-configurable reward shaping system |
| `python/balatro_env/rewards_config.yaml` | Default reward configuration with inline documentation |

### Example Agents (`python/examples/`)
Baseline agents and training utilities.

| Path | Description |
|------|-------------|
| `python/examples/random_strategy_agent.py` | Random baseline using strategy-based action space |
| `python/examples/basic_heuristic.py` | Rule-based agent with greedy play and discard thresholds |
| `python/examples/hierarchical_q_learning_agent.py` | Two-level Q-learning (policy + strategy selection) |
| `python/examples/strategy_action_encoder.py` | Converts 5 high-level strategies to card mask actions |
| `python/examples/q_learning_utils.py` | Checkpointing, plotting, and baseline comparison utilities |
| `python/examples/agent_visualizer.py` | Decision visualization tool (full and compact modes) |

### Python Tests (`python/tests/`)
Integration tests for Python environment.

| Path | Description |
|------|-------------|
| `python/tests/test_determinism.py` | Verifies identical seeds produce identical trajectories |
| `python/tests/test_enhanced_features.py` | Tests for best hand analysis and score prediction |
| `python/tests/test_reward_shaper.py` | Tests for reward configuration loading and shaping |

### Other Directories
| Path | Description |
|------|-------------|
| `build/` | CMake build artifacts (generated, not committed) |
| `models/` | Saved agent checkpoints and Q-tables |
| `docs/` | Project proposal and architecture documents |
| `tools/seed_replay/` | Determinism validation utility |

---

## Compilation Instructions

### Prerequisites
- CMake 3.14+
- C++17 compiler (MSVC 2022, GCC 7+, or Clang 6+)
- Python 3.8+
- Git

### Building on Windows (Visual Studio)

```bash
# From project root
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target _balatro_core
```

### Building on Linux/Mac

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=ON
cmake --build . --config Release
```

The compiled Python module (`_balatro_core.pyd` on Windows, `_balatro_core.so` on Linux) is automatically copied to `python/balatro_env/` after building.

### Installing Python Package

```bash
cd python
pip install -e .
```

Required Python packages (installed automatically):
- `numpy >= 1.21.0`
- `gymnasium >= 0.28.0`

Optional packages for agents:
- `pyyaml` - Reward configuration loading
- `matplotlib` - Training curve visualization
- `torch` - Deep Q-Network agents

---

## Running the Code

### Running C++ Tests

```bash
# From project root (Windows)
ctest --test-dir build/cpp/tests -C Release --verbose

# Linux/Mac
cd build && ctest --verbose
```

All 41 tests should pass, covering card encoding, hand evaluation, scoring, and determinism.

### Running Python Tests

```bash
cd python
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_determinism.py -v
```

### Training an Agent

```bash
# Train hierarchical Q-learning agent
cd python/examples
python hierarchical_q_learning_agent.py --mode train --episodes 10000

# With custom config
python hierarchical_q_learning_agent.py --mode train --config q_learning_config.yaml
```

### Evaluating an Agent

```bash
# Evaluate with visualization
python hierarchical_q_learning_agent.py --mode eval --load-model models/q_table_ep010000.pkl --visualize --episodes 10

# Compare agents
python basic_heuristic.py --episodes 100 --visualize --viz-mode compact
```

### Running Random Baseline

```bash
python random_strategy_agent.py --episodes 100 --seed 42
```

---

## Results and Outputs

### Model Checkpoints
Saved to `python/examples/models/` (or `models/` at project root):
- `q_table_ep{N}.pkl` - Q-table checkpoints at episode N
- `metadata.json` - Training hyperparameters and statistics

### Training Curves
Generated by `q_learning_utils.py`:
- `training_curves.png` - Win rate, average reward, epsilon over episodes

### Console Output
Agents print episode statistics:
```
Episode 100/1000 | Win Rate: 45.0% | Avg Reward: 250.3 | Epsilon: 0.60
```

### Visualization Output
The `agent_visualizer.py` provides two modes:

**Compact mode** (one line per step):
```
Step  1: PLAY [K♠, K♦, K♥, A♣, A♦] -> +368 chips (total: 368) | WIN!
```

**Full mode** (detailed breakdown):
```
======================================================================
STEP 1
======================================================================
Resources: 4 plays, 3 discards remaining
Progress:  0/300 chips (0.0%)

Hand dealt: [K♠, K♦, K♥, A♣, A♦, 7♠, 3♦, 2♣]
Best possible: Full House (368 chips)

>>> ACTION: PLAY 5 card(s)
    Selected: [K♠, K♦, K♥, A♣, A♦]

<<< RESULT: Full House
    Base: 40 chips × 4
    Rank sum: 10+10+10+11+11 = 52
    Total: (40 + 52) × 4 = 368 chips

Reward: 1.25 | WIN!
```

### Interpreting Results

| Metric | Good Performance | Notes |
|--------|------------------|-------|
| Win Rate | >50% | Random baseline ~10%, heuristic ~70% |
| Avg Reward | >200 | Higher is better, max ~1500 with bonuses |
| Q-table States | 800-2000 | Larger = more exploration |
| Training Episodes | 5000-10000 | Until win rate plateaus |

---

## Game Rules (v0 Simplified Ruleset)

- **Episode:** 4 plays, 3 discards, 8-card hand, 52-card deck
- **Objective:** Score ≥ target chips (default 300) before exhausting plays
- **Scoring:** `(base_chips + rank_sum) × base_mult`
- **Win:** Reach target at any moment
- **Loss:** 0 plays remaining and below target

### Hand Base Values
| Hand | Base Chips | Multiplier |
|------|-----------|------------|
| High Card | 5 | 1 |
| Pair | 10 | 2 |
| Two Pair | 20 | 2 |
| Three of a Kind | 30 | 3 |
| Straight | 30 | 4 |
| Flush | 35 | 4 |
| Full House | 40 | 4 |
| Four of a Kind | 60 | 7 |
| Straight Flush | 100 | 8 |
