# Balatro-Style Poker RL Simulator

## Contributors
- **Tyler Schreiber**
- **Alec Nartatez**

## Project Overview

This project implements a high-performance Balatro-style poker game simulator designed for reinforcement learning research. The simulator features a two-layer architecture: a C++ core providing fast poker game logic, and a Python wrapper for RL work.

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
pip install -r requirements.txt
```

---

## Running the Code

### Running C++ Tests

```bash
# From project root (Windows)
ctest --test-dir build/cpp/tests -C Release --verbose
```

### Running Python Tests

```bash
cd python
python -m pytest tests/ -v
```

---

## Directory Structure

### Root Files
| File | Description |
|------|-------------|
| `CMakeLists.txt` | Root CMake configuration for building C++ core and Python bindings |
| `CLAUDE.md` | Development guidelines for Claude Code AI assistant |
| `README.md` | comprehensive project documentation |

### C++ Core (`cpp/`)
balatro simulation engine written in C++17.

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
RL environment wrapper.

| Path | Description |
|------|-------------|
| `python/balatro_env/__init__.py` | Package initialization, exports BalatroBatchedSimEnv |
| `python/balatro_env/env.py` | Main Gymnasium environment class with Dict observation/action spaces |
| `python/balatro_env/reward_shaper.py` | YAML-configurable reward shaping system |
| `python/balatro_env/rewards_config.yaml` | Default reward configuration |

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
| `python/tests/test_enhanced_features.py` | Tests for best hand analysis and score prediction |
| `python/tests/test_reward_shaper.py` | Tests for reward configuration loading and shaping |

### Other Directories
| Path | Description |
|------|-------------|
| `build/` | CMake build artifacts (generated, not committed) |
| `models/` | Saved agent checkpoints and Q-tables |
| `docs/` | Project proposal and architecture documents |
| `tools/seed_replay/` | Determinism validation utility |
