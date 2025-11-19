# Balatro Poker Simulator for Reinforcement Learning

A high-performance C++ poker game simulator with Python bindings, designed for reinforcement learning research. Implements a simplified single-blind version of Balatro-style poker.

## Project Overview

This simulator provides:
- **Ultra-fast C++ core** with deterministic gameplay
- **Python bindings** via pybind11 for easy RL integration
- **Gym-style API** compatible with standard RL frameworks
- **Batched stepping** for improved performance
- **Deterministic replay** for reproducible experiments

## Architecture

```
cpp/                   - C++ simulation core
├── include/balatro/   - Public API headers
├── src/               - Core implementation
└── tests/             - C++ unit tests (41 tests, all passing)

bindings/              - pybind11 Python bindings
python/                - Python package
├── balatro_env/       - Main package
│   ├── env.py         - Gymnasium environment wrapper
│   └── _balatro_core  - Compiled C++ module
├── examples/          - Example scripts
└── tests/             - Python integration tests

tools/                 - Utilities
└── seed_replay/       - Determinism validation tool
```

## Building from Source

### Prerequisites

- CMake 3.14+
- C++17 compiler (MSVC, GCC, or Clang)
- Python 3.8+
- Git

### Build Steps

1. **Clone and navigate to project**
```bash
cd final_project
```

2. **Build C++ core and bindings**
```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=ON
cmake --build . --config Release
```

3. **Copy Python module to package**
```bash
# Windows
cp build/bindings/Release/_balatro_core.cp311-win_amd64.pyd python/balatro_env/

# Linux/Mac
cp build/bindings/_balatro_core.*.so python/balatro_env/
```

4. **Run tests**
```bash
# C++ tests (41 tests)
cd build && ctest -C Release

# Python tests (7 tests)
cd python && python tests/test_determinism.py
```

## Usage

### Basic Example (C++ Bindings)

```python
import sys
sys.path.insert(0, 'python/balatro_env')
import _balatro_core as core

# Create simulator
sim = core.Simulator()

# Reset episode
obs = sim.reset(target_score=300, seed=12345)
print(f"Initial observation: {list(obs)}")

# Take actions
actions = [core.ACTION_PLAY_BEST, core.ACTION_DISCARD_LOWEST_3]
result = sim.step_batch(actions)

print(f"Rewards: {list(result.rewards)}")
print(f"Done: {result.done}, Win: {result.win}")
```

### With Gymnasium Wrapper

```python
# Requires: pip install gymnasium numpy

from balatro_env import BalatroBatchedSimEnv

env = BalatroBatchedSimEnv(target_score=300)

obs, info = env.reset(seed=42)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print(f"Episode ended. Win: {info['win']}, Chips: {info['chips']}")
        break
```

### Seed Replay Tool

```bash
python tools/seed_replay/replay.py 300 12345 0 2 0 1

# Output:
# Replaying episode:
#   Target Score: 300
#   Seed: 12345
#   Actions: [0, 2, 0, 1]
# ...
# [PASS] Determinism verified
```

## Game Rules (v0 Simplified Ruleset)

### Episode Structure
- **Objective:** Score ≥ target_score chips before exhausting plays
- **Resources:** 4 plays, 3 discards per episode
- **Hand Size:** Always 8 cards
- **Deck:** Standard 52-card deck

### Scoring Formula
```
score = (base_chips + rank_sum) × base_mult
```

Example: Full House K♣K♦K♥A♠A♣
- Base chips: 40
- Rank sum: 10+10+10+11+11 = 52
- Base multiplier: 4
- **Total: (40 + 52) × 4 = 368 chips**

### Poker Hands (Base Values)

| Hand | Base Chips | Base Mult |
|------|-----------|-----------|
| High Card | 5 | 1 |
| Pair | 10 | 2 |
| Two Pair | 20 | 2 |
| Three of a Kind | 30 | 3 |
| Straight | 30 | 4 |
| Flush | 35 | 4 |
| Full House | 40 | 4 |
| Four of a Kind | 60 | 7 |
| Straight Flush | 100 | 8 |

### Observation Space (8-dimensional)

1. `plays_left` (0-4)
2. `discards_left` (0-3)
3. `chips_to_target` (0-∞)
4. `has_pair` (0/1)
5. `has_trips` (0/1)
6. `straight_potential` (0/1)
7. `flush_potential` (0/1)
8. `max_rank_bucket` (0-5)

## Test Coverage

### C++ Tests (41 passing)
- Card encoding and deck shuffling
- Hand evaluation (all 9 poker hands)
- Golden scoring tests
- State management
- Deterministic trajectories
- Episode termination
- Batch vs sequential equivalence

### Python Tests (7 passing)
- Deterministic reset
- Deterministic trajectory
- Batch vs sequential equivalence
- Observation size validation
- Action space validation
- Episode termination (win/loss)
- Padding after done

## Performance

The batched step interface significantly reduces Python/C++ call overhead:

```python
# Slow: 4 separate calls
for action in actions:
    result = sim.step_batch([action])

# Fast: Single batched call
result = sim.step_batch(actions)
```

## Development

### Running All Tests

```bash
# C++ tests
cd build && ctest -C Release --output-on-failure

# Python tests
cd python && python tests/test_determinism.py

# Seed replay validation
python tools/seed_replay/replay.py 300 12345 0 2 0
```

### Example Scripts

```bash
# Random agent baseline
python python/examples/random_agent.py
# Output: ~47% win rate on target_score=300
```

## Limitations (v0)

The following Balatro features are **not** implemented in v0:
- Jokers and their effects
- Card editions (Foil, Holographic, etc.)
- Tags and skip blind mechanics
- Shop phases and economy
- Multi-blind progression (Antes 1-8)
- Consumables (Tarot/Planet/Spectral cards)

These may be added in future versions after RL research phase.