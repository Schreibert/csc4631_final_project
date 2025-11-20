# Balatro Poker Simulator for Reinforcement Learning

A high-performance C++ poker game simulator with Python bindings, designed for reinforcement learning research. Implements a simplified single-blind version of Balatro-style poker.

## Project Overview

This simulator provides:
- **C++ core** with deterministic gameplay
- **Python bindings** via pybind11 for easy RL integration
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
cd csc4631_final_project
```

2. **Build C++ core and bindings**
```bash
# Windows
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target _balatro_core

# Linux/Mac
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=ON
cmake --build . --config Release
```

The Python module (`.pyd` or `.so`) is **automatically copied** to `python/balatro_env/`

3. **Run tests**
```bash
# C++ tests (41 tests)
cd build\cpp\tests
ctest -C Release

# Python tests (7 tests)
cd python && python tests/test_determinism.py
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

## Limitations (v0)

The following Balatro features are **not** implemented in v0:
- Jokers and their effects
- Card editions (Foil, Holographic, etc.)
- Tags and skip blind mechanics
- Shop phases and economy
- Multi-blind progression (Antes 1-8)
- Consumables (Tarot/Planet/Spectral cards)

These may be added in future versions after RL research phase.
