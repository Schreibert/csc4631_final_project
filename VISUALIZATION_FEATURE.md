# Agent Decision Visualization Feature

## Summary

Added comprehensive visualization capabilities to help understand agent decision-making in the Balatro poker environment. The visualization shows:

- **Hand dealt**: All cards with ranks and suits (e.g., `[AS, KH, QD, JC, 10S, 9H, 8D, 7C]`)
- **Actions chosen**: Whether the agent played or discarded, and which specific cards
- **Results**: Points scored, updated chip totals, and episode outcomes
- **Rewards**: Both raw chip gains and shaped rewards

## Files Added

1. **`python/examples/agent_visualizer.py`** - Core visualization module
   - `AgentVisualizer` class with two display modes
   - Auto-detects platform for Unicode/ASCII suit symbols
   - Handles full and compact visualization modes

2. **`python/examples/VISUALIZATION_README.md`** - User documentation
   - Usage examples for both agents
   - Explanation of visualization modes
   - Card notation reference
   - Tips and use cases

## Files Modified

1. **`python/examples/random_agent.py`**
   - Added `--visualize` flag to enable visualization
   - Added `--viz-mode` argument (choices: `full`, `compact`)
   - Integrated with `AgentVisualizer` class

2. **`python/examples/q_learning_agent.py`**
   - Added `--visualize` flag for evaluation mode
   - Added `--viz-mode` argument (choices: `full`, `compact`)
   - Updated `evaluate()` function to support visualization

## Usage Examples

### Random Agent

```bash
# Compact one-line summaries
python random_agent.py --num-episodes 1 --visualize --viz-mode compact

# Full detailed view
python random_agent.py --num-episodes 1 --visualize --viz-mode full --seed 123
```

### Q-Learning Agent

```bash
# Evaluate with visualization (compact mode)
python q_learning_agent.py --mode eval --load-model models/q_table_ep010000.pkl --visualize --viz-mode compact

# Evaluate with full details
python q_learning_agent.py --mode eval --load-model models/q_table_ep010000.pkl --visualize --viz-mode full --episodes 3
```

## Visualization Modes

### Compact Mode
One-line summaries showing hand, action, and result:
```
Step  1: PLAY [6C, 6C, 6C, 6C]               -> +  34 chips (total:   34) | R=   0.1
Step  2: PLAY [AH, AH, AH]                   -> +  29 chips (total:   63) | R=   0.1
Step  3: DISC [AS, AS, AS, AS]               -> +   0 chips (total:   63) | R=  -0.0
```

### Full Mode
Detailed step-by-step breakdown:
```
======================================================================
STEP 1
======================================================================
Resources: 4 plays, 3 discards remaining
Progress:  0/300 chips (0.0%)

Hand dealt: [4S, 4S, 4S, 4S, 4S, 4S, 4S, 4S]
Best possible: Pair (108 chips)

>>> ACTION: PLAY 3 card(s)
    Selected: [4S, 4S, 4S]

<<< RESULT:
    Chips scored: +25
    Total chips: 0 -> 25

Reward: 0.1 (raw: 25.0)
```

## Features

- **Platform-aware**: Auto-detects Windows and uses ASCII symbols (C, D, H, S) instead of Unicode
- **Episode summaries**: Shows win/loss, steps taken, final chips, and total reward
- **Best hand analysis**: Displays the strongest possible hand and its value (when available)
- **Reward comparison**: Shows both shaped rewards and raw chip gains
- **Clear formatting**: Uses separators and structured output for readability

## Benefits

1. **Debugging**: Quickly identify why agents fail or succeed
2. **Strategy analysis**: Understand decision patterns and learning progress
3. **Comparison**: Run same seed with different agents to compare strategies
4. **Education**: Learn poker hand values and optimal play patterns
5. **Reward tuning**: See how reward shaping affects agent behavior

## Technical Details

- Handles Unicode encoding issues on Windows automatically
- Supports both training and evaluation modes
- Minimal performance impact (only when enabled)
- Compatible with existing agent implementations
- No changes to core environment or training logic
