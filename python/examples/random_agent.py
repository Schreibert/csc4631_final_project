#!/usr/bin/env python3
"""
Random agent with direct card control.

Demonstrates the new action interface where the agent selects specific cards
to play or discard instead of using predefined action codes.
"""

import sys
import os
import random

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'balatro_env'))
import _balatro_core as core


# Card names for pretty printing
RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUIT_SYMBOLS = ['♣', '♦', '♥', '♠']


def print_observation(obs, step):
    """Pretty print current observation"""
    print(f"\n--- Step {step} ---")
    print(f"State: plays={obs.plays_left}, discards={obs.discards_left}, " +
          f"chips={obs.chips}/{obs.chips + obs.chips_to_target}, deck={obs.deck_remaining}")
    print("Hand:", end=" ")
    for i in range(core.HAND_SIZE):
        rank = obs.card_ranks[i]
        suit = obs.card_suits[i]
        print(f"[{i}]{RANK_NAMES[rank]}{SUIT_SYMBOLS[suit]}", end=" ")
    print()


def generate_random_valid_action(obs):
    """Generate a random valid action"""
    # Randomly choose to play or discard
    if obs.plays_left > 0 and obs.discards_left > 0:
        action_type = random.choice([core.PLAY, core.DISCARD])
    elif obs.plays_left > 0:
        action_type = core.PLAY
    elif obs.discards_left > 0:
        action_type = core.DISCARD
    else:
        return None  # No valid actions

    # Generate random card selection
    if action_type == core.PLAY:
        # Play 1-5 cards
        num_cards = random.randint(1, 5)
    else:
        # Discard 1-8 cards
        num_cards = random.randint(1, core.HAND_SIZE)

    # Select random card indices
    card_indices = random.sample(range(core.HAND_SIZE), num_cards)
    card_mask = [i in card_indices for i in range(core.HAND_SIZE)]

    # Create action
    action = core.Action()
    action.type = action_type
    action.card_mask = card_mask

    return action


def run_episode(target_score, seed, verbose=False):
    """Run one episode with random card selections"""
    sim = core.Simulator()
    obs = sim.reset(target_score, seed)

    total_reward = 0
    step = 0

    while step < 100:  # Max 100 steps as safety
        if verbose:
            print_observation(obs, step)

        # Generate random valid action
        action = generate_random_valid_action(obs)

        if action is None:
            if verbose:
                print("No valid actions available!")
            break

        # Validate action (optional but good for debugging)
        validation = sim.validate_action(action)
        if not validation.valid:
            if verbose:
                print(f"Generated invalid action: {validation.error_message}")
            break

        # Display action
        if verbose:
            action_type = "PLAY" if action.type == core.PLAY else "DISCARD"
            selected = [i for i, selected in enumerate(action.card_mask) if selected]
            print(f"Action: {action_type} cards {selected}")

        # Execute action
        result = sim.step_batch([action])
        obs = result.final_obs
        total_reward += result.rewards[0]
        step += 1

        if verbose and result.rewards[0] > 0:
            print(f"Reward: {result.rewards[0]}")

        if result.done:
            if verbose:
                print(f"\n{'WIN!' if result.win else 'LOSS'}")
                print(f"Final chips: {obs.chips}")
                print(f"Steps: {step}")
            return {
                'win': result.win,
                'steps': step,
                'total_reward': total_reward,
                'final_chips': obs.chips,
            }

    return {
        'win': False,
        'steps': step,
        'total_reward': total_reward,
        'final_chips': obs.chips,
    }


def main():
    target_score = 300
    num_episodes = 100
    verbose = False  # Set to True to see detailed output

    print("=" * 60)
    print("Random Agent - Direct Card Control")
    print("=" * 60)
    print(f"Running {num_episodes} episodes with random card selection...")
    print(f"Target score: {target_score}\n")

    if verbose:
        # Run one verbose episode first
        print("\n=== Verbose Episode Example ===")
        result = run_episode(target_score, seed=0, verbose=True)
        print("\n" + "=" * 60 + "\n")

    wins = 0
    total_steps = 0
    total_rewards = 0
    total_chips = 0

    for i in range(num_episodes):
        result = run_episode(target_score, seed=i, verbose=False)

        if result['win']:
            wins += 1

        total_steps += result['steps']
        total_rewards += result['total_reward']
        total_chips += result['final_chips']

        if (i + 1) % 10 == 0:
            print(f"Episode {i+1:3d}: " +
                  f"Win rate = {wins/(i+1):5.1%}, " +
                  f"Avg steps = {total_steps/(i+1):4.1f}, " +
                  f"Avg reward = {total_rewards/(i+1):6.1f}, " +
                  f"Avg chips = {total_chips/(i+1):5.1f}")

    print(f"\n{'=' * 60}")
    print("Final Results:")
    print(f"  Win rate:       {wins/num_episodes:5.1%}")
    print(f"  Average steps:  {total_steps/num_episodes:4.1f}")
    print(f"  Average reward: {total_rewards/num_episodes:6.1f}")
    print(f"  Average chips:  {total_chips/num_episodes:5.1f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
