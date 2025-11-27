#!/usr/bin/env python3
"""
Script to train behavior cloning from collected trajectory data.

This script:
1. Loads all Player 1 (leader) trajectories from a directory
2. Trains a BC agent to clone Player 1's behavior
3. Trains Player 2 (follower) to work with the BC agent as a fixed partner
"""

import argparse
import os
import glob
import numpy as np
import gym
from pathlib import Path

from pantheonrl.algos.bc import BC
from pantheonrl.common import trajsaver
from pantheonrl.common.multiagentenv import SimultaneousEnv
from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent
from pantheonrl.algos.bc import BCShell, reconstruct_policy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Note: Layout name translation: "asymmetric_advantages" -> "unident_s"


def load_and_combine_trajectories(data_dir, pattern="p1*.npy", layout_name="unident_s", translate_layout=True):
    """
    Load all trajectory files matching the pattern and combine Player 1's transitions.
    
    Args:
        data_dir: Directory containing trajectory files
        pattern: Glob pattern to match trajectory files (default: "p1*.npy")
        layout_name: Layout name for the environment
        translate_layout: Whether to translate layout names (e.g., "asymmetric_advantages" -> "unident_s")
    
    Returns:
        Combined TransitionsMinimal object containing all Player 1's transitions
    """
    # Translate layout name if needed
    if translate_layout:
        from overcookedgym.overcooked_utils import NAME_TRANSLATION
        actual_layout = NAME_TRANSLATION.get(layout_name, layout_name)
        if actual_layout != layout_name:
            print(f"Layout name translated: '{layout_name}' -> '{actual_layout}'")
            layout_name = actual_layout
    
    # Create environment to get observation/action spaces
    env = gym.make("OvercookedMultiEnv-v0", layout_name=layout_name)
    
    # Find all matching files
    pattern_path = os.path.join(data_dir, pattern)
    traj_files = sorted(glob.glob(pattern_path))
    
    if not traj_files:
        raise ValueError(f"No trajectory files found matching pattern: {pattern_path}")
    
    print(f"Found {len(traj_files)} trajectory files matching pattern '{pattern}'")
    
    # Determine transition class
    TransitionsClass = trajsaver.SimultaneousTransitions
    
    # Load and combine all trajectories
    all_ego_obs = []
    all_ego_acts = []
    
    for i, traj_file in enumerate(traj_files):
        try:
            print(f"Loading {i+1}/{len(traj_files)}: {os.path.basename(traj_file)}")
            transitions = TransitionsClass.read_transition(
                traj_file,
                env.observation_space,
                env.action_space
            )
            
            # Extract Player 1's (ego's) transitions
            ego_transitions = transitions.get_ego_transitions()
            
            all_ego_obs.append(ego_transitions.obs)
            all_ego_acts.append(ego_transitions.acts)
            
            print(f"  - Loaded {len(ego_transitions.obs)} transitions")
            
        except Exception as e:
            print(f"  - ERROR loading {traj_file}: {e}")
            continue
    
    if not all_ego_obs:
        raise ValueError("No valid trajectories could be loaded!")
    
    # Combine all transitions
    combined_obs = np.concatenate(all_ego_obs, axis=0)
    combined_acts = np.concatenate(all_ego_acts, axis=0)
    
    print(f"\nTotal combined transitions: {len(combined_obs)}")
    
    # Create combined TransitionsMinimal object
    combined_transitions = trajsaver.TransitionsMinimal(
        obs=combined_obs,
        acts=combined_acts
    )
    
    return combined_transitions, env


def train_bc_agent(expert_data, env, save_path, n_epochs=100, l2_weight=0.0, device="auto"):
    """
    Train a BC agent from expert data.
    
    Args:
        expert_data: TransitionsMinimal object containing expert demonstrations
        env: Gym environment
        save_path: Path to save the trained BC policy
        n_epochs: Number of training epochs
        l2_weight: L2 regularization weight
        device: Device to run training on
    
    Returns:
        Trained BC agent
    """
    print(f"\n{'='*60}")
    print("Training Behavior Cloning Agent")
    print(f"{'='*60}")
    print(f"Expert data size: {len(expert_data.obs)} transitions")
    print(f"Training epochs: {n_epochs}")
    print(f"L2 weight: {l2_weight}")
    
    # Create BC trainer
    bc_agent = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        expert_data=expert_data,
        l2_weight=l2_weight,
        device=device
    )
    
    # Train
    print("\nStarting training...")
    bc_agent.train(n_epochs=n_epochs)
    
    # Save policy
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        bc_agent.save_policy(save_path)
        print(f"\nBC policy saved to: {save_path}")
    
    return bc_agent


def train_follower_with_bc_partner(
    env,
    bc_policy_path,
    layout_name,
    total_timesteps=500000,
    save_path=None,
    device="auto",
    seed=None
):
    """
    Train Player 2 (follower) to work with BC-cloned Player 1 (leader) as a fixed partner.
    
    Args:
        env: Gym environment (should be wrapped with recorder if needed)
        bc_policy_path: Path to the saved BC policy for Player 1
        layout_name: Layout name for the environment
        total_timesteps: Total timesteps for training
        save_path: Path to save the trained follower agent
        device: Device to run training on
        seed: Random seed
    """
    print(f"\n{'='*60}")
    print("Training Follower Agent with BC Leader Partner")
    print(f"{'='*60}")
    
    # Load BC policy for Player 1
    print(f"Loading BC policy from: {bc_policy_path}")
    bc_policy = reconstruct_policy(bc_policy_path, device=device)
    
    # Wrap BC policy as a static agent (fixed partner)
    bc_agent = StaticPolicyAgent(bc_policy)
    
    # Add BC agent as partner
    env.add_partner_agent(bc_agent)
    
    print("BC agent added as fixed partner (Player 1 / Leader)")
    
    # Create follower agent (Player 2) - using PPO
    follower_config = {
        'env': env,
        'device': device,
        'verbose': 1
    }
    
    if seed is not None:
        follower_config['seed'] = seed
    
    follower_agent = OnPolicyAgent(PPO(policy='MlpPolicy', **follower_config))
    
    print(f"Follower agent (Player 2) created")
    print(f"Training for {total_timesteps} timesteps...")
    
    # Train follower
    follower_agent.learn(total_timesteps=total_timesteps)
    
    # Save follower agent
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        follower_agent.save(save_path)
        print(f"\nFollower agent saved to: {save_path}")
    
    return follower_agent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
Train Behavior Cloning from collected trajectory data and train a follower agent.

This script:
1. Loads all Player 1 (leader) trajectories from a directory
2. Trains a BC agent to clone Player 1's behavior
3. (Optional) Trains Player 2 (follower) to work with the BC agent

Example usage:
  # Only train BC agent
  python train_bc_from_collected_data.py p1nate_adrian \\
      --layout-name unident_s \\
      --bc-epochs 100 \\
      --bc-save models/bc_leader.pt

  # Train BC and then train follower
  python train_bc_from_collected_data.py p1nate_adrian \\
      --layout-name unident_s \\
      --bc-epochs 100 \\
      --bc-save models/bc_leader.pt \\
      --train-follower \\
      --follower-timesteps 500000 \\
      --follower-save models/follower_with_bc_leader.zip
        ''')
    
    parser.add_argument('data_dir',
                        help='Directory containing trajectory files (.npy)')
    
    parser.add_argument('--layout-name',
                        default='asymmetric_advantages',
                        help='Overcooked layout name (will be translated if needed. Default: asymmetric_advantages -> unident_s)')
    
    parser.add_argument('--pattern',
                        default='p1*.npy',
                        help='Pattern to match trajectory files (default: p1*.npy)')
    
    # BC training arguments
    parser.add_argument('--bc-epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for BC training (default: 100)')
    
    parser.add_argument('--bc-l2',
                        type=float,
                        default=0.0,
                        help='L2 regularization weight for BC (default: 0.0)')
    
    parser.add_argument('--bc-save',
                        default='models/bc_leader.pt',
                        help='Path to save BC policy (default: models/bc_leader.pt)')
    
    # Follower training arguments
    parser.add_argument('--train-follower',
                        action='store_true',
                        help='Train follower agent after BC training')
    
    parser.add_argument('--follower-timesteps',
                        type=int,
                        default=500000,
                        help='Total timesteps for follower training (default: 500000)')
    
    parser.add_argument('--follower-save',
                        default='models/follower_with_bc_leader.zip',
                        help='Path to save follower agent (default: models/follower_with_bc_leader.zip)')
    
    parser.add_argument('--device',
                        default='auto',
                        help='Device to run on (default: auto)')
    
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Step 1: Load and combine trajectories
    print(f"{'='*60}")
    print("Step 1: Loading and Combining Trajectories")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Layout: {args.layout_name}")
    
    expert_data, env = load_and_combine_trajectories(
        args.data_dir,
        pattern=args.pattern,
        layout_name=args.layout_name,
        translate_layout=True
    )
    
    # Step 2: Train BC agent
    print(f"\n{'='*60}")
    print("Step 2: Training BC Agent")
    print(f"{'='*60}")
    
    bc_agent = train_bc_agent(
        expert_data=expert_data,
        env=env,
        save_path=args.bc_save,
        n_epochs=args.bc_epochs,
        l2_weight=args.bc_l2,
        device=args.device
    )
    
    # Step 3: Train follower (optional)
    if args.train_follower:
        print(f"\n{'='*60}")
        print("Step 3: Training Follower Agent")
        print(f"{'='*60}")
        
        # Translate layout name for follower training
        from overcookedgym.overcooked_utils import NAME_TRANSLATION
        actual_layout = NAME_TRANSLATION.get(args.layout_name, args.layout_name)
        
        # Create fresh environment for follower training
        follower_env = gym.make("OvercookedMultiEnv-v0", layout_name=actual_layout)
        
        train_follower_with_bc_partner(
            env=follower_env,
            bc_policy_path=args.bc_save,
            layout_name=args.layout_name,
            total_timesteps=args.follower_timesteps,
            save_path=args.follower_save,
            device=args.device,
            seed=args.seed
        )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"BC policy saved to: {args.bc_save}")
    if args.train_follower:
        print(f"Follower agent saved to: {args.follower_save}")


if __name__ == '__main__':
    main()

