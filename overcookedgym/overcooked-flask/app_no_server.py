import os
import io
import json
import copy
import argparse
import torch
import numpy as np
import gym
import time
import glob

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from stable_baselines3 import PPO
from pantheonrl.common.vae_models import MPPI_agent

from overcookedgym.overcooked_utils import NAME_TRANSLATION
from pantheonrl.common.trajsaver import SimultaneousTransitions

# Global variables for MDP and planner
MDP = None
MLP = None


def action_idx_to_action(action_idx):
    """Convert action index (0-5) to Overcooked Action object."""
    return Action.ALL_ACTIONS[action_idx]


def get_prediction_ppo(s, policy):
    """Get action from PPO policy."""
    s = torch.tensor(s).unsqueeze(0).float()
    actions, states = policy.predict(observation=s)
    return int(actions[0])


def get_prediction_mppi(state_dict, policy, last_p0_action, last_p1_action):
    """Get action from MPPI policy."""
    action, mppi_agent_id = policy.predict(
        state_dict, 
        last_p0_action=last_p0_action, 
        last_p1_action=last_p1_action
    )
    return int(action)


def get_prediction_recorded(policy):
    """Get action from RecordedAgent."""
    action, _ = policy.predict()
    return int(action)


class RecordedAgent:
    """
    Agent that replays actions from a recorded trajectory.
    
    This allows recorded human/agent behavior to be used in run_episode()
    just like any other agent (PPO, MPPI).
    
    Usage:
        agent = RecordedAgent('./recorded_game.json', player_id=0)
        run_episode(agent, other_agent, layout_name)
    """
    
    def __init__(self, trajectory_file, player_id):
        """
        Initialize recorded agent.
        
        Args:
            trajectory_file: Path to JSON trajectory file
            player_id: Which player this agent represents (0 or 1)
        """
        self.trajectory_file = trajectory_file
        self.player_id = player_id
        self.current_step = 0
        
        # Load recorded actions
        with open(trajectory_file, 'r') as f:
            data = json.load(f)
        
        # Extract actions
        if 'traj' in data:
            ep_actions = data['traj'].get('ep_actions', [[]])
        else:
            ep_actions = data.get('ep_actions', [[]])
        
        # Store just this player's actions
        self.actions = [action_pair[player_id] for action_pair in ep_actions[0]]
        
        print(f"  RecordedAgent loaded: {os.path.basename(trajectory_file)}")
        print(f"    Player: {player_id}, Actions: {len(self.actions)}")

    def predict(self, observation=None, state_dict=None, last_p0_action=None, last_p1_action=None):
        """
        Get next action from recording.
        
        This matches the interface of both PPO (predict(observation)) 
        and MPPI (predict(state_dict, last_p0_action, last_p1_action)).
        
        Returns:
            tuple: (action, None) to match PPO interface
        """
        if self.current_step >= len(self.actions):
            # Ran out of recorded actions, return STAY
            action = 4
        else:
            action = self.actions[self.current_step]
        
        self.current_step += 1
        return action, None
    
    def reset(self):
        """Reset to beginning of recording."""
        self.current_step = 0
    
    def __repr__(self):
        return f"RecordedAgent(file={os.path.basename(self.trajectory_file)}, player={self.player_id}, step={self.current_step}/{len(self.actions)})"


def process_state(state_dict, layout_name):
    """Process state dict into feature vectors for both players."""
    global MDP, MLP
    
    def object_from_dict(object_dict):
        return ObjectState(**object_dict)

    def player_from_dict(player_dict):
        held_obj = player_dict.get("held_object")
        if held_obj is not None:
            player_dict["held_object"] = object_from_dict(held_obj)
        return PlayerState(**player_dict)

    def state_from_dict(state_dict):
        state_dict["players"] = [player_from_dict(p) for p in state_dict["players"]]
        objects_raw = state_dict.get("objects", {})
        if isinstance(objects_raw, dict):
            object_list = [object_from_dict(o) for _, o in objects_raw.items()]
        elif isinstance(objects_raw, list):
            object_list = [object_from_dict(o) for o in objects_raw]
        else:
            object_list = []
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        return OvercookedState(**state_dict)

    state = state_from_dict(copy.deepcopy(state_dict))
    
    result = MDP.featurize_state(state, MLP)
    
    return result


def convert_traj_to_simultaneous_transitions(traj_dict, layout_name):
    """Convert trajectory dict to SimultaneousTransitions format."""
    global MDP, MLP

    ego_obs = []
    alt_obs = []
    ego_act = []
    alt_act = []
    flags = []

    for state_list in traj_dict['ep_states']:  # loop over episodes
        ego_obs.append([process_state(state, layout_name)[0] for state in state_list])
        alt_obs.append([process_state(state, layout_name)[1] for state in state_list])

        # check pantheonrl/common/wrappers.py for flag values
        flag = [0 for state in state_list]
        flag[-1] = 1
        flags.append(flag)

    for action_list in traj_dict['ep_actions']:  # loop over episodes
        ego_act.append([joint_action[0] for joint_action in action_list])
        alt_act.append([joint_action[1] for joint_action in action_list])

    ego_obs = np.concatenate(ego_obs, axis=0)
    alt_obs = np.concatenate(alt_obs, axis=0)
    ego_act = np.concatenate(ego_act, axis=0)
    alt_act = np.concatenate(alt_act, axis=0)
    flags = np.concatenate(flags, axis=0)

    return SimultaneousTransitions(
        ego_obs,
        ego_act,
        alt_obs,
        alt_act,
        flags,
    )


def save_trajectory(episode_data, layout_name, save_path, file_name, episode_idx, p0_type, p1_type, algo="mppi"):
    """
    Save trajectory in EXACT web interface format.
    
    Args:
        episode_data: Dict with ep_states, ep_actions, ep_rewards, total_reward
        layout_name: Layout name string
        save_path: Directory to save to
        file_name: Base filename
        episode_idx: Episode number
        p0_type: Agent type for P0 ("ppo", "mppi", "human")
        p1_type: Agent type for P1
        algo: Algorithm name (default "mppi")
    """
    if not save_path:
        print("No save path specified, skipping save.")
        return
    
    # Create timestamp string in web interface format: "MM_DD_YYYY_HH:MM:SS_info"
    datetime_str = time.strftime("%m_%d_%Y_%H:%M:%S")
    traj_id = f"{datetime_str}_ep{episode_idx}"
    
    # Get MDP parameters (standard Overcooked parameters)
    mdp_params = {
        "layout_name": layout_name,
        "num_items_for_soup": 3,
        "rew_shaping_params": None,
        "cook_time": 20,
        "start_order_list": None
    }
    
    # Build trajectory dict in EXACT web interface format
    traj_dict = {
        "traj_id": traj_id,
        "traj": {
            "ep_states": episode_data['ep_states'],
            "ep_rewards": episode_data['ep_rewards'],
            "ep_actions": episode_data['ep_actions'],
            "mdp_params": [mdp_params]  # Note: list with single dict
        },
        "layout_name": layout_name,
        "algo": algo,
        "p0_strat": p0_type,
        "p1_strat": p1_type,
        "file_name": file_name
    }
    
    # Create filename with timestamp for file system
    datetime_file = time.strftime("%Y-%m-%d-%H-%M-%S-")
    filename = os.path.join(save_path, file_name + '_' + datetime_file)
    
    # Save full trajectory JSON
    os.makedirs(save_path, exist_ok=True)
    with open(filename + ".json", 'w') as f:
        json.dump(traj_dict, f, indent=2)
    
    print(f"✓ Saved trajectory to {filename}.json")
    
    # Save transitions minimal
    try:
        simultaneous_transitions = convert_traj_to_simultaneous_transitions(
            traj_dict, layout_name
        )
        simultaneous_transitions.write_transition(filename)
        print(f"✓ Saved transitions to {filename}.npy")
    except Exception as e:
        print(f"Warning: Could not save transitions: {e}")


def run_episode(agent_p0, agent_p1, layout_name, max_steps=400, verbose=True):
    """
    Run a complete episode and collect trajectory data.
    
    Args:
        agent_p0: Policy for player 0 (PPO, MPPI, or None for STAY)
        agent_p1: Policy for player 1 (PPO, MPPI, or None for STAY)
        layout_name: Name of the Overcooked layout
        max_steps: Maximum timesteps per episode
        verbose: Whether to print progress
        
    Returns:
        episode_data: Dictionary containing states, actions, and rewards
    """
    # Initialize episode data
    ep_states = []
    ep_actions = []
    ep_rewards = []
    
    # Get initial state
    state = MDP.get_standard_start_state()
    state_dict = state.to_dict()
    
    # Track last actions for MPPI (start with STAY)
    last_p0_action = 4
    last_p1_action = 4
    
    total_reward = 0
    
    for step in range(max_steps):
        # Store current state
        ep_states.append(copy.deepcopy(state_dict))
        
        # Get actions from agents
        if agent_p0 is None:
            action_p0 = 4  # STAY
        elif isinstance(agent_p0, RecordedAgent):
            action_p0 = get_prediction_recorded(agent_p0)
        elif isinstance(agent_p0, MPPI_agent):
            action_p0 = get_prediction_mppi(state_dict, agent_p0, last_p0_action, last_p1_action)
        else:  # PPO agent
            s0, _ = process_state(state_dict, layout_name)
            action_p0 = get_prediction_ppo(s0, agent_p0)
        
        if agent_p1 is None:
            action_p1 = 4  # STAY
        elif isinstance(agent_p1, RecordedAgent):
            action_p1 = get_prediction_recorded(agent_p1)
        elif isinstance(agent_p1, MPPI_agent):
            action_p1 = get_prediction_mppi(state_dict, agent_p1, last_p0_action, last_p1_action)
        else:  # PPO agent
            _, s1 = process_state(state_dict, layout_name)
            action_p1 = get_prediction_ppo(s1, agent_p1)
        
        # Store actions
        ep_actions.append([int(action_p0), int(action_p1)])
        
        # Execute actions in environment
        joint_action = (action_idx_to_action(action_p0), action_idx_to_action(action_p1))
        next_state, reward, done = MDP.get_state_transition(state, joint_action)
        
        # Store reward
        ep_rewards.append(float(reward))
        total_reward += reward
        
        # Update for next step
        state = next_state
        state_dict = state.to_dict()
        last_p0_action = action_p0
        last_p1_action = action_p1
        
        if verbose and (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{max_steps}, Reward: {total_reward}")
    
    if verbose:
        print(f"✓ Episode completed: {len(ep_states)} steps, Total Reward: {total_reward}")
    
    return {
        'ep_states': [ep_states],      # Wrapped in list for web format consistency
        'ep_actions': [ep_actions],
        'ep_rewards': [ep_rewards],
        'total_reward': total_reward,
    }


def run_simulation(args):
    """Run full simulation with trajectory collection."""
    
    print("="*70)
    print("OVERCOOKED STANDALONE SIMULATION")
    print("="*70)
    
    # Load agents
    print("\n📦 Loading agents...")
    agent_p0 = None
    agent_p1 = None
    p0_type = "human"  # Default to "human" like web interface
    p1_type = "human"

    num_episodes = args.num_episodes
    if args.replay_player is not None and args.replay_dir is not None:
        traj_files = glob.glob(os.path.join(args.replay_dir, "*.json"))
        if len(traj_files) == 0:
            print(f"Error: No JSON files found in {args.replay_dir}")
            return []
        
        if args.max_replays:
            traj_files = traj_files[:args.max_replays]
        
        print(f"\nFound {len(traj_files)} trajectory file(s) to replay")
        num_episodes = len(traj_files)
    else:
        traj_files = None
    
    if args.modelpath_p0:
        print(f"  P0: Loading PPO from {args.modelpath_p0}")
        agent_p0 = PPO.load(args.modelpath_p0)
        p0_type = "ppo"
    
    if args.modelpath_p1:
        print(f"  P1: Loading PPO from {args.modelpath_p1}")
        agent_p1 = PPO.load(args.modelpath_p1)
        p1_type = "ppo"
    
    if args.use_mppi_p0:
        print(f"  P0: Initializing MPPI (N={args.mppi_n}, H={args.mppi_h})")
        agent_p0 = MPPI_agent(N=5, H=20, agent_id=0, layout_name=args.layout_name)
        p0_type = "mppi"
    
    if args.use_mppi_p1:
        print(f"  P1: Initializing MPPI (N={args.mppi_n}, H={args.mppi_h})")
        agent_p1 = MPPI_agent(N=5, H=20, agent_id=1, layout_name=args.layout_name)
        p1_type = "mppi"
    
    if agent_p0 is None:
        print(f"  P0: Using STAY action (recorded as 'human')")
    if agent_p1 is None:
        print(f"  P1: Using STAY action (recorded as 'human')")
    
    # Run episodes
    print(f"\n🎮 Running {num_episodes} episode(s)...")
    all_trajectories = []
    
    for episode_idx in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode_idx + 1}/{num_episodes}")
        print(f"{'='*70}")

        if args.replay_player is not None and args.replay_dir is not None:
            traj_file = traj_files[episode_idx]
            print(f"Replaying: {os.path.basename(traj_file)}")
            
            recorded_agent = RecordedAgent(traj_file, args.replay_player)
            
            # Assign recorded agent to correct player
            if args.replay_player == 0:
                agent_p0 = recorded_agent
                p0_type = "recorded"
            else:
                agent_p1 = recorded_agent
                p1_type = "recorded"
        
        # Run episode
        episode_data = run_episode(
            agent_p0, 
            agent_p1, 
            args.layout_name,
            max_steps=args.max_steps,
            verbose=True
        )
        
        # Save trajectory in web interface format
        if args.trajs_savepath:
            file_name = f"{args.layout_name}_ep{episode_idx+1}"
            save_trajectory(
                episode_data,
                args.layout_name,
                args.trajs_savepath,
                file_name,
                episode_idx,
                p0_type,
                p1_type,
                algo=args.algo
            )
        
        all_trajectories.append(episode_data)
    
    # Print summary
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    total_rewards = [t['total_reward'] for t in all_trajectories]
    print(f"Episodes completed: {len(all_trajectories)}")
    print(f"Total reward range: [{min(total_rewards):.1f}, {max(total_rewards):.1f}]")
    print(f"Average reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    if args.trajs_savepath:
        print(f"Trajectories saved to: {args.trajs_savepath}")
    print("="*70)
    
    return all_trajectories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run Overcooked simulation standalone and collect trajectories"
    )
    
    # Agent configuration
    parser.add_argument('--modelpath_p0', type=str, default=None,
                        help="Path to PPO model for player 0")
    parser.add_argument('--modelpath_p1', type=str, default=None,
                        help="Path to PPO model for player 1")
    parser.add_argument('--use_mppi_p0', action='store_true',
                        help="Use MPPI agent for player 0")
    parser.add_argument('--use_mppi_p1', action='store_true',
                        help="Use MPPI agent for player 1")
    parser.add_argument('--mppi_n', type=int, default=60,
                        help="Number of MPPI trajectories (samples)")
    parser.add_argument('--mppi_h', type=int, default=20,
                        help="MPPI prediction horizon (timesteps)")
    
    # Replay configuration
    parser.add_argument('--replay_dir', type=str, default=None,
                        help="Directory containing recorded trajectories to replay")
    parser.add_argument('--replay_player', type=int, choices=[0, 1], default=None,
                        help="Which player to replay from recordings (0 or 1)")
    parser.add_argument('--max_replays', type=int, default=None,
                        help="Maximum number of trajectories to replay (default: all)")
    
    # Environment configuration
    parser.add_argument('--layout_name', type=str, required=True,
                        help="Overcooked layout name (e.g., 'cramped_room', 'asymmetric_advantages')")
    
    # Simulation configuration
    parser.add_argument('--num_episodes', type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument('--max_steps', type=int, default=400,
                        help="Maximum steps per episode")
    
    # Save configuration
    parser.add_argument('--trajs_savepath', type=str, default=None,
                        help="Directory to save trajectories (e.g., './trajectories')")
    parser.add_argument('--algo', type=str, default='mppi',
                        help="Algorithm identifier for saved files (e.g., 'mppi', 'ppo', 'mt')")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.modelpath_p0 or args.modelpath_p1 or args.use_mppi_p0 or args.use_mppi_p1):
        print("⚠️  Warning: No agents specified. Both players will use STAY action.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            exit(0)
    
    # Initialize MDP and planner (global variables)
    print(f"\n🗺️  Initializing environment: {args.layout_name}")
    MDP = OvercookedGridworld.from_layout_name(layout_name=args.layout_name)
    MLP = MediumLevelPlanner.from_pickle_or_compute(
        MDP, NO_COUNTERS_PARAMS, force_compute=False
    )
    print("✓ Environment initialized")
    
    # Run simulation
    trajectories = run_simulation(args)
    
    print("\n✅ Simulation complete!")