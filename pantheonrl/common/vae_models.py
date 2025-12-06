import torch
import torch.nn as nn

STATE_DIM = 10       
ACTION_DIM = 6       # One-hot encoded action
INPUT_DIM = 16       # 10 + 6 = 16 total input features

# Keep these the same (unless you want to tune them)
LATENT_DIM = 32
HIDDEN_DIM = 128
BATCH_SIZE = 64      # Decrease if you have very little data (e.g., 16 or 32)
LR = 1e-3
VAE_EPOCHS = 200
TRACKER_EPOCHS = 100
SEQ_LEN = 50

class StrategyVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        
        # --- Encoders ---
        # Encoder H: Human Trajectory -> z_h
        self.encoder_h_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu_h = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar_h = nn.Linear(hidden_dim, latent_dim)

        # Encoder A: Agent Trajectory -> z_a
        self.encoder_a_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu_a = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar_a = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder ---
        # Input: z_joint (z_h + z_a) concatenated with current state to predict next
        self.decoder_lstm = nn.LSTM(2*input_dim + (latent_dim * 2), hidden_dim, batch_first=True)
        self.fc_recon = nn.Linear(hidden_dim, 2*input_dim)
        # Apply softmax to the last action values of the output
        self.softmax_h = nn.Softmax(dim=-1)
        self.softmax_a = nn.Softmax(dim=-1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_human(self, x):
        _, (h, _) = self.encoder_h_lstm(x)
        h = h[-1] # Take last hidden state
        return self.fc_mu_h(h), self.fc_logvar_h(h)

    def encode_agent(self, x):
        _, (h, _) = self.encoder_a_lstm(x)
        h = h[-1]
        return self.fc_mu_a(h), self.fc_logvar_a(h)

    def forward(self, tau_h, tau_a):
        # 1. Encode
        mu_h, logvar_h = self.encode_human(tau_h)
        mu_a, logvar_a = self.encode_agent(tau_a)
        
        z_h = self.reparameterize(mu_h, logvar_h)
        z_a = self.reparameterize(mu_a, logvar_a)
        
        z_joint = torch.cat([z_h, z_a], dim=1) # (Batch, 2*Latent)

        # 2. Decode (Teacher Forcing Reconstruction)
        seq_len = tau_h.size(1)
        z_expanded = z_joint.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Reconstruct Human traj as proxy for joint dynamics
        decoder_input = torch.cat([tau_h, tau_a, z_expanded], dim=2)
        
        recon_out, _ = self.decoder_lstm(decoder_input)
        recon = self.fc_recon(recon_out)
        # Apply softmax to action part of output
        recon_h_actions = self.softmax_h(recon[:, :, STATE_DIM:STATE_DIM+ACTION_DIM])
        recon_a_actions = self.softmax_a(recon[:, :, 2*STATE_DIM+ACTION_DIM:])
        recon_h_states = recon[:, :, :STATE_DIM]
        recon_a_states = recon[:, :, STATE_DIM+ACTION_DIM:2*STATE_DIM+ACTION_DIM]
        recon = torch.cat([recon_h_states, recon_h_actions, recon_a_states, recon_a_actions], dim=2)
        
        return recon, mu_h, logvar_h, mu_a, logvar_a, z_joint
    
    def generate(self, initial_state_h, initial_state_a, z_joint, horizon=20):
        """
        Autoregressive generation for MPPI planning.
        Does not use ground truth; feeds own predictions back in.
        """
        state_h = initial_state_h # Shape: (Batch, 1, Input_Dim)
        state_a = initial_state_a # Shape: (Batch, 1, Input_Dim)
        preds = []
        
        # Initialize LSTM hidden state
        hidden = None 
        
        for _ in range(horizon):
            # Input: Concat current state and latent strategy
            # z_joint needs to be reshaped to (Batch, 1, Latent*2)
            z_input = z_joint.unsqueeze(1)
            decoder_input = torch.cat([state_h, state_a, z_input], dim=2)
            
            # Step LSTM
            out, hidden = self.decoder_lstm(decoder_input, hidden)
            
            # Project back to state space
            next_state = self.fc_recon(out)
            recon_h_actions = self.softmax_h(next_state[:, :, STATE_DIM:STATE_DIM+ACTION_DIM])
            recon_a_actions = self.softmax_a(next_state[:, :, 2*STATE_DIM+ACTION_DIM:])
            recon_h_states = next_state[:, :, :STATE_DIM]
            recon_a_states = next_state[:, :, STATE_DIM+ACTION_DIM:2*STATE_DIM+ACTION_DIM]
            next_state = torch.cat([recon_h_states, recon_h_actions, recon_a_states, recon_a_actions], dim=2)
            preds.append(next_state)
            
            # Update current state for next step
            state_h = next_state[:, :, :STATE_DIM + ACTION_DIM]
            state_a = next_state[:, :, STATE_DIM + ACTION_DIM:]
            
        return torch.cat(preds, dim=1)

class BeliefTracker(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, partial_tau_h):
        # partial_tau_h can be any length
        _, (h, _) = self.lstm(partial_tau_h)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

from typing import Optional
from collections import deque
import numpy as np
import os
import json
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState

def get_action_from_states(current_state, next_state, player_idx="follower"):
    """
    Infers the discrete action (0-5) by comparing position changes between two timesteps.
    This effectively acts as an 'Inverse Dynamics Model', translating state transitions
    back into the actions that caused them.

    Args:
        current_state (Tensor): The state vector at time t. Shape: (16,)
        next_state (Tensor): The state vector at time t+1. Shape: (16,)
        player_idx (str): "leader" (indices 0-4) or "follower" (indices 5-9).
                          Determines which agent's movement we analyze.

    Returns:
        int: The inferred action ID (0-5).
    """
    # 1. Select coordinates based on who we are controlling
    # The input vector is ALWAYS normalized: [Leader, Follower]
    # Indices 0-1 are Leader X,Y. Indices 5-6 are Follower X,Y.
    if player_idx == "leader":
        curr_pos = current_state[0:2]
        next_pos = next_state[0:2]
        curr_held = current_state[4] # Index 4 is Leader's 'Held' status
        next_held = next_state[4]
    else: # follower (Agent)
        curr_pos = current_state[5:7]
        next_pos = next_state[5:7]
        curr_held = current_state[9] # Index 9 is Follower's 'Held' status
        next_held = next_state[9]

    # 2. Calculate Displacement
    dx = next_pos[0] - curr_pos[0]
    dy = next_pos[1] - curr_pos[1]
    
    # Threshold for movement to ignore small numerical noise from the VAE
    MOVE_THRESH = 0.01 
    
    # 3. Determine Action Logic
    # Check for "Interact" first: Held status changed, but didn't move much.
    held_changed = abs(next_held - curr_held) > 0.5
    # if held_changed and abs(dx) < MOVE_THRESH and abs(dy) < MOVE_THRESH:
    #     return 5 # Assume 5 is Interact
    if held_changed:
        return 5 # Assume 5 is Interact

    # Check Movement Direction
    if abs(dx) > abs(dy):
        # Horizontal Movement dominates
        if dx > MOVE_THRESH: return 2  # East/Right
        if dx < -MOVE_THRESH: return 3 # West/Left
    else:
        # Vertical Movement dominates
        if dy > MOVE_THRESH: return 1  # South/Down
        if dy < -MOVE_THRESH: return 0 # North/Up
        
    return 4 # Stay (No significant movement or interaction)

def state_dict_to_vae_observation(state_dict, player_id, action=None):
    """
    Convert web interface state dict to 16-dim observation format expected by VAE.
    
    Args:
        state_dict: State from web interface
        player_id: Which player is the MPPI agent (0 or 1)
        action: Optional action to include (0-5). If None, uses zeros.
    
    Returns:
        16-dim numpy array: [state (10 dims), action one-hot (6 dims)]
    """
    obs = []
    
    # Determine leader (human) vs follower (agent)
    # MPPI agent is the follower
    if player_id == 0:
        leader_idx, follower_idx = 1, 0  # Human is p1, agent is p0
    else:
        leader_idx, follower_idx = 0, 1  # Human is p0, agent is p1
    
    leader = state_dict['players'][leader_idx]
    follower = state_dict['players'][follower_idx]
    
    # Joint state (10 dims) - ALWAYS normalized to [Leader, Follower]
    obs.extend([
        float(leader['position'][0]),
        float(leader['position'][1]),
        float(leader['orientation'][0]),
        float(leader['orientation'][1]),
        1.0 if leader.get('held_object') is not None else 0.0,
        float(follower['position'][0]),
        float(follower['position'][1]),
        float(follower['orientation'][0]),
        float(follower['orientation'][1]),
        1.0 if follower.get('held_object') is not None else 0.0
    ])
    
    # One-hot action (6 dims)
    if action is not None:
        action_vec = [0.0] * 6
        if 0 <= action < 6:
            action_vec[action] = 1.0
        obs.extend(action_vec)
    else:
        obs.extend([0.0] * 6)  # No action / dummy action
    
    return np.array(obs, dtype=np.float32)

class MPPI_agent(Agent):
    def __init__(self, N, T, H, layout_name="cramped_room"):
        self.N = N        # number of trajectories
        self.T = T        # time steps of history
        self.H = H        # horizon length
        self.lambda_ = 1.0  # temperature parameter
        self.history_h = deque(maxlen=T)
        self.history_a = deque(maxlen=T)
        self.layout_name = layout_name
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # FIX: Make it self.DEVICE

        self.vae_dir = "pantheonrl/common/saved_vae_models/"

        with open(os.path.join(self.vae_dir, "vae_config.json"), 'r') as f:
            vae_conf = json.load(f)
        with open(os.path.join(self.vae_dir, "tracker_config.json"), 'r') as f:
            tracker_conf = json.load(f)
        with open(os.path.join(self.vae_dir, "strategy_map.json"), 'r') as f:
            strategy_map = json.load(f)

        # Load VAE model
        self.vae = StrategyVAE(
            input_dim=vae_conf['input_dim'],
            latent_dim=vae_conf['latent_dim'],
            hidden_dim=vae_conf['hidden_dim']
        ).to(self.DEVICE)  # FIX: Use self.DEVICE
        self.vae.load_state_dict(torch.load(self.vae_dir + "strategy_vae.pth", map_location=self.DEVICE))
        self.vae.eval()
        
        # Load belief tracker model
        self.tracker = BeliefTracker(
            input_dim=tracker_conf['input_dim'],
            latent_dim=tracker_conf['latent_dim'],
            hidden_dim=tracker_conf['hidden_dim']
        ).to(self.DEVICE)  # FIX: Use self.DEVICE
        self.tracker.load_state_dict(torch.load(self.vae_dir + "belief_tracker.pth", map_location=self.DEVICE))
        self.tracker.eval()

    def step(self, state_dict, trajectories, return_mode="weighted"):

        optimal_traj, reward = self.trajectory_reward(trajectories, state_dict)
        return optimal_traj

    def intitialize_MDP(self, state_dict):
        MDP = OvercookedGridworld.from_layout_name(layout_name=self.layout_name)
        # print(state_dict)
        for player in state_dict['players']:
            if 'held_object' not in player:
                player['held_object'] = None
        if 'objects' in state_dict:
            if isinstance(state_dict['objects'], dict):
                # Web interface format: {'2,1': {'name': 'dish', 'position': [2, 1]}}
                # Convert to list: [{'name': 'dish', 'position': (2, 1), 'state': None}]
                object_list = []
                for pos_key, obj in state_dict['objects'].items():
                    normalized_obj = obj.copy() if isinstance(obj, dict) else obj
                    
                    # Ensure position is a tuple
                    if isinstance(normalized_obj.get('position'), list):
                        normalized_obj['position'] = tuple(normalized_obj['position'])
                    
                    # Ensure state field exists
                    if 'state' not in normalized_obj:
                        normalized_obj['state'] = None
                        
                    object_list.append(normalized_obj)
                
                state_dict['objects'] = object_list
            elif isinstance(state_dict['objects'], list):
                # Already a list, just normalize each object
                normalized_objects = []
                for obj in state_dict['objects']:
                    normalized_obj = obj.copy() if isinstance(obj, dict) else obj
                    
                    # Ensure position is a tuple
                    if isinstance(normalized_obj.get('position'), list):
                        normalized_obj['position'] = tuple(normalized_obj['position'])
                    
                    # Ensure state field exists
                    if 'state' not in normalized_obj:
                        normalized_obj['state'] = None
                        
                    normalized_objects.append(normalized_obj)
                
                state_dict['objects'] = normalized_objects
        else:
            state_dict['objects'] = []
        # print(state_dict)
        state = OvercookedState.from_dict(state_dict)
        return MDP, state
        
    def trajectory_reward(self, trajectories, state_dict):
        # MDP = OvercookedGridworld.from_layout_name(layout_name=self.layout_name)
        # print(state)
        # for player in state['players']:
        #     if 'held_object' not in player:
        #         player['held_object'] = None
        # if 'objects' in state:
        #     if isinstance(state['objects'], dict):
        #         # Web interface format: {'2,1': {'name': 'dish', 'position': [2, 1]}}
        #         # Convert to list: [{'name': 'dish', 'position': (2, 1), 'state': None}]
        #         object_list = []
        #         for pos_key, obj in state['objects'].items():
        #             normalized_obj = obj.copy() if isinstance(obj, dict) else obj
                    
        #             # Ensure position is a tuple
        #             if isinstance(normalized_obj.get('position'), list):
        #                 normalized_obj['position'] = tuple(normalized_obj['position'])
                    
        #             # Ensure state field exists
        #             if 'state' not in normalized_obj:
        #                 normalized_obj['state'] = None
                        
        #             object_list.append(normalized_obj)
                
        #         state['objects'] = object_list
        #     elif isinstance(state['objects'], list):
        #         # Already a list, just normalize each object
        #         normalized_objects = []
        #         for obj in state['objects']:
        #             normalized_obj = obj.copy() if isinstance(obj, dict) else obj
                    
        #             # Ensure position is a tuple
        #             if isinstance(normalized_obj.get('position'), list):
        #                 normalized_obj['position'] = tuple(normalized_obj['position'])
                    
        #             # Ensure state field exists
        #             if 'state' not in normalized_obj:
        #                 normalized_obj['state'] = None
                        
        #             normalized_objects.append(normalized_obj)
                
        #         state['objects'] = normalized_objects
        # else:
        #     state['objects'] = []
        # print(state)
        # state = OvercookedState.from_dict(state)
        
        optimal_traj = []
        optimal_traj_reward = float('-inf')
        for trajectory in trajectories:
            MDP, state = self.intitialize_MDP(state_dict)
            traj_actions = trajectory
            traj_total_reward = 0
            for action in traj_actions:
                joint_action = (Action.ALL_ACTIONS[action], Action.ALL_ACTIONS[0])
                next_state, reward, _ = MDP.get_state_transition(state, joint_action)
                state = next_state
                traj_total_reward += reward
            if traj_total_reward > optimal_traj_reward:
                optimal_traj_reward = traj_total_reward
                optimal_traj = traj_actions
        return optimal_traj, optimal_traj_reward
        
    def predict(self, state_dict):

        observation_h = state_dict_to_vae_observation(state_dict, player_id=1, action=0)
        observation_a = state_dict_to_vae_observation(state_dict, player_id=2, action=0)

        # Convert observation to tensor if needed
        if not isinstance(observation_h, torch.Tensor):
            observation_h = torch.tensor(observation_h, dtype=torch.float32).to(self.DEVICE)
        if not isinstance(observation_a, torch.Tensor):
            observation_a = torch.tensor(observation_a, dtype=torch.float32).to(self.DEVICE)
        
        # Ensure observation has correct shape (batch, seq, features)
        if observation_h.dim() == 1:
            observation_h = observation_h.unsqueeze(0).unsqueeze(0)  # (16,) -> (1, 1, 16)
        elif observation_h.dim() == 2:
            observation_h = observation_h.unsqueeze(1)  # (1, 16) -> (1, 1, 16)

        if observation_a.dim() == 1:
            observation_a = observation_a.unsqueeze(0).unsqueeze(0)  # (16,) -> (1, 1, 16)
        elif observation_a.dim() == 2:
            observation_a = observation_a.unsqueeze(1)  # (1, 16) -> (1, 1, 16)
        
        self.history_h.append(observation_h)
        self.history_a.append(observation_a)

        # Convert deque to tensor for model input
        history_h_tensor = torch.cat(list(self.history_h), dim=1)  # Concatenate along sequence dimension
        history_a_tensor = torch.cat(list(self.history_a), dim=1)  # Concatenate along sequence dimension

        with torch.no_grad():
            # use history_tensor instead of self.history
            mu_h, logvar_h = self.tracker(history_h_tensor)
            std_h = torch.exp(0.5 * logvar_h)
            print("std: ", np.linalg.norm(std_h.cpu().numpy()))
            
            # Sample Human Intent
            z_h = torch.distributions.Normal(mu_h, std_h).sample((self.N,)).squeeze(1)
            
            # Sample Agent Intent
            z_a = torch.randn(self.N, self.vae.latent_dim).to(self.DEVICE)
            # z_a = z_h.clone()
            
            # Combine into Joint Strategy
            z_joint = torch.cat([z_h, z_a], dim=1)
            
            # Get current state from history tensor
            current_state_h = history_h_tensor[:, -1:, :].repeat(self.N, 1, 1)
            current_state_a = history_a_tensor[:, -1:, :].repeat(self.N, 1, 1)
            
            # Generate state trajectories
            state_trajectories = self.vae.generate(current_state_h, current_state_a, z_joint, horizon=self.H)
            state_trajectories_h = state_trajectories[:, :, :STATE_DIM + ACTION_DIM]
            state_trajectories_a = state_trajectories[:, :, STATE_DIM + ACTION_DIM:]

            # print("state_trajectories: ", state_trajectories.shape)
            
            action_trajectories = []
            for traj_idx in range(self.N):
                actions = []
                for t in range(self.H - 1):  # H-1 because we need pairs of states
                    curr_state = state_trajectories_a[traj_idx, t, :]
                    next_state = state_trajectories_a[traj_idx, t+1, :]
                    action = np.argmax(curr_state.cpu().numpy()[-6:])
                    # action = get_action_from_states(curr_state.cpu().numpy(), 
                    #                                next_state.cpu().numpy(), 
                    #                                player_idx="follower")
                    actions.append(action)
                # Add final action (or repeat last action)
                # actions.append(actions[-1] if actions else 4)  # Default to STAY if empty
                action_trajectories.append(actions)
            
            action_trajectories = np.array(action_trajectories)
            # action_trajectories = np.array([traj[traj != 4] for traj in action_trajectories], dtype=object)

            print(action_trajectories)

        # Find best trajectory using actual MDP rollouts
        best_traj = self.step(state_dict, action_trajectories, return_mode="best")
        print("best traj: ", best_traj)
        
        # Return first action from best trajectory
        return [best_traj, observation_h.cpu().numpy()]