import torch
import torch.nn as nn

# STATE_DIM = 10       
# ACTION_DIM = 6       # One-hot encoded action
# INPUT_DIM = 16       # 10 + 6 = 16 total input features

# # Keep these the same (unless you want to tune them)
# LATENT_DIM = 32
# HIDDEN_DIM = 128
# BATCH_SIZE = 64      # Decrease if you have very little data (e.g., 16 or 32)
# LR = 1e-3
# VAE_EPOCHS = 200
# TRACKER_EPOCHS = 100
# SEQ_LEN = 50

# Cell 3: Model Classes
class StrategyVAE(nn.Module):
    def __init__(self, input_state_dim, input_p0_act_dim, input_p1_act_dim, num_past_actions_in_generator, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_state_dim = input_state_dim
        self.input_p0_act_dim = input_p0_act_dim
        self.input_p1_act_dim = input_p1_act_dim
        
        # --- Encoders ---
        # Encoder H: Human Trajectory -> z_h
        self.encoder_p0_lstm = nn.LSTM(input_state_dim + input_p0_act_dim, self.hidden_dim, batch_first=True)
        self.fc_mu_h = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar_h = nn.Linear(self.hidden_dim, self.latent_dim)

        # Encoder A: Agent Trajectory -> z_a
        self.encoder_p1_lstm = nn.LSTM(input_state_dim + input_p1_act_dim, self.hidden_dim, batch_first=True)
        self.fc_mu_a = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar_a = nn.Linear(self.hidden_dim, self.latent_dim)

        # --- Decoder ---
        # Input: z_joint (z_h + z_a) concatenated with current state to predict next
        # self.decoder_lstm = nn.LSTM(input_state_dim + input_p0_act_dim + input_p1_act_dim + (self.latent_dim * 2), self.hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_state_dim + (num_past_actions_in_generator * (input_p0_act_dim + input_p1_act_dim)) + (self.latent_dim * 2), self.hidden_dim, batch_first=True)
        self.fc_recon = nn.Linear(self.hidden_dim, input_state_dim + input_p0_act_dim + input_p1_act_dim)
        # Apply softmax to the last action values of the output
        self.softmax = nn.Softmax(dim=-1)
        self.argmax = lambda x: torch.zeros_like(x).scatter_(
            -1, torch.argmax(x, dim=-1, keepdim=True), 1.0
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_p0(self, x):
        _, (h, _) = self.encoder_p0_lstm(x)
        h = h[-1] # Take last hidden state
        return self.fc_mu_h(h), self.fc_logvar_h(h)

    def encode_p1(self, x):
        _, (h, _) = self.encoder_p1_lstm(x)
        h = h[-1]
        return self.fc_mu_a(h), self.fc_logvar_a(h)

    def forward(self, state, action_p0, action_p1, past_actions_p0, past_actions_p1):
        # 1. Encode
        tau_p0 = torch.cat([state, action_p0], dim=2)
        tau_p1 = torch.cat([state, action_p1], dim=2)
        mu_h, logvar_h = self.encode_p0(tau_p0)
        mu_a, logvar_a = self.encode_p1(tau_p1)
        
        z_h = self.reparameterize(mu_h, logvar_h)
        z_a = self.reparameterize(mu_a, logvar_a)
        
        z_joint = torch.cat([z_h, z_a], dim=1) # (Batch, 2*Latent)

        # 2. Decode (Teacher Forcing Reconstruction)
        seq_len = state.size(1)
        z_expanded = z_joint.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Reconstruct Human traj as proxy for joint dynamics
        decoder_input = torch.cat([state, past_actions_p0, past_actions_p1, z_expanded], dim=2)
        
        recon_out, _ = self.decoder_lstm(decoder_input)
        recon = self.fc_recon(recon_out)
        
        return recon, mu_h, logvar_h, mu_a, logvar_a, z_joint
    
    def generate(self, initial_state, z_joint, past_actions_p0, past_actions_p1, horizon=20):
        """
        Autoregressive generation for MPPI planning.
        Does not use ground truth; feeds own predictions back in.
        """
        state_h = initial_state # Shape: (Batch, 1, Input_Dim)
        preds = []
        
        # Initialize LSTM hidden state
        hidden = None 
        last_state = initial_state
        z_input = z_joint
        
        N = z_joint.size(0)
        p0_actions = torch.zeros((N, horizon, self.input_p0_act_dim)).to(initial_state.device)
        p1_actions = torch.zeros((N, horizon, self.input_p1_act_dim)).to(initial_state.device)
        with torch.no_grad():
            for idx in range(horizon):
                # Input: Concat current state and latent strategy
                # z_joint needs to be reshaped to (Batch, 1, Latent*2)
                
                decoder_input = torch.cat([last_state, past_actions_p0, past_actions_p1, z_input], dim=1)

                # Step LSTM
                out, hidden = self.decoder_lstm(decoder_input, hidden)
                
                # Project back to state space
                reconstruction = self.fc_recon(out)
                recon_states = reconstruction[:, :self.input_state_dim]
                recon_states[:, 4:8] = self.argmax(self.softmax(recon_states[:, 4:8]))
                recon_states[:, 12:16] = self.argmax(self.softmax(recon_states[:, 12:16]))
                for i in range(10):
                    recon_states[:, 16 + 2+ i*6:16 + 6 + i*6] = self.argmax(self.softmax(recon_states[:, 16 + 2+ i*6:16 + 6 + i*6]))
                # recon_states = torch.round(recon_states.to(DEVICE))
                recon_states[:, 0:4] = torch.round(recon_states[:, 0:4])    # P0 pos/orient only
                recon_states[:, 8:12] = torch.round(recon_states[:, 8:12])  # P1 pos/orient only

                # recon_p0_actions = self.argmax(self.softmax(reconstruction[:, self.input_state_dim:self.input_state_dim + self.input_p0_act_dim]))
                # recon_p1_actions = self.argmax(self.softmax(reconstruction[:, self.input_state_dim + self.input_p0_act_dim:self.input_state_dim + self.input_p0_act_dim + self.input_p1_act_dim]))
                wait_action_index = 4
                p0_probs = self.softmax(reconstruction[:, self.input_state_dim:self.input_state_dim + self.input_p0_act_dim])
                p0_probs[:, wait_action_index] *= 0.3  # Make wait less likely
                p0_probs = p0_probs / p0_probs.sum(dim=-1, keepdim=True)
                recon_p0_actions = self.argmax(p0_probs)

                p1_probs = self.softmax(reconstruction[:, self.input_state_dim + self.input_p0_act_dim:self.input_state_dim + self.input_p0_act_dim + self.input_p1_act_dim])
                p1_probs[:, wait_action_index] *= 0.3  # Make wait less likely
                p1_probs = p1_probs / p1_probs.sum(dim=-1, keepdim=True)
                recon_p1_actions = self.argmax(p1_probs)
                
                reconstruction = torch.cat([recon_states, recon_p0_actions, recon_p1_actions], dim=1)
                preds.append(reconstruction)

                last_state = recon_states
                past_actions_p0 = torch.cat([past_actions_p0[:, 6:], recon_p0_actions], dim=1)
                past_actions_p1 = torch.cat([past_actions_p1[:, 6:], recon_p1_actions], dim=1)

                p0_actions[:, idx, :] = recon_p0_actions
                p1_actions[:, idx, :] = recon_p1_actions
            
        p0_actions = torch.argmax(p0_actions, axis=2)
        p1_actions = torch.argmax(p1_actions, axis=2)
        return p0_actions.cpu().numpy(), p1_actions.cpu().numpy()

class BeliefTracker(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, partial_tau_h):
        # partial_tau_h can be any length
        _, (h, _) = self.lstm(partial_tau_h)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

class ScoringModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        score = self.fc2(h)
        return score

from typing import Optional
from collections import deque
import numpy as np
import os
import json
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState

class MPPI_agent(Agent):
    def __init__(self, N, H, agent_id=1, layout_name="cramped_room"):
        self.agent_id = agent_id  # Follower # TODO: Make MPPI work as either agent!
        self.p0_idx = 0
        self.p1_idx = 1
        self.N = N        # number of trajectories
        self.H = H        # horizon length
        self.lambda_ = 1.0  # temperature parameter
        self.layout_name = layout_name
        # self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DEVICE = torch.device("cuda")
        print("Using device:", self.DEVICE)
        

        self.vae_dir = "PantheonRL/pantheonrl/common/saved_models_with_scoring/"

        with open(os.path.join(self.vae_dir, "vae_config.json"), 'r') as f:
            vae_conf = json.load(f)
        with open(os.path.join(self.vae_dir, "tracker_config.json"), 'r') as f:
            tracker_conf = json.load(f)
        with open(os.path.join(self.vae_dir, "scoring_model_config.json"), 'r') as f:
            scoring_model_conf = json.load(f)
        with open(os.path.join(self.vae_dir, "strategy_map.json"), 'r') as f:
            strategy_map = json.load(f)

        self.p0_action_history = np.zeros((vae_conf['action_dim'] * vae_conf['num_past_actions_in_generator'],), dtype=np.float32)
        self.p1_action_history = np.zeros((vae_conf['action_dim'] * vae_conf['num_past_actions_in_generator'],), dtype=np.float32)
        self.STATE_DIM = vae_conf['state_dim']
        self.ACTION_DIM = vae_conf['action_dim']

        # Load VAE model
        self.vae = StrategyVAE(
            input_state_dim=vae_conf['state_dim'], 
            input_p0_act_dim=vae_conf['action_dim'], 
            input_p1_act_dim=vae_conf['action_dim'],
            num_past_actions_in_generator=vae_conf['num_past_actions_in_generator'],
            latent_dim=vae_conf['latent_dim'],
            hidden_dim=vae_conf['hidden_dim']
        ).to(self.DEVICE)  # FIX: Use self.DEVICE
        self.vae.load_state_dict(torch.load(self.vae_dir + "strategy_vae.pth", map_location=self.DEVICE))
        self.vae.eval()
        
        # Load belief tracker model
        self.tracker = BeliefTracker(
            input_dim=tracker_conf['input_dim'],
            latent_dim=tracker_conf['latent_dim']*2,
            hidden_dim=tracker_conf['hidden_dim']
        ).to(self.DEVICE)  # FIX: Use self.DEVICE
        self.tracker.load_state_dict(torch.load(self.vae_dir + "belief_tracker.pth", map_location=self.DEVICE))
        self.tracker.eval()

        self.scoring_model = ScoringModel(
            input_dim=scoring_model_conf['input_dim'],
            hidden_dim=scoring_model_conf['hidden_dim']
        ).to(self.DEVICE)

    def step(self, state_dict, trajectories, z_joint_batched, return_mode="weighted"):

        optimal_traj, reward, optimal_traj_idx = self.trajectory_reward(trajectories, z_joint_batched, state_dict)
        # print("Optimal Trajectory Reward:", reward)
        # print("Optimal Trajectory Index:", optimal_traj_idx)
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
        
    def trajectory_reward(self, trajectories, latent_states, state_dict):   
        with torch.no_grad():
            strategy_compatibility_reward_batch = self.scoring_model(latent_states).cpu().numpy()
        optimal_traj = []
        optimal_traj_idx = -1
        optimal_reward = float('-inf')
        for idx, trajectory in enumerate(trajectories):
            MDP, state = self.intitialize_MDP(state_dict)
            traj_actions = trajectory
            traj_total_reward = 0
            strategy_compatibility_reward = strategy_compatibility_reward_batch[idx]
            for action in traj_actions:
                joint_action = (Action.ALL_ACTIONS[action[0]], Action.ALL_ACTIONS[action[1]])
                next_state, reward, _ = MDP.get_state_transition(state, joint_action)
                state = next_state
                traj_total_reward += reward
            total_reward = traj_total_reward + (strategy_compatibility_reward*0.05)
            # total_reward = traj_total_reward
            if total_reward > optimal_reward:
                optimal_reward = total_reward
                optimal_traj = traj_actions
                optimal_traj_idx = idx
        return optimal_traj, optimal_reward, optimal_traj_idx
        
    def state_dict_to_vae_state(self, state_dict):
        players = state_dict['players']
        p0 = players[self.p0_idx]
        p1 = players[self.p1_idx]
        
        p0_held = [1, 0, 0, 0]
        if p0.get('held_object'):
            obj = p0.get('held_object')
            if obj['name'] == "onion":
                p0_held = [0, 1, 0, 0]
            elif obj['name'] == "dish":
                p0_held = [0, 0, 1, 0]
            elif obj['name'] == "soup":
                p0_held = [0, 0, 0, 1]

        p1_held = [1, 0, 0, 0]
        if p1.get('held_object'):
            obj = p1.get('held_object')
            if obj['name'] == "onion":
                p1_held = [0, 1, 0, 0]
            elif obj['name'] == "dish":
                p1_held = [0, 0, 1, 0]
            elif obj['name'] == "soup":
                p1_held = [0, 0, 0, 1]

        # Include first 10 items in state_dict (pox_x, pos_y, type (one hot onion/dish/soup))
        state_items = [0] * (6 * 10)
        for idx, item in enumerate(state_dict.get('items', [])[:10]):
            state_items[idx*6 + 0] = float(item['position'][0])
            state_items[idx*6 + 1] = float(item['position'][1])
            if item['name'] == 'onion':
                state_items[idx*6 + 3] = 1.0
            elif item['name'] == 'dish':
                state_items[idx*6 + 4] = 1.0
            elif item['name'] == 'soup':
                state_items[idx*6 + 5] = 1.0
            else:
                state_items[idx*6 + 2] = 1.0

        state = [
            float(p0['position'][0]), float(p0['position'][1]),
            float(p0['orientation'][0]), float(p0['orientation'][1])]+ p0_held + [
            float(p1['position'][0]), float(p1['position'][1]),
            float(p1['orientation'][0]), float(p1['orientation'][1])] + p1_held + state_items
        
        return np.array(state, dtype=np.float32)

    def one_hot_to_action(self, one_hot_vec):
        return int(np.argmax(one_hot_vec))

    def predict(self, state_dict, last_p0_action, last_p1_action):

        state = self.state_dict_to_vae_state(state_dict)
        
        # One-Hot Encode Actions
        last_p0_act_vec = [0]*6
        if 0 <= last_p0_action < 6: 
            last_p0_act_vec[last_p0_action] = 1.0
        self.p0_action_history = np.roll(self.p0_action_history, -6)
        
        last_p1_act_vec = [0]*6
        if 0 <= last_p1_action < 6: 
            last_p1_act_vec[last_p1_action] = 1.0
        self.p1_action_history = np.roll(self.p1_action_history, -6)

        # Convert observation to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.DEVICE)

        last_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.DEVICE)  # Shape: (1, T, STATE_DIM)
        last_action_p0_tensor = torch.tensor(last_p0_act_vec, dtype=torch.float32).unsqueeze(0).to(self.DEVICE)
        last_action_p1_tensor = torch.tensor(last_p1_act_vec, dtype=torch.float32).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            # use history_tensor instead of self.history
            mu, logvar = self.tracker(torch.cat([last_state_tensor, last_action_p0_tensor, last_action_p1_tensor], dim=1))
            # std = torch.exp(0.5 * logvar)
            # z_joint_batched = torch.distributions.Normal(mu, std).sample((self.N,)).squeeze(1).to(self.DEVICE)

            mu_p0 = mu[:mu.size(0)//2]
            mu_p1 = mu[mu.size(0)//2:]
            logvar_p0 = logvar[:logvar.size(0)//2]
            logvar_p1 = logvar[logvar.size(0)//2:]
            std_p0 = torch.exp(0.5 * logvar_p0)
            std_p1 = torch.exp(0.5 * logvar_p1)

            if self.agent_id == 0:
                z_p0 = torch.distributions.Normal(mu_p0, std_p0*5).sample((self.N,)).to(self.DEVICE)
                z_p1 = mu_p1.repeat(self.N, 1).to(self.DEVICE)
            else:
                z_p0 = mu_p0.repeat(self.N, 1).to(self.DEVICE)
                z_p1 = torch.distributions.Normal(mu_p1, std_p1*5).sample((self.N,)).to(self.DEVICE)
            
            z_joint_batched = torch.cat([z_p0, z_p1], dim=1).squeeze(1).to(self.DEVICE)
            
            p0_action_history_tensor_batched = torch.tensor(self.p0_action_history, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1).to(self.DEVICE)
            p1_action_history_tensor_batched = torch.tensor(self.p1_action_history, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1).to(self.DEVICE)
            last_state_tensor_batched = last_state_tensor.repeat(self.N, 1).to(self.DEVICE)

            p0_action_batched, p1_action_batched = self.vae.generate(initial_state=last_state_tensor_batched, z_joint=z_joint_batched, past_actions_p0=p0_action_history_tensor_batched, past_actions_p1=p1_action_history_tensor_batched, horizon=self.H)
            action_trajectories_p0 = p0_action_batched
            action_trajectories_p1 = p1_action_batched
            action_trajectories = np.zeros((self.N, self.H, 2), dtype=int)
            action_trajectories[:, :, 0] = action_trajectories_p0
            action_trajectories[:, :, 1] = action_trajectories_p1

        # Find best trajectory using actual MDP rollouts
        best_traj = self.step(state_dict, action_trajectories, z_joint_batched, return_mode="best")[:, self.agent_id].flatten()

        # Get first action that is not 4 (STAY)
        first_action = None
        for action in best_traj:
            if action != 4:
                first_action = action
                break
        
        # If all actions are 4, default to the first action
        if first_action is None:
            first_action = best_traj[0] if len(best_traj) > 0 else 4
        
        # Return first action from best trajectory
        return [first_action, self.agent_id]