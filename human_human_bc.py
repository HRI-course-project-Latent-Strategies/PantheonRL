#!/usr/bin/env python3
"""
Behavior Cloning (BC) for Overcooked using an SB3 ActorCriticPolicy.

- Loads human-human json trajectories from a directory.
- Infers which player is human from the filename prefix (p1*, p2*).
- Featurizes states via Overcooked's MediumLevelPlanner.
- Trains an SB3 ActorCriticPolicy with supervised BC (neg log-likelihood).
- Saves the trained policy in SB3's policy format (.zip).

Example:
    python train_overcooked_bc_sb3.py \
        --data-root /home/mehul/HRI/PantheonRL/human-human-game-data \
        --env OvercookedMultiEnv-v0 \
        --layout-name unident_s \
        --total-epochs 500 \
        --batch-size 64 \
        --save-path saved_bc_policies/bc_overcooked_unident_s.zip
"""

import os
import glob
import json
import copy
from typing import Any, Dict, Mapping, Optional, Type, Union, Iterable, Tuple

import numpy as np
import torch
import torch as th
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam

import gym
from stable_baselines3.common import policies
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device as sb3_get_device

from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    ObjectState,
)
from overcooked_ai_py.mdp.actions import Direction
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from pantheonrl.common import trajsaver
from pantheonrl.common.multiagentenv import SimultaneousEnv
from trainer import generate_env, ENV_LIST, LAYOUT_LIST

import argparse

from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent
from pantheonrl.algos.bc import BCShell, reconstruct_policy

# =========================================================
# Exceptions / small helpers
# =========================================================

class EnvException(Exception):
    """Raise when parameters do not align with environment."""


class ConstantLRSchedule:
    """SB3-style constant LR schedule."""
    def __call__(self, _):
        return 1.0


class FeedForward32Policy(policies.ActorCriticPolicy):
    """
    SB3 ActorCriticPolicy with a simple [32, 32] MLP.
    Suitable for Overcooked featurized observations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[32, 32])


# =========================================================
# BC core
# =========================================================

class BC:
    DEFAULT_BATCH_SIZE: int = 32

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        policy_class: Type[policies.BasePolicy] = FeedForward32Policy,
        policy_kwargs: Optional[Mapping[str, Any]] = None,
        expert_data: Optional[Iterable[Mapping[str, Any]]] = None,
        optimizer_cls: Type[th.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, th.device] = "auto",
    ):
        """
        Behavioral cloning on top of an SB3 policy (ActorCriticPolicy).

        The training objective is:
            loss = -E[log pi(a|s)]   # neg log-likelihood
                   - ent_weight * H[pi(.|s)]
                   + l2_weight * ||theta||^2
        """
        if optimizer_kwargs and "weight_decay" in optimizer_kwargs:
            raise ValueError("Use `l2_weight` instead of `weight_decay`.")

        self.action_space = action_space
        self.observation_space = observation_space
        self.policy_class = policy_class
        self.device = sb3_get_device(device)

        # SB3 policy kwargs
        self.policy_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=ConstantLRSchedule(),
        )
        self.policy_kwargs.update(policy_kwargs or {})

        # SB3 ActorCriticPolicy
        self.policy: ActorCriticPolicy = self.policy_class(**self.policy_kwargs).to(self.device)

        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(self.policy.parameters(), **optimizer_kwargs)

        self.expert_data_loader: Optional[Iterable[Mapping[str, Any]]] = None
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

        if expert_data is not None:
            self.set_expert_data_loader(expert_data)

    def set_expert_data_loader(
        self,
        expert_data: Iterable[Mapping[str, Any]],
    ) -> None:
        """
        expert_data: any iterable that yields dicts with keys:
            - "obs":  Tensor or np.ndarray [B, obs_dim]
            - "acts": Tensor or np.ndarray [B] or [B, act_dim]
        Typically a torch DataLoader around OvercookedBCDataset.
        """
        self.expert_data_loader = expert_data

    def _calculate_loss(
        self,
        obs: Union[th.Tensor, np.ndarray],
        acts: Union[th.Tensor, np.ndarray],
    ) -> Tuple[th.Tensor, Dict[str, float]]:
        obs = th.as_tensor(obs, device=self.device).detach()
        acts = th.as_tensor(acts, device=self.device).detach()

        _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        l2_norms = [th.sum(th.square(w)) for w in self.policy.parameters()]
        l2_norm = sum(l2_norms) / 2.0  # /2 cancels gradient of square

        ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        stats_dict = dict(
            neglogp=float(neglogp.item()),
            loss=float(loss.item()),
            entropy=float(entropy.item()),
            ent_loss=float(ent_loss.item()),
            prob_true_act=float(prob_true_act.item()),
            l2_norm=float(l2_norm.item()),
            l2_loss=float(l2_loss.item()),
        )
        return loss, stats_dict

    def train(
        self,
        n_epochs: int,
        log_interval: int = 100,
    ):
        """
        Simple training loop: go over dataset for `n_epochs`.
        """
        assert self.expert_data_loader is not None, "Call set_expert_data_loader first."
        batch_num = 0

        for epoch in range(n_epochs):
            for batch in self.expert_data_loader:
                loss, stats_dict = self._calculate_loss(batch["obs"], batch["acts"])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_num % log_interval == 0:
                    print(
                        f"[Epoch {epoch} | Batch {batch_num}] "
                        f"loss={stats_dict['loss']:.4f} "
                        f"neglogp={stats_dict['neglogp']:.4f} "
                        f"entropy={stats_dict['entropy']:.4f} "
                        f"prob_true_act={stats_dict['prob_true_act']:.4f}"
                    )
                batch_num += 1

    def save_policy(self, policy_path: str) -> None:
        # """
        # Save only the SB3 policy (ActorCriticPolicy).

        # This creates a standard SB3 `.zip` that you can later load via:
        #     policy = FeedForward32Policy.load(policy_path)
        # """
        # os.makedirs(os.path.dirname(policy_path), exist_ok=True)
        # th.save(self.policy, policy_path)
        # self.policy.save(policy_path)
        """
        Saves:
            - SB3 policy: <policy_path>.zip
            - PyTorch weights (state_dict): <policy_path>.pt
        """
        dirpath = os.path.dirname(policy_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        zip_path = policy_path if policy_path.endswith(".zip") else policy_path + ".zip"
        pt_path  = policy_path.replace(".zip", "") + ".pt"
        pth_path = policy_path.replace(".zip", "") + ".pth"   # <-- PantheonRL-compatible
        # SB3 format (for SB3 loading)
        self.policy.save(zip_path)

        # Raw PyTorch weights
        torch.save(self.policy.state_dict(), pt_path)

        torch.save(self.policy, pth_path)
        
        print(f"Saved SB3 policy to: {zip_path}")
        print(f"Saved PyTorch state_dict to: {pt_path}")
        print(f"Saved PantheonRL-compatible .pth to: {pth_path}")
# =========================================================
# JSON → OvercookedState conversion (safe)
# =========================================================

def json_to_state_safe(s: Dict[str, Any]) -> Optional[OvercookedState]:
    s = copy.deepcopy(s)

    # ---------- SAFE PLAYERS ----------
    fixed_players = []
    for p in s.get("players", []):
        # Ensure required keys exist
        if "position" not in p:
            # Bad player frame → skip this whole state safely
            return None

        if "orientation" not in p:
            p["orientation"] = [0, 1]  # default orientation

        # Fix orientation if it's a list vector
        if isinstance(p["orientation"], list):
            vec = tuple(p["orientation"])
            mapping = {
                (0, 1): Direction.SOUTH,
                (0, -1): Direction.NORTH,
                (1, 0): Direction.EAST,
                (-1, 0): Direction.WEST,
            }
            p["orientation"] = mapping.get(vec, Direction.NORTH)

        # Fix held_object
        ho = p.get("held_object", None)
        if ho is not None:
            # Fix missing position
            if "position" not in ho:
                ho["position"] = p["position"]  # assume held in hand

            # Fix malformed object dicts
            if "name" not in ho:
                ho["name"] = "unknown"
        else:
            p["held_object"] = None

        fixed_players.append(p)

    s["players"] = [PlayerState.from_dict(p) for p in fixed_players]

    # ---------- SAFE OBJECTS ----------
    raw_objects = s.get("objects", {})
    fixed_objects = {}

    if isinstance(raw_objects, list):
        for o in raw_objects:
            if not isinstance(o, dict):
                continue
            if "position" not in o:
                continue   # skip malformed objects
            if "name" not in o:
                o["name"] = "unknown"
            try:
                ob = ObjectState.from_dict(o)
                fixed_objects[tuple(ob.position)] = ob
            except Exception:
                continue

    elif isinstance(raw_objects, dict):
        for _, o in raw_objects.items():
            if not isinstance(o, dict):
                continue
            if "position" not in o:
                continue
            if "name" not in o:
                o["name"] = "unknown"
            try:
                ob = ObjectState.from_dict(o)
                fixed_objects[tuple(ob.position)] = ob
            except Exception:
                continue

    s["objects"] = fixed_objects

    # ---------- FINAL SAFE CONVERSION ----------
    try:
        return OvercookedState(**s)
    except Exception:
        return None


# =========================================================
# Dataset that chooses player based on filename (p1*/p2*)
# =========================================================

class OvercookedBCDataset(Dataset):
    """
    Dataset for BC on Overcooked:

    - `data_json` is a single trajectory file (with "traj" dict inside).
    - `filename` is used to determine which player is treated as human:
        - if basename startswith("p1") → learn player 0
        - else → learn player 1
    - Featurizes each state with mdp.featurize_state + MediumLevelPlanner.
    """

    def __init__(self, data_json, mdp, mlp, filename: str):
        self.mdp = mdp
        self.mlp = mlp

        base = os.path.basename(filename)
        # if base.startswith("p1"):
        #     self.agent_idx = 0
        # else:
        #     self.agent_idx = 1
            
        self.agent_idx = 0 # TRAINING FOR PLAYER 0 ALWAYS
        print(f"Learning actions for PLAYER {self.agent_idx} from {base}")

        self.states = []
        self.actions = []

        traj = data_json["traj"]
        ep_states = traj["ep_states"]
        ep_actions = traj["ep_actions"]

        for ep_s, ep_a in zip(ep_states, ep_actions):
            for s, a in zip(ep_s, ep_a):
                state_obj = json_to_state_safe(s)
                if state_obj is None:
                    continue  # skip bad frames
                self.states.append(state_obj)
                self.actions.append(a)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        a0, a1 = self.actions[idx]

        # Compute features for both agents
        obs0, obs1 = self.mdp.featurize_state(state, self.mlp)

        # Select correct agent
        if self.agent_idx == 0:
            obs = obs0
            act = a0
        else:
            obs = obs1
            act = a1

        return {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "acts": torch.tensor(act, dtype=torch.long),
        }


# =========================================================
# MDP + MediumLevelPlanner construction
# =========================================================

def create_mlp(mdp: OvercookedGridworld) -> MediumLevelPlanner:
    counter_locs = mdp.get_counter_locations()

    mlp_params = {
        "wait_allowed": True,
        "counter_drop": counter_locs,
        "counter_pickup": counter_locs,
        "start_orientations": False,
        "same_motion_goals": True,
        "counter_goals": counter_locs,
    }

    mlp = MediumLevelPlanner.from_pickle_or_compute(
        mdp,
        mlp_params,
        force_compute=True,
    )
    return mlp


# =========================================================
# Utility: build dataset from a data_root
# =========================================================

def build_dataset_from_root(
    data_root: str,
    default_layout: str,
) -> ConcatDataset:
    """
    Recursively loads all *.json under `data_root` and concatenates them
    into a single dataset for BC.
    """
    json_files = glob.glob(os.path.join(data_root, "**/p1*.json"), recursive=True)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {data_root}")

    datasets = []
    skipped = 0

    for filename in json_files:
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            layout_name = data.get("traj", {}).get("layout_name", default_layout)
            mdp = OvercookedGridworld.from_layout_name(layout_name)
            mlp = create_mlp(mdp)

            ds = OvercookedBCDataset(data, mdp, mlp, filename)
            if len(ds) == 0:
                print(f"SKIPPED (empty) {filename}")
                skipped += 1
                continue

            datasets.append(ds)
            print(f"Loaded ✓ {filename} with {len(ds)} samples")

        except Exception as e:
            print(f"SKIPPED ✗ {filename} because: {e}")
            skipped += 1

    if not datasets:
        raise RuntimeError(f"All JSON files under {data_root} failed to load.")

    full_dataset = ConcatDataset(datasets)
    print(f"\nTotal samples in concatenated dataset: {len(full_dataset)}")
    print(f"Total files skipped: {skipped}")
    return full_dataset


# =========================================================
# Env checking
# =========================================================

def input_check(args):
    # Env checking
    if args.env == "OvercookedMultiEnv-v0":
        if not args.env_config.get("layout_name"):
            raise EnvException(f"layout_name needed for {args.env}")
        elif args.env_config["layout_name"] not in LAYOUT_LIST:
            raise EnvException(
                f"{args.env_config['layout_name']} is not a valid layout"
            )

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
    pth_path = bc_policy_path.replace(".zip", ".pth")
    bc_policy = reconstruct_policy(pth_path, device=device)
    
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
        follower_agent.model.save(save_path)
    
        print(f"\nFollower agent saved to: {save_path}")
    
    return follower_agent


# =========================================================
# Main / CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="BC algorithm on human-human Overcooked trajectories.",
    )

    # Env args
    parser.add_argument(
        "--env",
        choices=ENV_LIST,
        default="OvercookedMultiEnv-v0",
        help="The environment to train in",
    )
    parser.add_argument(
        "--layout-name",
        type=str,
        default="unident_s",
        help="Overcooked layout name (used if not in JSON)",
    )
    parser.add_argument(
        "--framestack",
        "-f",
        type=int,
        default=1,
        help="Number of observations to stack (passed to generate_env)",
    )

    # Data args
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing human-human JSON trajectories.",
    )

    # Training args
    parser.add_argument(
        "--total-epochs",
        "-t",
        type=int,
        default=10,
        help="Number of epochs to run BC over the dataset.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Batch size for BC DataLoader.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="L2 weight on policy parameters.",
    )
    parser.add_argument(
        "--ent-weight",
        type=float,
        default=1e-3,
        help="Entropy bonus coefficient.",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        help="Device to run PyTorch on (e.g. 'cpu', 'cuda', or 'auto').",
    )

    # Save path
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save the trained SB3 policy (.zip).",
    )
    
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='Random seed')
    
    args = parser.parse_args()

    # Build env_config for generate_env
    args.env_config = {"layout_name": args.layout_name}
    args.record = None  # required by generate_env in this repo

    # Sanity check
    input_check(args)

    # 1) Build env (to get observation_space and action_space)
    env, alt_env = generate_env(args)
    print(f"Environment: {env}; Partner env: {alt_env}")

    if isinstance(env, SimultaneousEnv):
        TransitionsClass = trajsaver.SimultaneousTransitions
    else:
        TransitionsClass = trajsaver.TurnBasedTransitions
    # (We don't actually use TransitionsClass here, but keep for consistency.)

    # 2) Build dataset & dataloader from data_root
    full_dataset = build_dataset_from_root(args.data_root, args.layout_name)
    loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

    # Quick sanity check
    first_batch = next(iter(loader))
    print("Example batch obs shape:", first_batch["obs"].shape)
    print("Example batch acts shape:", first_batch["acts"].shape)

    # 3) Instantiate BC with SB3 policy
    bc = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy_class=FeedForward32Policy,
        policy_kwargs=None,
        expert_data=None,  # we set it next line
        optimizer_kwargs=dict(lr=3e-4),
        ent_weight=args.ent_weight,
        l2_weight=args.l2,
        device=args.device,
    )
    bc.set_expert_data_loader(loader)

    # 4) Train
    bc.train(n_epochs=args.total_epochs, log_interval=50)

    # 5) Save SB3 policy (ActorCriticPolicy)
    bc.save_policy(args.save_path)
    # follower_env = gym.make("OvercookedMultiEnv-v0", layout_name=args.layout_name)
    
    follower_env, _ = generate_env(args)
    
    follower_save_path = args.save_path.replace(".zip", "_follower.zip")
    
    train_follower_with_bc_partner(
        env=follower_env,
        bc_policy_path=args.save_path,
        layout_name=args.layout_name,
        total_timesteps=500000,
        save_path=follower_save_path,
        device=args.device,
        seed=args.seed
    )
    print(f"\nSaved BC policy to: {args.save_path}")
    print("You can later load it with:")
    print(f"    from {__name__} import FeedForward32Policy")
    print(f"    policy = FeedForward32Policy.load('{args.save_path}')")


if __name__ == "__main__":
    main()
