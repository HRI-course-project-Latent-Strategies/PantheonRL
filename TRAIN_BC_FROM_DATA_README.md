# Training Behavior Cloning from Collected Data

This guide explains how to use `train_bc_from_collected_data.py` to:
1. Train a BC agent to clone Player 1's (leader's) behavior from collected trajectories
2. Train Player 2 (follower) to work with the BC agent as a fixed partner

## Quick Start

### Step 1: Train BC Agent to Clone Player 1 (Leader)

```bash
python train_bc_from_collected_data.py p1nate_adrian \
    --layout-name asymmetric_advantages \
    --bc-epochs 100 \
    --bc-save models/bc_leader.pt
```

This will:
- Load all `p1*.npy` files from the `p1nate_adrian` directory
- Extract Player 1's (leader's) transitions from each file
- Combine all transitions
- Train a BC agent to clone Player 1's behavior
- Save the BC policy to `models/bc_leader.pt`

### Step 2: Train Player 2 (Follower) with BC Leader

```bash
python train_bc_from_collected_data.py p1nate_adrian \
    --layout-name asymmetric_advantages \
    --bc-epochs 100 \
    --bc-save models/bc_leader.pt \
    --train-follower \
    --follower-timesteps 500000 \
    --follower-save models/follower_with_bc_leader.zip
```

This will:
- Train the BC agent (as in Step 1)
- Load the BC agent as a fixed partner
- Train Player 2 (follower) using PPO to work with the BC leader
- Save the follower agent to `models/follower_with_bc_leader.zip`

## Complete Example

```bash
# Train BC agent from all Player 1 trajectories
python train_bc_from_collected_data.py p1nate_adrian \
    --layout-name asymmetric_advantages \
    --pattern "p1*.npy" \
    --bc-epochs 150 \
    --bc-l2 0.0001 \
    --bc-save models/bc_p1_leader.pt \
    --train-follower \
    --follower-timesteps 1000000 \
    --follower-save models/p2_follower_with_bc_p1.zip \
    --seed 42
```

## Arguments

### Required Arguments
- `data_dir`: Directory containing trajectory files (.npy)

### Optional Arguments

#### Layout Configuration
- `--layout-name`: Layout name (default: `asymmetric_advantages`)
  - Will be automatically translated to Python layout name (e.g., `asymmetric_advantages` → `unident_s`)

#### BC Training
- `--pattern`: Glob pattern to match trajectory files (default: `p1*.npy`)
- `--bc-epochs`: Number of epochs for BC training (default: 100)
- `--bc-l2`: L2 regularization weight (default: 0.0)
- `--bc-save`: Path to save BC policy (default: `models/bc_leader.pt`)

#### Follower Training
- `--train-follower`: Flag to enable follower training after BC training
- `--follower-timesteps`: Total timesteps for follower training (default: 500000)
- `--follower-save`: Path to save follower agent (default: `models/follower_with_bc_leader.zip`)

#### General
- `--device`: Device to run on (default: `auto`, options: `auto`, `cpu`, `cuda`)
- `--seed`: Random seed for reproducibility

## Understanding the Data

The `p1nate_adrian` directory contains:
- **Player 1 files**: `p1*.npy` - Player 1 is the leader
- **Player 2 files**: `p2*.npy` - Player 2 is the follower

Each `.npy` file contains a `SimultaneousTransitions` object with:
- Ego observations and actions (Player 1)
- Alt observations and actions (Player 2)
- Flags indicating episode boundaries

## Workflow

1. **Data Collection**: Trajectories are collected with Player 1 as leader and Player 2 as follower
2. **BC Training**: Player 1's behavior is cloned using behavior cloning
3. **Follower Training**: Player 2 is trained using RL (PPO) to work with the BC-cloned Player 1

## Layout Name Translation

The script automatically translates layout names:
- `asymmetric_advantages` → `unident_s`
- `cramped_room` → `simple`
- `coordination_ring` → `random1`
- `forced_coordination` → `random0`
- `counter_circuit` → `random3`

If your layout name is already a Python layout name (e.g., `unident_s`), it will be used as-is.

## Using the Trained Agents

### Using the BC Leader Agent

```python
from pantheonrl.algos.bc import reconstruct_policy
from pantheonrl.common.agents import StaticPolicyAgent

# Load BC policy
bc_policy = reconstruct_policy("models/bc_leader.pt", device="auto")

# Wrap as static agent
bc_agent = StaticPolicyAgent(bc_policy)

# Use in environment
env.add_partner_agent(bc_agent)
```

### Using the Trained Follower Agent

```python
from stable_baselines3 import PPO
from pantheonrl.common.agents import OnPolicyAgent

# Load follower agent
follower_model = PPO.load("models/follower_with_bc_leader.zip")
follower_agent = OnPolicyAgent(follower_model)

# Use in environment
env.add_partner_agent(follower_agent)
```

Or use with `trainer.py`:

```bash
python trainer.py OvercookedMultiEnv-v0 PPO FIXED \
    --env-config '{"layout_name":"unident_s"}' \
    --alt-config '{"type":"BC","location":"models/bc_leader.pt"}' \
    --ego-save models/follower_with_bc.zip
```

## Troubleshooting

### No trajectory files found
- Check that the `data_dir` path is correct
- Verify the pattern matches your file naming (default: `p1*.npy`)
- Ensure `.npy` files exist in the directory

### Shape mismatch errors
- Make sure the layout name matches the layout used when trajectories were collected
- Verify the environment setup matches the original data collection setup

### Out of memory
- Reduce the number of trajectory files loaded at once
- Use a smaller batch size (modify BC class parameters)
- Train in multiple stages

### Poor BC performance
- Increase number of epochs (`--bc-epochs`)
- Add L2 regularization (`--bc-l2 0.0001`)
- Ensure you have enough diverse trajectories
- Check that Player 1's behavior is consistent across trajectories

## Example Output

```
============================================================
Step 1: Loading and Combining Trajectories
============================================================
Data directory: p1nate_adrian
Pattern: p1*.npy
Layout: asymmetric_advantages
Layout name translated: 'asymmetric_advantages' -> 'unident_s'
Found 54 trajectory files matching pattern 'p1*.npy'
Loading 1/54: p1adrian_nate_2025-11-20-16-02-31-.npy
  - Loaded 400 transitions
Loading 2/54: p1adrian_nate_2025-11-20-16-03-36-.npy
  - Loaded 395 transitions
...
Total combined transitions: 21600

============================================================
Step 2: Training BC Agent
============================================================
Expert data size: 21600 transitions
Training epochs: 100
L2 weight: 0.0

Starting training...
batch: 1/675  epoch: 0/100
...
BC policy saved to: models/bc_leader.pt

============================================================
Step 3: Training Follower Agent
============================================================
Loading BC policy from: models/bc_leader.pt
BC agent added as fixed partner (Player 1 / Leader)
Follower agent (Player 2) created
Training for 500000 timesteps...
...
Follower agent saved to: models/follower_with_bc_leader.zip

============================================================
Training Complete!
============================================================
```

