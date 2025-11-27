#!/usr/bin/env python3
"""
Script to visualize agent position heatmaps from trajectory JSON files.
Creates a heatmap overlay on the gridworld layout showing time spent at each position.

Usage:
    # Basic usage (will try to extract layout name from JSON)
    python visualize_heatmap.py trajectory.json
    
    # Specify layout name explicitly
    python visualize_heatmap.py trajectory.json --layout_name cramped_room
    
    # Save to file instead of displaying
    python visualize_heatmap.py trajectory.json --output heatmap.png
    
    # Create combined heatmap for both players
    python visualize_heatmap.py trajectory.json --combined --output combined_heatmap.png
    
    # Create separate heatmaps for each player (default)
    python visualize_heatmap.py trajectory.json --output heatmap
    
    # The script handles JSON files saved from app.py (with 'traj' wrapper) 
    # as well as direct trajectory files (with 'ep_states' or 'ep_observations')
"""

import json
import argparse
import sys
import os
import numpy as np
import matplotlib
# Try to use a backend that works, fall back to Agg if display not available
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# Add paths for imports - need to add the parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
overcooked_ai_path = os.path.join(project_root, 'overcookedgym/human_aware_rl/overcooked_ai')
if os.path.exists(overcooked_ai_path):
    sys.path.insert(0, overcooked_ai_path)
sys.path.insert(0, project_root)

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcookedgym.overcooked_utils import NAME_TRANSLATION


def parse_state(state):
    """Parse state which might be a dict or JSON string."""
    if isinstance(state, str):
        return json.loads(state)
    return state


def extract_positions_from_json(json_data):
    """
    Extract position data for both players from JSON trajectory.
    
    Returns:
        tuple: (player0_positions, player1_positions) where each is a list of (x, y) tuples
    """
    player0_positions = []
    player1_positions = []
    
    # Handle both 'ep_states' and 'ep_observations' formats
    # Also handle cases where data is wrapped in 'traj' key (from app.py)
    if 'traj' in json_data:
        traj_data = json_data['traj']
        episodes = traj_data.get('ep_states', traj_data.get('ep_observations', []))
    else:
        episodes = json_data.get('ep_states', json_data.get('ep_observations', []))
    
    # Iterate through all episodes
    for episode in episodes:
        for state in episode:
            state_dict = parse_state(state)
            
            if 'players' in state_dict and len(state_dict['players']) >= 2:
                # Extract positions for both players
                # Position is stored as [x, y] in the JSON
                pos0 = state_dict['players'][0]['position']
                pos1 = state_dict['players'][1]['position']
                
                # Convert to tuple, handling both list and tuple formats
                player0_pos = tuple(pos0) if isinstance(pos0, (list, tuple)) else pos0
                player1_pos = tuple(pos1) if isinstance(pos1, (list, tuple)) else pos1
                
                player0_positions.append(player0_pos)
                player1_positions.append(player1_pos)
    
    return player0_positions, player1_positions


def count_position_visits(positions):
    """
    Count how many times each position was visited.
    
    Args:
        positions: List of (x, y) position tuples
        
    Returns:
        dict: Dictionary mapping (x, y) -> count
    """
    position_counts = defaultdict(int)
    for pos in positions:
        position_counts[pos] += 1
    return position_counts


def get_grid_dimensions(layout_name):
    """Get grid dimensions from layout."""
    mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name)
    return mdp.height, mdp.width, mdp.terrain_mtx


def create_heatmap(position_counts, height, width, terrain_mtx, player_name="Player", cmap='YlOrRd'):
    """
    Create a heatmap visualization of position visits.
    
    Args:
        position_counts: Dictionary mapping (x, y) -> count
        height: Grid height
        width: Grid width
        terrain_mtx: Terrain matrix for the layout
        player_name: Name of the player for the title
        cmap: Colormap to use
    """
    # Create a 2D array for the heatmap
    # Note: terrain_mtx is indexed as terrain_mtx[y][x], and positions are (x, y)
    heatmap_data = np.zeros((height, width))
    
    # Fill in the position counts
    for (x, y), count in position_counts.items():
        # Ensure coordinates are within bounds
        if 0 <= x < width and 0 <= y < height:
            heatmap_data[y, x] = count
        else:
            print(f"WARNING: Position ({x}, {y}) is out of bounds for grid {width}x{height}")
    
    # Normalize for visualization (0-1 range)
    max_count = heatmap_data.max() if heatmap_data.max() > 0 else 1
    heatmap_normalized = heatmap_data / max_count
    
    print(f"  Heatmap data: min={heatmap_data.min()}, max={heatmap_data.max()}, sum={heatmap_data.sum()}")
    print(f"  Non-zero cells: {np.count_nonzero(heatmap_data)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, width*1.5), max(10, height*1.5)))
    
    # Display heatmap using imshow
    if max_count > 0:
        im = ax.imshow(heatmap_normalized, cmap=cmap, alpha=0.7, interpolation='nearest', 
                      origin='upper', vmin=0, vmax=1, aspect='equal')
    else:
        # Create empty heatmap if no data
        im = ax.imshow(np.zeros((height, width)), cmap=cmap, alpha=0.3, 
                      origin='upper', vmin=0, vmax=1, aspect='equal')
    
    # Overlay terrain
    terrain_colors = {
        'X': '#8B4513',  # Brown for walls
        ' ': '#F5F5DC',  # Beige for empty
        'P': '#D3D3D3',  # Light gray for counter
        'O': '#FFD700',  # Gold for onion
        'T': '#FF6347',  # Tomato red
        'D': '#32CD32',  # Lime green for delivery
        'S': '#87CEEB',  # Sky blue for serving
    }
    
    # Draw terrain grid and labels
    for y in range(height):
        for x in range(width):
            terrain_char = terrain_mtx[y][x] if y < len(terrain_mtx) and x < len(terrain_mtx[y]) else ' '
            color = terrain_colors.get(terrain_char, '#FFFFFF')
            
            # Draw rectangle for terrain (semi-transparent overlay)
            if terrain_char != ' ':
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                        linewidth=2, edgecolor='black', 
                                        facecolor=color, alpha=0.4, zorder=2)
                ax.add_patch(rect)
                
                # Add terrain label
                ax.text(x, y, terrain_char, ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                       zorder=3)
    
    # Add position count annotations
    for (x, y), count in position_counts.items():
        if 0 <= x < width and 0 <= y < height and count > 0:
            # Show count
            text_color = 'white' if max_count > 0 and heatmap_normalized[y, x] > 0.5 else 'black'
            ax.text(x, y, f'{count}', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=text_color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=1.5),
                   zorder=4)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if max_count > 0:
        cbar.set_label('Normalized Visit Count', rotation=270, labelpad=20)
    else:
        cbar.set_label('No Data', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('X Position (Column)', fontsize=12)
    ax.set_ylabel('Y Position (Row)', fontsize=12)
    ax.set_title(f'{player_name} Position Heatmap\n(Time Spent at Each Grid Cell)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, alpha=0.5, linestyle='--', linewidth=1, color='black', zorder=1)
    
    plt.tight_layout()
    return fig, ax


def visualize_heatmaps(json_file, layout_name=None, output_file=None, separate_players=True):
    """
    Main function to visualize heatmaps from JSON trajectory file.
    
    Args:
        json_file: Path to JSON trajectory file
        layout_name: Layout name (if None, tries to extract from JSON)
        output_file: Output file path (if None, displays interactively)
        separate_players: If True, create separate heatmaps for each player
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Get layout name
    if layout_name is None:
        # Try to get from top level
        if 'layout_name' in json_data:
            server_layout_name = json_data['layout_name']
            layout_name = NAME_TRANSLATION.get(server_layout_name, server_layout_name)
        # Try to get from mdp_params
        elif 'mdp_params' in json_data and len(json_data['mdp_params']) > 0:
            mdp_params = json_data['mdp_params'][0] if isinstance(json_data['mdp_params'], list) else json_data['mdp_params']
            if 'layout_name' in mdp_params:
                layout_name = mdp_params['layout_name']
        # Try to get from traj wrapper
        elif 'traj' in json_data and 'layout_name' in json_data:
            server_layout_name = json_data['layout_name']
            layout_name = NAME_TRANSLATION.get(server_layout_name, server_layout_name)
        else:
            raise ValueError("Layout name not found in JSON and not provided as argument. "
                           "Please provide --layout_name argument.")
    
    # Extract positions
    player0_positions, player1_positions = extract_positions_from_json(json_data)
    
    print(f"Extracted {len(player0_positions)} positions for Player 0")
    print(f"Extracted {len(player1_positions)} positions for Player 1")
    
    if len(player0_positions) == 0 and len(player1_positions) == 0:
        print("WARNING: No positions extracted! Check JSON structure.")
        print(f"JSON keys: {list(json_data.keys())}")
        if 'traj' in json_data:
            print(f"traj keys: {list(json_data['traj'].keys())}")
        return
    
    # Get grid dimensions
    height, width, terrain_mtx = get_grid_dimensions(layout_name)
    print(f"Grid dimensions: {height} x {width}")
    
    # Count positions
    player0_counts = count_position_visits(player0_positions)
    player1_counts = count_position_visits(player1_positions)
    
    print(f"Player 0 visited {len(player0_counts)} unique positions")
    print(f"Player 1 visited {len(player1_counts)} unique positions")
    if len(player0_counts) > 0:
        print(f"Player 0 position range: x=[{min(x for x,y in player0_counts.keys())}, {max(x for x,y in player0_counts.keys())}], y=[{min(y for x,y in player0_counts.keys())}, {max(y for x,y in player0_counts.keys())}]")
    if len(player1_counts) > 0:
        print(f"Player 1 position range: x=[{min(x for x,y in player1_counts.keys())}, {max(x for x,y in player1_counts.keys())}], y=[{min(y for x,y in player1_counts.keys())}, {max(y for x,y in player1_counts.keys())}]")
    
    # Determine output file names
    if output_file:
        base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
        output_p0 = f"{base_name}_player0.png"
        output_p1 = f"{base_name}_player1.png"
        output_combined = output_file if '.' in output_file else f"{output_file}.png"
    else:
        # Auto-generate output file names from input JSON filename
        json_basename = os.path.splitext(os.path.basename(json_file))[0]
        output_p0 = f"{json_basename}_player0_heatmap.png"
        output_p1 = f"{json_basename}_player1_heatmap.png"
        output_combined = f"{json_basename}_combined_heatmap.png"
        print(f"No --output specified, will save to: {output_p0}, {output_p1}")
    
    # Create visualizations
    if separate_players:
        # Create separate heatmaps for each player
        print("\nCreating Player 0 heatmap...")
        fig0, ax0 = create_heatmap(player0_counts, height, width, terrain_mtx, 
                                   player_name="Player 0", cmap='Reds')
        plt.savefig(output_p0, dpi=150, bbox_inches='tight')
        print(f"✓ Saved Player 0 heatmap to {output_p0}")
        plt.close(fig0)
        
        print("Creating Player 1 heatmap...")
        fig1, ax1 = create_heatmap(player1_counts, height, width, terrain_mtx, 
                                   player_name="Player 1", cmap='Blues')
        plt.savefig(output_p1, dpi=150, bbox_inches='tight')
        print(f"✓ Saved Player 1 heatmap to {output_p1}")
        plt.close(fig1)
    else:
        # Create combined heatmap
        print("\nCreating combined heatmap...")
        combined_counts = defaultdict(int)
        for pos, count in player0_counts.items():
            combined_counts[pos] += count
        for pos, count in player1_counts.items():
            combined_counts[pos] += count
        
        fig, ax = create_heatmap(combined_counts, height, width, terrain_mtx, 
                                player_name="Both Players", cmap='YlOrRd')
        plt.savefig(output_combined, dpi=150, bbox_inches='tight')
        print(f"✓ Saved combined heatmap to {output_combined}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize agent position heatmaps from trajectory JSON files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('json_file', type=str, help='Path to JSON trajectory file')
    parser.add_argument('--layout_name', type=str, default=None,
                       help='Layout name (if not provided, will try to extract from JSON)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (if not provided, displays interactively)')
    parser.add_argument('--combined', action='store_true',
                       help='Create a single combined heatmap instead of separate ones for each player')
    
    args = parser.parse_args()
    
    visualize_heatmaps(
        args.json_file,
        layout_name=args.layout_name,
        output_file=args.output,
        separate_players=not args.combined
    )


if __name__ == '__main__':
    main()

