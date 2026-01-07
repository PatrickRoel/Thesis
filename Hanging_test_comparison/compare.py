import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from kite_fem.saveload import load_fem_structure
from kite_fem.Plotting import plot_structure
from kite_fem.Functions import set_new_origin

os.chdir(os.path.dirname(os.path.abspath(__file__)))



def compare(filename: str, fig=None, axes=None):
    df = pd.read_csv('measurements/'+filename+'.csv')
    grouped_coords = {}
    groups = []
    for group_name, group_data in df.groupby('group'):
        coords = group_data[['y', 'x', 'z']].values
        grouped_coords[group_name] = coords*[-1,1,1]
        groups.append(group_name)

    # Create figure with 3 subplots if not provided
    if fig is None or axes is None:
        fig = plt.figure(figsize=(14, 3))
        # Create GridSpec with width ratios: Top (2), Side (3), Front (7)
        gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[7,2.5, 7, 0])
        axes = [
            fig.add_subplot(gs[0, 0]),  # XY view (Top)
            fig.add_subplot(gs[0, 2]),  # YZ view (Front) - moved to right
            fig.add_subplot(gs[0, 1]),  # XZ view (Side) - moved to middle
        ]
        # fig.suptitle(filename, fontsize=18)

    # Calculate origin based on the midpoint between strut3 and strut4
    strut3_coords = grouped_coords.get("strut3", np.array([]))
    strut4_coords = grouped_coords.get("strut4", np.array([]))
    le_coords = grouped_coords.get("LE", np.array([]))
    
    if len(strut3_coords) > 0 and len(strut4_coords) > 0 and len(le_coords) > 0:
        # Get the first point of each strut (where they connect to the LE)
        strut3_y = strut3_coords[0, 1]
        strut4_y = strut4_coords[0, 1]
        
        # Calculate the midpoint y-coordinate
        symmetry_y = (strut3_y + strut4_y) / 2
        
        # Find the LE point(s) closest to this symmetry plane
        le_y = le_coords[:, 1]
        y_distances = np.abs(le_y - symmetry_y)
        
        # Find the two closest points
        sorted_indices = np.argsort(y_distances)
        
        if len(le_coords) > 1:
            idx1, idx2 = sorted_indices[0], sorted_indices[1]
            y1, y2 = le_coords[idx1, 1], le_coords[idx2, 1]
            
            # If the two closest points straddle the symmetry plane, interpolate
            if (y1 <= symmetry_y <= y2) or (y2 <= symmetry_y <= y1):
                # Linear interpolation weight
                if abs(y2 - y1) > 1e-10:  # Avoid division by zero
                    weight = abs(symmetry_y - y1) / abs(y2 - y1)
                    origin = le_coords[idx1] * (1 - weight) + le_coords[idx2] * weight
                else:
                    origin = le_coords[sorted_indices[0]]
            else:
                # Use the closest point
                origin = le_coords[sorted_indices[0]]
        else:
            origin = le_coords[0]
            
        print(f"{filename}: strut3_y={strut3_y:.3f}, strut4_y={strut4_y:.3f}, symmetry_y={symmetry_y:.3f}, origin_y={origin[1]:.3f}")
    else:
        # Fallback: use midpoint of LE extent
        if len(le_coords) > 0:
            le_y = le_coords[:, 1]
            symmetry_y = (le_y.min() + le_y.max()) / 2
            y_distances = np.abs(le_y - symmetry_y)
            closest_idx = np.argmin(y_distances)
            origin = le_coords[closest_idx]
            print(f"{filename}: Fallback - using LE midpoint, symmetry_y={symmetry_y:.3f}")
        else:
            origin = np.array([0, 0, 0])
            print(f"{filename}: No LE coords found, using [0,0,0]")
    
    # Shift all coordinates by the origin
    for group_name in grouped_coords:
        grouped_coords[group_name] = grouped_coords[group_name] - origin



    results = load_fem_structure('Model results/'+filename+'.csv')
    set_new_origin(results,29)
    coords = results.coords_current.reshape(-1,3)
    TE_coords = coords[26]

    # Get strut 4 coordinates
    strut4_coords = grouped_coords["strut4"]

    strut4_first = strut4_coords[0]
    # Calculate rotation angle to align strut4_first with TE_coords
    # Project both points onto the xz plane (y is rotation axis)
    strut4_xz = np.array([strut4_first[0], strut4_first[2]])
    TE_xz = np.array([TE_coords[0], TE_coords[2]])
    
    # Calculate angles from y-axis
    angle_strut4 = np.arctan2(strut4_xz[0], strut4_xz[1])
    angle_TE = np.arctan2(TE_xz[0], TE_xz[1])
    
    # Rotation angle needed
    rotation_angle = angle_TE - angle_strut4
    # Create rotation matrix around y-axis
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])
    
    # Apply rotation to all grouped coordinates
    for group_name in grouped_coords:
        grouped_coords[group_name] = grouped_coords[group_name] @ rotation_matrix.T

    # Enforce symmetry through XZ plane (y=0)
    # Define symmetric pairs: strut0<->strut7, strut1<->strut6, strut2<->strut5, strut3<->strut4
    symmetric_pairs = [
        ('strut0', 'strut7'),
        ('strut1', 'strut6'),
        ('strut2', 'strut5'),
        ('strut3', 'strut4'),
    ]
    
    # Process symmetric pairs
    for group1, group2 in symmetric_pairs:
        if group1 in grouped_coords and group2 in grouped_coords:
            coords1 = grouped_coords[group1]
            coords2 = grouped_coords[group2]
            
            if len(coords1) == 0 or len(coords2) == 0:
                continue
            
            # Match nodes by index (assuming same number of nodes in each strut)
            min_len = min(len(coords1), len(coords2))
            
            symmetrized_coords1 = []
            symmetrized_coords2 = []
            
            for i in range(min_len):
                node1 = coords1[i]
                node2 = coords2[i]
                
                # Average the pair
                avg_node = (node1 + node2) / 2
                # Make it symmetric in y
                avg_y = (node1[1] - node2[1]) / 2
                
                # Create symmetric versions
                sym_node1 = avg_node.copy()
                sym_node1[1] = avg_y
                sym_node2 = avg_node.copy()
                sym_node2[1] = -avg_y
                
                symmetrized_coords1.append(sym_node1)
                symmetrized_coords2.append(sym_node2)
            
            # Update the groups
            grouped_coords[group1] = np.array(symmetrized_coords1)
            grouped_coords[group2] = np.array(symmetrized_coords2)
    
    # Process groups on the symmetry plane (LE and CAN)
    for group_name in ['LE', 'CAN']:
        if group_name in grouped_coords:
            coords = grouped_coords[group_name].copy()
            if len(coords) == 0:
                continue
            
            # For each node, find its mirror partner and average them in-place
            y_coords = coords[:, 1]
            
            # Track which nodes have been processed
            processed = np.zeros(len(coords), dtype=bool)
            
            for i in range(len(coords)):
                if processed[i]:
                    continue
                
                node = coords[i]
                # Mirror the node across y=0
                mirrored = node.copy()
                mirrored[1] = -mirrored[1]
                
                # Find the closest unprocessed node to the mirrored position
                distances = np.linalg.norm(coords - mirrored, axis=1)
                distances[processed] = np.inf  # Ignore already processed nodes
                distances[i] = np.inf  # Don't match with itself
                
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                threshold = 3  # meters
                if min_dist < threshold:
                    # Found a match - average them
                    partner_node = coords[min_idx]
                    avg_x = (node[0] + partner_node[0]) / 2
                    avg_y_mag = abs((node[1] - partner_node[1]) / 2)
                    avg_z = (node[2] + partner_node[2]) / 2
                    
                    # Update both nodes to their symmetric positions
                    # Keep the sign of y based on original position
                    coords[i] = [avg_x, avg_y_mag if node[1] > 0 else -avg_y_mag, avg_z]
                    coords[min_idx] = [avg_x, avg_y_mag if partner_node[1] > 0 else -avg_y_mag, avg_z]
                    
                    processed[i] = True
                    processed[min_idx] = True
                else:
                    # No match found - keep the node but make it symmetric
                    coords[i][1] = abs(node[1]) if node[1] > 0 else -abs(node[1])
                    processed[i] = True
            
            grouped_coords[group_name] = coords

    # Collect all coordinate data to determine common Z limits
    all_coords = np.vstack([coords for coords in grouped_coords.values() if len(coords) > 0])
    x_range = (all_coords[:, 0].min(), all_coords[:, 0].max())
    y_range = (all_coords[:, 1].min(), all_coords[:, 1].max())
    z_range = (-all_coords[:, 2].max(), -all_coords[:, 2].min())  # Flipped z
    
    # Add margins to vertical ranges (10% margin on each side for zoom out)
    margin_factor = 0.25
    y_span = y_range[1] - y_range[0]
    y_margin = y_span * margin_factor
    y_range = (y_range[0] - y_margin, y_range[1] + y_margin)
    
    # For side view Z-axis: add margin
    z_span = z_range[1] - z_range[0]
    z_margin = z_span * margin_factor
    z_range_with_margin = (z_range[0] - z_margin, z_range[1] + z_margin)
    z_span_with_margin = z_range_with_margin[1] - z_range_with_margin[0]
    z_center_with_margin = (z_range_with_margin[0] + z_range_with_margin[1]) / 2
    
    # For front view Z-axis: smaller margin (2% for tighter zoom)
    if filename == "load_case_1" or "load_case_6":
        front_margin_factor = 0.99
    else:
        front_margin_factor = 0.5
    z_margin_front = z_span * front_margin_factor
    z_range_front = (z_range[0] - z_margin_front, z_range[1] + z_margin_front)
    z_range_front = (-1,3)
    z_span_front = z_range_front[1] - z_range_front[0]
    z_center_front = (z_range_front[0] + z_range_front[1]) / 2
    
    # Plot on all axes
    for ax_idx, ax in enumerate(axes):
        # Plot measurements (use label only for first line in front view for legend)
        first_measurement = True
        for group_name in grouped_coords:
            if group_name != "CAN":
                xyz = grouped_coords[group_name]
                label = "Measurements" if (ax_idx == 1 and first_measurement) else None
                if ax_idx == 0:  # XY view (Top) - rotated: Y on x-axis, X on y-axis
                    ax.plot(xyz[:, 1], xyz[:, 0], color="black", linewidth=2.5)
                elif ax_idx == 1:  # YZ view (Front)
                    ax.plot(xyz[:, 1], -xyz[:, 2], color="black", linewidth=2.5, label=label)
                    first_measurement = False
                elif ax_idx == 2:  # XZ view (Side)
                    ax.plot(xyz[:, 0], -xyz[:, 2], color="black", linewidth=2.5)

        # Plot FEM structure on 2D views (use label only for first line in front view for legend)
        coords = results.coords_current.reshape(-1, 3)
        
        # Plot beam elements
        first_model = True
        for beam in results.beam_elements:
            n1, n2 = beam.beam.n1, beam.beam.n2
            n1_coords = coords[n1]
            n2_coords = coords[n2]
            
            label = "Model results" if (ax_idx == 1 and first_model) else None
            if ax_idx == 0:  # XY view (Top) - rotated: Y on x-axis, X on y-axis
                ax.plot([n1_coords[1], n2_coords[1]], 
                       [n1_coords[0], n2_coords[0]], color="red", linewidth=2.5)
            elif ax_idx == 1:  # YZ view (Front)
                ax.plot([n1_coords[1], n2_coords[1]], 
                       [-n1_coords[2], -n2_coords[2]], color="red", linewidth=2.5, label=label)
                first_model = False
            elif ax_idx == 2:  # XZ view (Side)
                ax.plot([n1_coords[0], n2_coords[0]], 
                       [-n1_coords[2], -n2_coords[2]], color="red", linewidth=2.5)
        
        # Set labels and aspect ratio
        if ax_idx == 0:  # XY view (Top) - rotated: Y on x-axis, X on y-axis
            ax.set_xlabel('y [m]', fontsize=18)
            ax.set_ylabel('x [m]', fontsize=18)
            # ax.set_title('Top View (XY)', fontsize=14, fontweight='bold')
            # Use actual data ranges (swapped)
            ax.set_xlim(-4.75,4.75)
            ax.set_ylim(-0.25,3)
        elif ax_idx == 1:  # YZ view (Front)
            ax.set_xlabel('y [m]', fontsize=18)
            ax.set_ylabel('z [m]', fontsize=18)
            # ax.set_title('Front View (YZ)', fontsize=14, fontweight='bold')
            # Use actual Y range, tighter Z range for zoomed in view
            ax.set_xlim(-4.75,4.75)
            ax.set_ylim(-0.25,3)
            # Add legend slightly above center
            ax.legend(loc='center', bbox_to_anchor=(0.5, 0.6), fontsize=14, framealpha=0.9)
        elif ax_idx == 2:  # XZ view (Side)
            ax.set_xlabel('x [m]', fontsize=18)
            ax.set_ylabel('z [m]', fontsize=18)
            # ax.set_title('Side View (XZ)', fontsize=14, fontweight='bold')
            # Use actual X range, common Z range with margin for vertical alignment
            ax.set_xlim(-0.25,3)
            ax.set_ylim(-0.25,3)
        # Y-axis is fixed, X-axis adjusts to maintain equal aspect (1:1)
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(axis='both', which='major', labelsize=18)
        # Set integer ticks only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    return fig, axes

for i in range(1,11):
    fig, axes = compare("load_case_"+str(i))
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig(f'figures/load_case_{i}.pdf', dpi=300, bbox_inches='tight')

plt.show()