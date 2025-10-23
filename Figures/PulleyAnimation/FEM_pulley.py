import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure
from PIL import Image
import io

# Define initial conditions and connectivity matrix
initial_conditions = [[[-1.0, 0.0, 0.0], [0, 0, 0], 1, True],[[0.0, 0.0, 0.0], [0, 0, 0], 1, False],[[1.0, 0.0, 0.0], [0, 0, 0], 1, True]]
pulley_matrix = [[0,1,2,1000,0,2.5]]

Pulleys = FEM_structure(initial_conditions, pulley_matrix = pulley_matrix)

def fe_vector(force,angle):
    fe = np.zeros(18)
    fe[6+1] = np.sin(np.deg2rad(angle))*force
    fe[6] = np.cos(np.deg2rad(angle))*force
    return fe

def plot(Pulleys,fe):
    fig, ax = plt.subplots(figsize=(6, 4))  # Added figsize for better GIF quality
    coords = Pulleys.coords_current
    coords = np.asarray(coords).ravel()
    xs = coords[0::3]
    ys = coords[1::3]
    for i in range(3):
        if i == 1:
            color = "red"
        else:
            color = "black"
        ax.scatter(xs[i],ys[i],color=color,zorder=20)
    for i in range(2):
        ax.plot([xs[i],xs[i+1]],[ys[i],ys[i+1]],color="black")
    fx = fe[6]/100
    fy = fe[7]/100
    ax.quiver(xs[1], ys[1], fx, fy, angles='xy', scale_units='xy', scale=1, color='red')

    ax.set_aspect("equal")
    ax.set_xlim(-1.8,1.8)
    ax.set_ylim(-0.2,1.3)
    ax.axis('off')
    return ax,fig

frames = []
for angle in range(35, 148, 1):
    fe = fe_vector(50, angle)
    Pulleys.solve(fe=fe, tolerance=1e-2, max_iterations=5000, step_limit=0.3,
                  relax_init=0.5, relax_update=0.95, k_update=1,print_info=False)
    Pulleys.reinitialise()
    ax, fig = plot(Pulleys,fe)
    
    # Capture the figure as PIL Image
    fig.canvas.draw()
    
    # Convert to PIL Image using buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    # Add to frames list
    img = Image.open(buf)
    frames.append(img.copy())  # .copy() to ensure image persists after buffer closes
    
    # Clean up
    buf.close()
    plt.close(fig)

# Save as GIF
total_ms = 15000  # total GIF duration in milliseconds
reverse_frames = frames[:-1][::-1]
# freeze a few frames at the beginning and end
freeze_start = 20
freeze_end = 20

reverse_frames = frames[:-1][::-1]

if frames:
    start_dup = [frames[0].copy() for _ in range(min(freeze_start, len(frames)))]
    end_dup = [frames[-1].copy() for _ in range(min(freeze_end, len(frames)))]
    frames = start_dup + frames + end_dup

# recompute reverse sequence from the updated frames
frames = frames + reverse_frames

if frames:
    per_frame = max(1, int(total_ms / len(frames)))
    frames[0].save('pulley_animation.gif', 
                   save_all=True, 
                   append_images=frames[1:], 
                   duration=per_frame,  # per-frame duration so total ≈ total_ms
                   loop=0)

print(f"GIF saved with {len(frames)} frames")