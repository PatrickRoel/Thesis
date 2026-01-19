import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import plot_structure, plot_convergence
import csv
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define initial conditions and connectivity matrix
initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, True],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[3.0, 0.0, 0.0], [0, 0, 0], 1, True],[[0.5, -1.0, 0.0], [0, 0, 0], 1, False],[[2.5, -1.0, 0.0], [0, 0, 0], 1, False],[[1.5, -1.5, 0.0], [0, 0, 0], 1, False]]
l0 = 2.2
pulley_matrix = [[0,4,1,5000,0,l0],[2,5,3,5000,0,l0],[4,6,5,5000,0,l0]]

Pulleys = FEM_structure(initial_conditions, pulley_matrix = pulley_matrix)
fe = np.zeros(Pulleys.N)
fx = np.tan(np.deg2rad(15))*100
fy = -100
fe[(Pulleys.num_nodes-1)*6+1] = -100
fe[(Pulleys.num_nodes-1)*6] = np.tan(np.deg2rad(15))*100

ax1,fig1 = plot_structure(Pulleys,fe=fe, plot_external_forces=True,plot_displacements=False,fe_magnitude=0.35,plot_2d=True,e_colors = ['red', 'blue', 'black', 'green'] ,v_colors = ['red', 'darkgreen'],n_scale = [35,35,35])
Pulleys.solve(fe = fe, tolerance=1e-3, convergence_criteria="residual", max_iterations=5000, step_limit=0.3, relax_init=0.5,relax_update=0.95, k_update=1)
ax2,fig2 = plot_structure(Pulleys, fe=fe, plot_external_forces=True,fe_magnitude=0.35,plot_2d=True,  e_colors = ['red', 'blue', 'black', 'green'] ,v_colors = ['red', 'darkgreen'],n_scale = [35,35,35])


coords = Pulleys.coords_current
coords = coords.reshape(-1, 3)
connections = [[0,4],[1,6],[2,5]]
point1 = coords[4]+(coords[4]-coords[6])*0.67
point2 = coords[6]+(np.array([fx,fy,0]))*-0.0177
point3 = coords[5]+(coords[5]-coords[6])*1.18
points = [point1,point2,point3]

for connection in connections:
    xyz1 = points[connection[0]]
    xyz2 = coords[connection[1]]
    ax2.plot([xyz1[0], xyz2[0]], [xyz1[1], xyz2[1]], '--', linewidth=1, color="black")


def plot_arc(center,point1,point2,scale):
    # Calculate vectors from center to endpoints
    vec1 = np.array(point1) - np.array(center)
    vec2 = np.array(point2) - np.array(center)

    # Calculate angles
    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])

    # Ensure we draw the shorter arc
    if abs(angle2 - angle1) > np.pi:
        if angle2 > angle1:
            angle1 += 2*np.pi
        else:
            angle2 += 2*np.pi

    # Calculate radius (use average of distances)
    radius = (np.linalg.norm(vec1) + np.linalg.norm(vec2)) / scale

    # Create arc
    angles = np.linspace(angle1, angle2, 50)
    arc_x = center[0] + radius * np.cos(angles)
    arc_y = center[1] + radius * np.sin(angles)

    ax2.plot(arc_x, arc_y, '-', linewidth=1, color="black")

# Calculate arc parameters
centers = [coords[4],coords[4],coords[6],coords[6],coords[5],coords[5]]
points1 = [coords[0],points[0],coords[4],points[1],coords[2],points[2]]
points2 = [points[0],coords[1],points[1],coords[5],points[2],coords[3]]
scales = [5,4,6,5.5,5,5]
for center,point1,point2,scale in zip(centers,points1,points2,scales):
    plot_arc(center,point1,point2,scale)

# ax1.set_title("Initial Configuration")
# ax2.set_title("Deformed Configuration")
fontsize = 18
ax2.text(0.75, -0.33, r'$\alpha_1$', fontsize=fontsize, ha='center', va='center')
ax2.text(0.98, -0.25, r'$\alpha_2$', fontsize=fontsize, ha='center', va='center')
ax2.text(1.8, -1.18, r'$\beta_1$', fontsize=fontsize, ha='center', va='center')
ax2.text(2.1, -1.15, r'$\beta_2$', fontsize=fontsize, ha='center', va='center')
ax2.text(2.25, -0.48, r'$\gamma_1$', fontsize=fontsize, ha='center', va='center')
ax2.text(2.5, -0.46, r'$\gamma_2$', fontsize=fontsize, ha='center', va='center')


for i,coord in enumerate(coords):
    if i<3:
        ax2.text(coord[0]+0.075,coord[1]+0.075,f'$\\text{{n}}_{{{i+1}}}$',fontsize=fontsize, ha='center', va='center' )
    elif i==3:
        ax2.text(coord[0]-0.115,coord[1]+0.075,f'$\\text{{n}}_{{{i+1}}}$',fontsize=fontsize, ha='center', va='center' )
    else:
        ax2.text(coord[0]+0.126,coord[1],f'$\\text{{n}}_{{{i+1}}}$',fontsize=fontsize, ha='center', va='center' )

coords_init = Pulleys.coords_init
coords_init = coords_init.reshape(-1,3)
for i,coord in enumerate(coords_init):
    if i<3:
        ax1.text(coord[0]+0.095,coord[1]+0.075,f'$\\text{{n}}_{{{i+1}}}$',fontsize=fontsize, ha='center', va='center' )
    elif i==3:
        ax1.text(coord[0]-0.115,coord[1]+0.075,f'$\\text{{n}}_{{{i+1}}}$',fontsize=fontsize, ha='center', va='center' )
    elif i+1 ==7:
        ax1.text(coord[0]+0.14,coord[1]-0.075,f'$\\text{{n}}_{{{i+1}}}$',fontsize=fontsize, ha='center', va='center' )

    else:
        ax1.text(coord[0]+0.14,coord[1]+0.05,f'$\\text{{n}}_{{{i+1}}}$',fontsize=fontsize, ha='center', va='center' )

# midpoint = [1.5,-0.05,0]
# vect = [0.5,0,0]
# ax1.quiver(midpoint[0],midpoint[1],vect[0],0,color="black",width=0.005,scale=1.1,scale_units='xy')
# ax1.quiver(midpoint[0],midpoint[1],-vect[0],0,color="black",width=0.005,scale=1.1,scale_units='xy')

# ax1.text(1.5, 0.025, r'$1$m', fontsize=10, ha='center', va='center')

ax1.set_ylim(-2.15, 0.15)
ax2.set_ylim(-2.15, 0.15)
ax1.grid()
ax2.grid()

# Set larger font sizes for axis labels and ticks
for ax in [ax1, ax2]:
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

# ax1.legend()

ax2.legend(fontsize=14)

def calculate_angle(line1, line2):
    """
    Calculate the angle between two lines.
    
    Args:
        line1: [point1, point2] - two points defining the first line
        line2: [point2, point3] - two points defining the second line
    
    Returns:
        angle in radians between the two lines
    """
    # Extract points
    p1, p2 = line1
    p2_check, p3 = line2
    
    # Convert to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Calculate direction vectors
    vec1 = p2 - p1
    vec2 = p3 - p2
    
    # Calculate angle using dot product
    dot_product = np.dot(vec1, vec2)
    magnitudes = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    # Handle zero magnitude case
    if magnitudes == 0:
        return 0
    
    # Calculate angle
    cos_angle = dot_product / magnitudes
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle = np.rad2deg(angle)
    return angle





y = -5
x = np.tan(np.deg2rad(15))*5

point_force = coords[6]+[x,y,0]

lines1 = [[coords[0],coords[4]],[coords[1],coords[4]],[coords[4],coords[6]],[coords[5],coords[6]],[coords[2],coords[5]],[coords[3],coords[5]]]
lines2 = [[coords[4],coords[6]],[coords[4],coords[6]],[coords[6],point_force],[coords[6],point_force],[coords[5],coords[6]],[coords[5],coords[6]]]
angles = []
for line1,line2 in zip(lines1,lines2):
    angles.append(calculate_angle(line1,line2))

# Save angles to CSV
headers = ['alpha1', 'alpha2', 'beta1', 'beta2', 'gamma1', 'gamma2']
with open('angles.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    writer.writerow([f'{angle:.2f}' for angle in angles])

# ax1.set_axis_off()
# ax2.set_axis_off()





    
plt.show()
fig1.savefig('pulley_initial_configuration.pdf', bbox_inches='tight', dpi=300)
fig2.savefig('pulley_deformed_configuration.pdf', bbox_inches='tight', dpi=300)