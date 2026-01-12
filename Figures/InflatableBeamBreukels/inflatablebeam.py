import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
script_dir = os.path.dirname(os.path.abspath(__file__))

#bending
def F_inflatablebeam(p, r, v):
    # Coefficients
    C1 = 6582.82
    C2 = -272.43
    C3 = 40852.38
    C4 = 14.31
    C5 = 271865251.42
    C6 = 215.93
    C7 = 14021.79
    C8 = -589.05
    
    # Numerator and denominator
    denom = (C1 * r + C2) * p**2 + (C3 * r**3 + C4)
    numer = (C5 * r**5 + C6) * p + (C7 * r + C8)
    
    # Formula
    result = denom * (1 - np.exp(-(numer / denom) * v))
    return result

def v_collapse(p,r):
    # Coefficients
    C9 = 322.55
    C10 = 0.0239
    C11 = 5.3833
    C12 = 0.0461
    v = (C9*r**4+C10)*p+C11*r**2+C12
    return v

def torsion(p,r,phi):
    C13 = 1467
    C14 = 40.908
    C15 = -191.8
    C16 = 47.406
    C17 = -17703
    C18 = 358.05
    C19 = 0.0918
    c1 = ((C13*r+C14)*p+(C15*r+C16))
    c2 = ((C17*r**4)*np.log(p)+(C18*r**3+C19))
    T = c1*np.arctan(c2*np.deg2rad(phi))
    return T

def plotinflatablebeam(p,d,ls,ax):
    r = d/2
    v_max = v_collapse(p,r)
    v = np.linspace(0, v_max, 100)
    v_step = v[1] - v[0]
    F = F_inflatablebeam(p,r,v)
    v = np.append(v, v_max+v_step)
    F = np.append(F, 0)
    ax.plot(v*1000,F,color="black",linestyle=ls,linewidth=1.5)
    phi_max = 140
    phi = np.linspace(0, phi_max, 100)
    T = torsion(p, r, phi)
    ax.plot(phi, T,color="red",linestyle=ls,linewidth=1.5)
    return ax

p_lst = [0.3,0.5] #bar
d_lst = [0.16,0.16] #m
linestyles = ['-', '--','-','--']
fig, ax = plt.subplots(figsize=(4,4))


for p,d,linestyle in zip(p_lst,d_lst,linestyles):
    ax = plotinflatablebeam(p,d,linestyle,ax)


for mode in ["bending","torsion"]:
    if mode == "bending":
        color = "black"
    else:
        color = "red"
    for p,d,linestyle in zip(p_lst,d_lst,linestyles):
        plt.plot([], [], color=color, linestyle=linestyle, label=f"p={p}bar, d={d*100}cm, {mode}")



import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import plot_structure

def instiantiate(d,p):
    length  = 1  
    elements = 1
    l0 = length/elements
    initital_conditions = []
    for i in range(elements+1):
        initital_conditions.append([[i*length/elements, 0.0, 0.0], [0, 0, 0], 1, True if i==0 else False])
    beam_matrix = []
    for i in range(elements):
        beam_matrix.append([i, i+1, d, p,l0])
    inflatable_beam = FEM_structure(initital_conditions, beam_matrix=beam_matrix)
    return inflatable_beam

def solve_tip_load(inflatable_beam,tip_load):
    fe = np.zeros(inflatable_beam.N)
    fe[1::6][-1] = -tip_load
    inflatable_beam.solve(        fe=fe,
            max_iterations=1000,
            tolerance=0.00001,
            step_limit=0.25,
            relax_init=0.5,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25
            )
    deflection = -inflatable_beam.coords_current[-2]*1000
    collapsed = False
    for beam in inflatable_beam.beam_elements:
        if beam.collapsed == True:
            collapsed = True
    inflatable_beam.reset()

    return deflection, collapsed

def solve_tip_moment(inflatable_beam,tip_moment):
    fe = np.zeros(inflatable_beam.N)
    fe[3::6][-1] = -tip_moment
    inflatable_beam.solve(        fe=fe,
            max_iterations=1000,
            tolerance=0.00001,
            step_limit=0.5,
            relax_init=1,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25
            )
    rotation = -np.rad2deg(inflatable_beam.coords_rotations_current[-3])
    inflatable_beam.reset()
    return rotation

pressures = p_lst
diameters = d_lst
inflatable_beams = []

for pressure,diameter in zip(pressures,diameters):
    inflatable_beam = instiantiate(diameter,pressure)
    inflatable_beams.append(inflatable_beam)

tip_loads = np.arange(5,95,10)
tip_moments = np.arange(5,120,10)

for inflatable_beam in inflatable_beams:
    deflections = []
    rotations = []
    collapsed = []

    for tip_load in tip_loads:
        deflection,collapse = solve_tip_load(inflatable_beam,tip_load)
        deflections.append(deflection)
        collapsed.append(collapse)
    for tip_moment in tip_moments:
        rotation = solve_tip_moment(inflatable_beam,tip_moment)
        rotations.append(rotation)
    ax.scatter(rotations,tip_moments,marker="x",zorder=20,color="black")
    ax.scatter(deflections,tip_loads,marker="+",zorder=20,color="red")



    

ax.legend(loc='upper left', fontsize=8)
ax.set_xlabel("Deflection [mm] / Deflection angle [deg]")
ax.set_ylabel("Tip force [N] / Tip moment [Nm]")
# ax.minorticks_on() 
ax.grid(which="major",color="grey")
# ax.grid(which="minor",color="lightgrey")
ax.set_xlim(0,120)
ax.set_ylim(0,120)
fig.tight_layout()

fig.savefig(os.path.join(script_dir, 'Inflatablebeamdeflections.pdf'))

# Read the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "beam_comparison_results.csv")

df = pd.read_csv(csv_path)

# Create the plot
fig2, ax2 = plt.subplots(figsize=(4,4))
ax2.grid()
ax2.scatter(df['Element Length (m)'], df['Normalised error (-)'], marker='+',color = "black",zorder=20)
ax2.set_xlabel('Element length (m)')
ax2.set_ylabel('Deflection Error / Element length (-)')
# plt.title('Error vs Element Length')
fig2.tight_layout()

# Create composite figure with fig and fig2 side by side
fig_composite = plt.figure(figsize=(8, 4))
gs = fig_composite.add_gridspec(1, 2)

# Copy ax content to first subplot
ax_left = fig_composite.add_subplot(gs[0, 0])
for line in ax.get_lines():
    ax_left.plot(line.get_xdata(), line.get_ydata(), 
                 color=line.get_color(), 
                 linestyle=line.get_linestyle(), 
                 linewidth=line.get_linewidth(),
                 label=line.get_label())
for collection in ax.collections:
    ax_left.scatter(collection.get_offsets()[:, 0], 
                   collection.get_offsets()[:, 1],
                   marker=collection.get_paths()[0],
                   color=collection.get_facecolors()[0],
                   zorder=20)
ax_left.legend(loc='upper left', fontsize=8)
ax_left.set_xlabel("Deflection [mm] / Deflection angle [deg]")
ax_left.set_ylabel("Tip force [N] / Tip moment [Nm]")
ax_left.grid(which="major", color="grey")
ax_left.set_xlim(0, 120)
ax_left.set_ylim(0, 120)

# Copy ax2 content to second subplot
ax_right = fig_composite.add_subplot(gs[0, 1])
for collection in ax2.collections:
    ax_right.scatter(collection.get_offsets()[:, 0], 
                    collection.get_offsets()[:, 1],
                    marker='+',
                    color='black',
                    zorder=20)
ax_right.grid()
ax_right.set_xlabel('Element length (m)')
ax_right.set_ylabel('Deflection Error / Element length (-)')
ax_left.text(0.5, -0.25, '(a)', transform=ax_left.transAxes, fontsize=12)
ax_right.text(0.5, -0.25, '(b)', transform=ax_right.transAxes, fontsize=12)
fig_composite.tight_layout()
fig_composite.savefig(os.path.join(script_dir, 'composite_figure.pdf'))