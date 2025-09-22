import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure
import pandas as pd
import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))


initital_conditions = []
length = 10 #m
elements = 5

for i in range(elements+1):
    initital_conditions.append([[i*length/elements, 0.0, 0.0], [0, 0, 0], 1, True if i==0 else False])

E = 210e5 # Pa
I = 1.6e-5  # m^4
A = 0.1  # m^2
print("E*I", E*I)
L = length/elements #m

beam_matrix = []

for i in range(elements):
    beam_matrix.append([i, i+1, E, A, I])

steel_beam = FEM_structure(initital_conditions, beam_matrix=beam_matrix)

load_params_lst = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
Vertical_disp_lst = []
Horizontal_disp_lst = []
Beam_angle_lst = []

for load_param in load_params_lst:
    tip_load = load_param*E*I/(length**2)
    fe = np.zeros(steel_beam.N)
    fe[1::6][-1] = -tip_load
    steel_beam.solve(        fe=fe,
            max_iterations=2000,
            tolerance=0.01,
            step_limit=2,
            relax_init=1,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25
            )

    Vertical_disp_lst.append(-steel_beam.coords_rotations_current[1::6][-1]/length)
    Horizontal_disp_lst.append(1-steel_beam.coords_rotations_current[0::6][-1]/length)
    Beam_angle_lst.append(-steel_beam.coords_rotations_current[-1])
    steel_beam.reinitialise()

df = pd.DataFrame({
"Load_param": load_params_lst,
"Vertical_disp": Vertical_disp_lst,
"Horizontal_disp": Horizontal_disp_lst,
"Beam_angle": Beam_angle_lst
}) 

csv_path = os.path.join(os.path.dirname(script_dir), 'Figures', 'Cantilever','Results', "FEM_timoshenko6node_lowdeflection.csv")

# Append to CSV (create if doesn't exist)
if not os.path.isfile(csv_path):
    df.to_csv(csv_path,  header=True, index=False)
else:
    df.to_csv(csv_path, mode='a', header=False, index=False)