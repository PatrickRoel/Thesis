

#make a beam from 1 m beam elements
#relate deflection to length

from kite_fem.FEMStructure import FEM_structure
import numpy as np
import matplotlib.pyplot as plt

def instiantiate(d,p,L,elements):
    length  = L  
    initital_conditions = []
    for i in range(elements+1):
        initital_conditions.append([[i*length/elements, 0.0, 0.0], [0, 0, 0], 1, True if i==0 else False])
    beam_matrix = []
    for i in range(elements):
        beam_matrix.append([i, i+1, d, p])
    inflatable_beam = FEM_structure(initital_conditions, beam_matrix=beam_matrix)
    return inflatable_beam

def solve_tip_load(inflatable_beam,tip_load):
    fe = np.zeros(inflatable_beam.N)
    fe[1::6][-1] = -tip_load
    converged = inflatable_beam.solve(        fe=fe,
            max_iterations=1000,
            tolerance=0.00001,
            step_limit=0.05,
            relax_init=0.25,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25
            )
    deflection = -inflatable_beam.coords_current[-2]*1000
    inflatable_beam.reinitialise()
    return deflection,converged

d = 0.35
p = 0.2

beam2 = instiantiate(d,p,2,2)


loads = np.linspace(0,100,20)
defl2_lst = []
load_lst = []
for load in loads:
    defl2,converged = solve_tip_load(beam2,load)
    collapsed = False
    for element in beam2.beam_elements:
        if element.collapsed:
            collapsed = True
    if not converged:
        break
    if not collapsed:
        defl2_lst.append(defl2)
        load_lst.append(load)


plt.plot(defl2_lst,load_lst,label="L=2m target")



beam = instiantiate(d,p,2,3)

for load in load_lst:
    defl,converged = solve_tip_load(beam,load)
    collapsed = False
    for element in beam.beam_elements:
        if element.collapsed:
            collapsed = True
    if not converged:
        break
    if not collapsed:
        plt.scatter(defl,load,marker="+",color="black")






plt.grid()
plt.xlabel("Deflection (mm)")
plt.ylabel("Tip Load (N)")
plt.title("Inflatable Beam Tip Load vs Deflection (2m)")
plt.show()
