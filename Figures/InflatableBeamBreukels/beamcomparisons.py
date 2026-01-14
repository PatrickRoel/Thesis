

#make a beam from 1 m beam elements
#relate deflection to length

from kite_fem.FEMStructure import FEM_structure
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def instiantiate(d,p,L,elements):
    length  = L  
    initital_conditions = []
    l0 = length/elements
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
    converged,residual = inflatable_beam.solve(        fe=fe,
            max_iterations=1000,
            tolerance=0.00001,
            step_limit=0.05,
            relax_init=0.25,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25,
            convergence_criteria = "residual"
            
            )
    deflection = -inflatable_beam.coords_current[-2]*1000
    return deflection,converged

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


def compare_collapse_deflection(d,p,length,elements,max_range):
    L_el = length/elements
    if L_el <= max_range:
        beam = instiantiate(d,p,length,length)
        beam2 = instiantiate(d,p,length,elements)

        # v = v_collapse(p,d/2) 
        # load = F_inflatablebeam(p, d/2, v)
        defl2_lst = []
        load_lst = []
        loads = np.linspace(0,100,25)
        for load in loads:
            defl2,converged = solve_tip_load(beam,load)
            collapsed = False
            for element in beam.beam_elements:
                if element.collapsed:
                    collapsed = True
            if not converged or collapsed:
                break
            if not collapsed:
                defl2_lst.append(defl2)
                load_lst.append(load)
        load = load_lst[-1]
        defl,converged = solve_tip_load(beam,load)
        defl2,converged2 = solve_tip_load(beam2,load)

        error = abs(defl-defl2)
        errorpercentage = error / defl * 100
        converged = min(converged,converged2)
    else:
        error = 0
        errorpercentage = 0
        converged = False
    return error,errorpercentage,converged




d = 0.25
p = 0.7
Lengths = [1,2,4,5,6]
Elements = [3,2,1]
max_range = 3
Length_array = []
Error_array = []
for length in Lengths:
    for element in Elements:
        L_el = length/element
        error,errorpercentage,converged = compare_collapse_deflection(d,p,length,element,max_range)
        norm_error = error/L_el/1000
        if converged:
            plt.scatter(L_el,errorpercentage,marker="+",color="black")
            Length_array.append(L_el)
            Error_array.append(errorpercentage)

plt.legend()
plt.grid()
plt.xlabel("Element length (m)")
plt.ylabel("Normalised error (-)")

plt.show()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "beam_comparison_results.csv")

# Save to CSV
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Element Length (m)', 'Deflection error (%)'])
    for length, error in zip(Length_array, Error_array):
        writer.writerow([length, error])

print(f"Results saved to: {csv_path}")