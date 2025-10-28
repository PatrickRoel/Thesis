import numpy as np
import matplotlib.pyplot as plt
import os
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
    T = ((C13*r+C14)*p+(C15*r+C16))*np.arctan(((C17*r**4)*np.log(p)+(C18*r**3+C19))*np.deg2rad(phi))
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

p_lst = [0.4] #bar
d_lst = [0.18] #m
linestyles = ['-', '--']
fig, ax = plt.subplots(figsize=(5,5))


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
import csv


def instiantiate(d,p,elements):
    length  = 1  
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
            max_iterations=500,
            tolerance=0.001,
            step_limit=0.25,
            relax_init=0.5,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25,
            print_info=False
            )
    deflection = -inflatable_beam.coords_current[-2]*1000
    inflatable_beam.reinitialise()

    return deflection, converged

def solve_tip_moment(inflatable_beam,tip_moment):
    fe = np.zeros(inflatable_beam.N)
    fe[3::6][-1] = -tip_moment
    inflatable_beam.solve(        fe=fe,
            max_iterations=500,
            tolerance=0.001,
            step_limit=0.5,
            relax_init=1,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25
            )
    rotation = -np.rad2deg(inflatable_beam.coords_rotations_current[-3])
    inflatable_beam.reinitialise()
    return rotation

pressures = p_lst
diameters = d_lst
inflatable_beams = []
elements = [1,2,3,5,10,25]
for pressure,diameter in zip(pressures,diameters):
    for element in elements:
        inflatable_beam = instiantiate(diameter,pressure,element)
        inflatable_beams.append(inflatable_beam)

tip_loads = np.arange(5,95,5)
tip_moments = np.arange(5,120,10)

#first run
# alpha = np.arange(-0.35,0.35,0.05)
# beta = np.arange(0.75,1.25,0.05)
# gamma = np.arange(0.75,1,0.025)
#used element options 3,5,10
#best option = alpha  = -0.2,  beta = 1.2, gamma = 0.85

#second run
# alpha = np.arange(-0.25, -0.14, 0.01)
# beta = np.arange(1.15, 1.21, 0.01)
# gamma = np.arange(0.775, 1, 0.025)
#used element options [1,2,3,5,10]

#third run
alpha = np.arange(-0.35,0.35,0.1)
beta = np.arange(0.75,1.4,0.1)
gamma = np.arange(0.75,1,0.05)

abg = []
errors = []
converged_list = []
total = len(alpha) * len(beta) * len(gamma)
count = 0
print(len(inflatable_beams))
for a in alpha:
    for b in beta:
        for g in gamma:
            errors_temp = []
            converged_list_temp = []
            for inflatable_beam in inflatable_beams:
                inflatable_beam.update_beam_parameters(a, b, g)
                deflections = []
                rotations = []
                for tip_load in tip_loads:
                    deflection, converged = solve_tip_load(inflatable_beam, tip_load)
                    deflections.append(deflection)
                converged_list_temp.append(converged)
                deflections = np.array(deflections)
                force_target = F_inflatablebeam(pressure, diameter/2, deflections/1000)
                error = np.linalg.norm(tip_loads - force_target)
                errors_temp.append(error)
            error = np.mean(errors_temp)
            abg.append(np.array([a, b, g]))
            converged_list.append(min(converged_list_temp))#TODO check if this works
            errors.append(error)
            count += 1
            pct = count / total * 100
            print(f" abg: {float(a):.6g}, {float(b):.6g}, {float(g):.6g}  Error: {float(error):.6g}  Progress: {count}/{total} ({pct:.1f}%)")

pressures = np.ones_like(errors)*p_lst[0]
diameters = np.ones_like(errors)*d_lst[0]
elements = np.ones_like(errors)*5

out_path = os.path.join(script_dir, "inflatable_beam_results.csv")
# number of finite elements used in instiantiate()


with open(out_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["pressure", "diameter", "alpha", "beta", "gamma", "error", "converged"])

    idx = 0
    for a in alpha:
        for b in beta:
            for g in gamma:
                print(idx)
                pressure_val = pressures[idx] 
                diameter_val = diameters[idx] 
                err_val = errors[idx] 
                conv_val = converged_list[idx]
                writer.writerow([pressure_val, diameter_val, a, b, g, err_val, bool(conv_val)])
                idx += 1

print(f"Wrote results to: {out_path}")







# filter results by convergence using the collected converged_list
converged_mask = np.array(converged_list, dtype=bool)
if converged_mask.size == 0 or not converged_mask.any():
    print("No converged solutions found.")
else:
    errors = np.array(errors)[converged_mask]
    abg = np.array(abg)[converged_mask]
    min_error_index = np.argmin(errors)
    best_abg = abg[min_error_index]
    print("Best parameters found:")
    print(f" a = {best_abg[0]}")
    print(f" b = {best_abg[1]}")
    print(f" g = {best_abg[2]}")
    print(f" Error = {errors[min_error_index]}")

