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

p_lst = [0.3,0.4,0.5] #bar
d_lst = [0.18,0.18,0.18] #m
linestyles = ['-', '--', '-.']
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


ax.legend(loc='upper left', fontsize=9)
ax.set_xlabel("Deflection [mm] / Deflection angle [deg]")
ax.set_ylabel("Tip force [N] / Tip moment [Nm]")
ax.minorticks_on() 
ax.grid(which="major",color="grey")
ax.grid(which="minor",color="lightgrey")
ax.set_xlim(0,140)
ax.set_ylim(0,160)
fig.tight_layout()

fig.savefig(os.path.join(script_dir, 'Inflatablebeamdeflections.pdf'))
