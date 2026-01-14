import os
script_dir = os.path.dirname(os.path.abspath(__file__))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = ["b","orange","g","r","purple","brown","pink","gray"]

def import_data(file_name):
    """
    Import data from a CSV file.
    """
    path = os.path.join(script_dir, file_name)
    data = pd.read_csv(path)
    dictionary = {col: data[col].tolist() for col in data.columns}
    return dictionary
    
PSM2D = import_data("PSM2D.csv")
PSM3D = import_data("PSM3D.csv")
Literature = import_data("Literature.csv")
FEM_timoshenko2 = import_data("FEM_timoshenko2node.csv")
FEM_timoshenko4 = import_data("FEM_timoshenko4node.csv")
FEM_timoshenko6 = import_data("FEM_timoshenko6node.csv")
FEM_timoshenko2_lowdeflection = import_data("FEM_timoshenko2node_lowdeflection.csv")
FEM_timoshenko4_lowdeflection = import_data("FEM_timoshenko4node_lowdeflection.csv")
FEM_timoshenko6_lowdeflection = import_data("FEM_timoshenko4node_lowdeflection.csv")

PSM2D["L_ratio"] 



#L_ratio,Load_param,Force_Angle,Vertical_disp,Horizontal_disp,Beam_angle
L_ratio = np.array(PSM2D["L_ratio"])
Load_param = np.array(PSM2D["Load_param"])
Vertical_disp = np.array(PSM2D["Vertical_disp"])
Horizontal_disp = np.array(PSM2D["Horizontal_disp"])
Beam_angle = np.array(PSM2D["Beam_angle"])

L_ratios = np.unique_values(L_ratio)
fig1, ax1 = plt.subplots(figsize=(4,4))
fig2, ax2 = plt.subplots(figsize=(4,4))
fig3, ax3 = plt.subplots(figsize=(4,4))
fig5, ax5 = plt.subplots(figsize=(4,4))
fig6, ax6 = plt.subplots(figsize=(4,4))
fig7, ax7 = plt.subplots(figsize=(4,4))

for i in [0,2]:
    Bool = np.zeros_like(L_ratio,dtype=bool)
    Bool = np.where(L_ratio == L_ratios[i],True,Bool)
    DOF = int((L_ratios[i]*4+4)*3-12)
    ax1.plot(Vertical_disp[Bool],Load_param[Bool], label = f"PSM, DOF = {DOF}",color=colors[i],)
    ax2.plot(Horizontal_disp[Bool],Load_param[Bool], label = f"PSM, DOF = {DOF}",color=colors[i])
    ax3.plot(Beam_angle[Bool],Load_param[Bool], label = f"PSM, DOF = {DOF}",color=colors[i])
    ax5.plot(Vertical_disp[Bool],Load_param[Bool], label = f"PSM, DOF = {DOF}",color=colors[i],)
    ax6.plot(Horizontal_disp[Bool],Load_param[Bool], label = f"PSM, DOF = {DOF}",color=colors[i])
    ax7.plot(Beam_angle[Bool],Load_param[Bool], label = f"PSM, DOF = {DOF}",color=colors[i])

ax1.plot(Literature["Vertical_disp"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")
ax2.plot(Literature["Horizontal_disp"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")
ax3.plot(Literature["Beam_angle"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")



ax1.plot(FEM_timoshenko2["Vertical_disp"],FEM_timoshenko2["Load_param"], linestyle="--",color="gray",label="Timoshenko, DOF=6")
ax2.plot(FEM_timoshenko2["Horizontal_disp"],FEM_timoshenko2["Load_param"], linestyle="--",color="gray",label="Timoshenko, DOF=6")
ax3.plot(FEM_timoshenko2["Beam_angle"],FEM_timoshenko2["Load_param"], linestyle="--",color="gray",label="Timoshenko, DOF=6")
ax1.plot(FEM_timoshenko4["Vertical_disp"],FEM_timoshenko4["Load_param"], linestyle="--",color="purple",label="Timoshenko, DOF=18")
ax2.plot(FEM_timoshenko4["Horizontal_disp"],FEM_timoshenko4["Load_param"], linestyle="--",color="purple",label="Timoshenko, DOF=18")
ax3.plot(FEM_timoshenko4["Beam_angle"],FEM_timoshenko4["Load_param"], linestyle="--",color="purple",label="Timoshenko, DOF=18")
# ax1.plot(FEM_timoshenko6["Vertical_disp"],FEM_timoshenko6["Load_param"], linestyle="--",color="brown",label="Timoshenko, DOF=30")
# ax2.plot(FEM_timoshenko6["Horizontal_disp"],FEM_timoshenko6["Load_param"], linestyle="--",color="brown",label="Timoshenko, DOF=30")
# ax3.plot(FEM_timoshenko6["Beam_angle"],FEM_timoshenko6["Load_param"], linestyle="--",color="brown",label="Timoshenko, DOF=30")



ax1.legend(fontsize="small")
ax1.set_xlabel(f"Vertical Non-Dimensional Deflection "+r"$w/L$ (-)")
ax1.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax1.set_xlim(0,1)
ax1.set_ylim(0,10)
ax1.grid()
fig1.tight_layout()
ax2.legend(fontsize="small")
ax2.set_xlabel(f"Horizonal Non-Dimensional Deflection "+r"$u/L$ (-)")
ax2.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax2.set_xlim(0,1)
ax2.set_ylim(0,10)
ax2.grid()
fig2.tight_layout()
ax3.legend(fontsize="small")
ax3.set_xlabel(f"Tip Angle "+r"$\theta_0$ (rad)")
ax3.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax3.set_xlim(0,1.5)
ax3.set_ylim(0,10)
ax3.grid()
fig3.tight_layout()



# 3D Cantilever PSM
# Unpack data
Force_angle = PSM3D["Force_Angle"]
Vertical_disp = PSM3D["Vertical_disp"]
Horizontal_disp = PSM3D["Horizontal_disp"]
Beam_angle = PSM3D["Beam_angle"]
# Plotting 3D PSM Cantilever force angle sweep vs target values
fig4, ax4 = plt.subplots(figsize=(8,4))
ax4.plot(Force_angle, Vertical_disp, color=colors[0],label=r"$w/L$")
ax4.plot([0,90],[0.46326,0.46326], linestyle="--",color=colors[0])
ax4.plot(Force_angle, Horizontal_disp,color=colors[1], label=r"$u/L$")
ax4.plot([0,90],[0.13981,0.13981], linestyle="--",color=colors[1])
ax4.plot(Force_angle, Beam_angle,color=colors[2], label=r"$\theta_0$")
ax4.plot([0,90],[0.72876,0.72876], linestyle="--",color=colors[2])
ax4.grid()
ax4.set_xlabel(r"Force angle $\alpha$ (°)")
ax4.set_ylabel(r"$\mathrm{Non\text{-}dimensional\ deflection}\ w/L,\ u/L\ [-]$" "\n" r"$\mathrm{angle}\ \theta_0\ (rad)$")
ax4.set_xlim(0,90)
ax4.set_ylim(0,1)
ax4.legend(fontsize="small")
fig4.tight_layout()


fig1.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Vertical_disp.pdf'))
fig2.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Horizontal_disp.pdf'))
fig3.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Beam_angle.pdf'))
fig4.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_PSM_Force_angle.pdf'))




ax5.plot(Literature["Vertical_disp"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")
ax6.plot(Literature["Horizontal_disp"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")
ax7.plot(Literature["Beam_angle"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")

ax5.plot(FEM_timoshenko2["Vertical_disp"],FEM_timoshenko2["Load_param"], linestyle="--",color="gray",label="Timoshenko, DOF=6")
ax6.plot(FEM_timoshenko2["Horizontal_disp"],FEM_timoshenko2["Load_param"], linestyle="--",color="gray",label="Timoshenko, DOF=6")
ax7.plot(FEM_timoshenko2["Beam_angle"],FEM_timoshenko2["Load_param"], linestyle="--",color="gray",label="Timoshenko, DOF=6")

ax5.plot(FEM_timoshenko4["Vertical_disp"],FEM_timoshenko4["Load_param"], linestyle="--",color="purple",label="Timoshenko, DOF=18")
ax6.plot(FEM_timoshenko4["Horizontal_disp"],FEM_timoshenko4["Load_param"], linestyle="--",color="purple",label="Timoshenko, DOF=18")
ax7.plot(FEM_timoshenko4["Beam_angle"],FEM_timoshenko4["Load_param"], linestyle="--",color="purple",label="Timoshenko, DOF=18")

# ax5.plot(FEM_timoshenko6_lowdeflection["Vertical_disp"],FEM_timoshenko6_lowdeflection["Load_param"], linestyle="--",color="brown",label="Timoshenko, DOF=30")
# ax6.plot(FEM_timoshenko6_lowdeflection["Horizontal_disp"],FEM_timoshenko6_lowdeflection["Load_param"], linestyle="--",color="brown",label="Timoshenko, DOF=30")
# ax7.plot(FEM_timoshenko6_lowdeflection["Beam_angle"],FEM_timoshenko6_lowdeflection["Load_param"], linestyle="--",color="brown",label="Timoshenko, DOF=30")



ax5.legend(fontsize="small")
ax5.set_xlabel(f"Vertical Non-Dimensional Deflection "+r"$w/L$ (-)")
ax5.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax5.set_xlim(0,1)
ax5.set_ylim(0,3)
ax5.grid()
fig5.tight_layout()
ax6.legend(fontsize="small")
ax6.set_xlabel(f"Horizonal Non-Dimensional Deflection "+r"$u/L$ (-)")
ax6.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax6.set_xlim(0,1)
ax6.set_ylim(0,3)
ax6.grid()
fig6.tight_layout()
ax7.legend(fontsize="small")
ax7.set_xlabel(f"Tip Angle "+r"$\theta_0$ (rad)")
ax7.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax7.set_xlim(0,1)
ax7.set_ylim(0,3)
ax7.grid()
fig7.tight_layout()

fig5.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Vertical_disp_small.pdf'))
fig6.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Horizontal_disp_small.pdf'))
fig7.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Beam_angle_small.pdf'))
