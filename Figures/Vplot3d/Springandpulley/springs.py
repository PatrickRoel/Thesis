import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from itertools import product, combinations
from vplot3d.vplot3d import init_view, Line, Vector, Point, Arc, ArcMeasure, Polygon, save_svg_tex, save_svg

# Inkscape and tex path
os.environ["PATH"] += r";C:\Program Files\Inkscape\bin"
os.environ["PATH"] += r";C:\texlive\2025\bin\windows"
os.environ["PATH"] += r";C:\Users\roele\Documents\Aerospace engineering\git\Repositories\Thesis\.venv\Scripts"

#save path
save_file_path = Path(__file__).resolve().parent 


fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')
ax.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio for x, y, z
ax.set_axis_off()

# Initialize vector diagram
init_view(width=600, height=300,
          xmin=0, xmax=3, ymin=0, ymax=1.5, zmin=-0, zmax=1,
          zoom=2, elev=90, azim=-90)

n1 = np.array([0,0,0])
n2 = np.array([3,1,0])
l1 = np.linalg.norm(n2-n1)

F1 = Vector(n1,n2/l1/2,linewidth=2,color="red",zorder=20)
F2 = Vector(n2, (n1-n2)/l1/2,linewidth=2,color="red",zorder=20)

p = Point(n1,color="black",facecolor="black",bgcolor="black",zorder=50)
p = Point(n2,color="black",facecolor="black",bgcolor="black",zorder=50)

spring = Line(n1,n2,linewidth=2)


ax.annotate3D(r'$\text{n}_1$', xyz=n1+[-0.4,-0.2,0], xytext=(0,0))
ax.annotate3D(r'$\text{n}_2$', xyz=n2+[0.1,0,0], xytext=(0,0))


ax.annotate3D(r'$\textbf{f}_\text{int}$', xyz=n1+[0.0,0.25,0], xytext=(0,0), color= "red")
ax.annotate3D(r'$\textbf{f}_\text{int}$', xyz=n2+[-0.6,0.05,0], xytext=(0,0),  color= "red")

n1_lower = n1-[0,0.2,0]
n2_lower = n2-[0,0.2,0]
n_half = n1_lower+0.5*(n2_lower-n1_lower)

# p = Point(n1_lower,color="black",facecolor="black",bgcolor="black",zorder=50)
# p = Point(n_half,color="black",facecolor="black",bgcolor="black",zorder=50)

l1 = Vector(n_half,n1_lower-n_half,linewidth=.75)
l2 = Vector(n_half,n2_lower-n_half,linewidth=.75)

ax.annotate3D(r'$l_{12}$', xyz=n_half-[-0.0,0.3,0], xytext=(0,0))

# # Right-handed coordinate system: x (red), y (green), z (blue)
# x = Vector(np.array([0, 0, 0]), np.array([2, 0, 0]), linewidth=2, color="r")
# y = Vector(np.array([0, 0, 0]), np.array([0, 2, 0]), linewidth=2, color="g")
# z = Vector(np.array([0, 0, 0]), np.array([0, 0, 2]), linewidth=2, color="b")



os.chdir(save_file_path)
save_svg_tex('spring',scour=True,fontsize=28)
os.chdir(Path.cwd().parent)  
