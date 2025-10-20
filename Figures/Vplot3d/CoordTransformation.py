import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from itertools import product, combinations
from vplot3d.vplot3d import init_view, Line, Vector, Point, Arc, ArcMeasure, Polygon, save_svg_tex, save_svg

# Inkscape and tex path
os.environ["PATH"] += r";C:\Program Files\Inkscape\bin"
os.environ["PATH"] += r";C:\texlive\2025\bin\windows"

#save path
save_file_path = Path(__file__).resolve().parent 


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d', proj_type='ortho')
# ax.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio for x, y, z
# ax.set_axis_off()

# # Initialize vector diagram
# # Initialize vector diagram
# init_view(width=600, height=600,
#           xmin=-1, xmax=2, ymin=-1.5, ymax=1.5, zmin=-1.5, zmax=2,
#           zoom=1.5, elev=30, azim=-60)


# # Right-handed coordinate system: x (red), y (green), z (blue)
# x = Vector(np.array([0, 0, 0]), np.array([2, 0, 0]), linewidth=2, color="r")
# y = Vector(np.array([0, 0, 0]), np.array([0, 0, 2]), linewidth=2, color="g")
# z = Vector(np.array([0, 0, 0]), np.array([0, -2, 0]), linewidth=2, color="b")

# theta = np.deg2rad(30)
# # Rotation matrix around x-axis
# R_x = np.array([
#     [1, 0, 0],
#     [0, np.cos(theta), -np.sin(theta)],
#     [0, np.sin(theta),  np.cos(theta)]
# ])

# # Rotated axes
# y_rot = Vector(np.array([0, 0, 0]), R_x @ np.array([0, 0, 2]), linewidth=2, color="g", linestyle="--")
# z_rot = Vector(np.array([0, 0, 0]), R_x @ np.array([0, -2, 0]), linewidth=2, color="b", linestyle="--")

# am = ArcMeasure(v1=0.5*np.array([0,0,1]),v2=R_x @ (np.array([0, 0, 1])*0.5),vn=np.array([1,0,0]),color="g",scale=1)
# am2 = ArcMeasure(v1=0.5*np.array([0,-1,0]),v2=R_x @ (np.array([0, -1, 0])*0.5),vn=np.array([1,0,0]),color="b",scale=1)

# ax.annotate3D(r'$x$', xyz=np.array([2.1, 0.0, 0.0]), xytext=(0,0))
# ax.annotate3D(r'$y$', xyz=np.array([0, 0.0, 2.1]), xytext=(0,0))
# ax.annotate3D(r'$z$', xyz=np.array([0, -2.3, 0.0]), xytext=(0,0))

# ax.annotate3D(r'$\varphi$', xyz=np.array([0, -.5, 1.3]), xytext=(0,0))
# ax.annotate3D(r'$\varphi$', xyz=np.array([0, -1.4, -0.2]), xytext=(0,0))

# os.chdir(save_file_path)
# save_svg_tex('CoordTransformation',scour=False,fontsize=32)
# # save_svg('CoordTransformation')
# os.chdir(Path.cwd().parent)  
# plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d', proj_type='ortho')
# ax.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio for x, y, z
# ax.set_axis_off()

# # Initialize vector diagram
# init_view(width=600, height=600,
#           xmin=-1, xmax=2, ymin=-1.5, ymax=1.5, zmin=-1.5, zmax=2,
#           zoom=1.5, elev=30, azim=-60)

# # Right-handed coordinate system: x (red), y (green), z (blue)
# x = Vector(np.array([0, 0, 0]), np.array([2, 0, 0]), linewidth=2, color="r")
# y = Vector(np.array([0, 0, 0]), np.array([0, 0, 2]), linewidth=2, color="g")
# z = Vector(np.array([0, 0, 0]), np.array([0, -2, 0]), linewidth=2, color="b")

# theta = np.deg2rad(30)
# # Rotation matrix around y-axis
# R_y = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta),  np.cos(theta), 0],
#     [0, 0, 1],

# ])

# # Rotated axes
# x_rot = Vector(np.array([0, 0, 0]), R_y @ np.array([2, 0, 0]), linewidth=2, color="r", linestyle="--")
# z_rot = Vector(np.array([0, 0, 0]), R_y @ np.array([0, -2, 0]), linewidth=2, color="b", linestyle="--")

# am = ArcMeasure(v1=0.5*np.array([1,0,0]),v2=R_y @ (np.array([1, 0, 0])*0.5),vn=np.array([0,0,1]),color="r",scale=1)
# am2 = ArcMeasure(v1=0.5*np.array([0,-1,0]),v2=R_y @ (np.array([0, -1, 0])*0.5),vn=np.array([0,0,1]),color="b",scale=1)

# ax.annotate3D(r'$x$', xyz=np.array([2.1, 0.0, 0.0]), xytext=(0,0))
# ax.annotate3D(r'$y$', xyz=np.array([0, 0.0, 2.1]), xytext=(0,0))
# ax.annotate3D(r'$z$', xyz=np.array([0, -2.3, 0.0]), xytext=(0,0))

# ax.annotate3D(r'$\theta$', xyz=np.array([0, -1, -0.4]), xytext=(0,0))
# ax.annotate3D(r'$\theta$', xyz=np.array([1.3, 0, 0.08]), xytext=(0,0))

# os.chdir(save_file_path)
# save_svg_tex('CoordTransformation2',scour=False,fontsize=32)
# # save_svg('CoordTransformation')
# os.chdir(Path.cwd().parent)  
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')
ax.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio for x, y, z
ax.set_axis_off()

# Initialize vector diagram
init_view(width=600, height=600,
          xmin=-1, xmax=2, ymin=-1.5, ymax=1.5, zmin=-1.5, zmax=2,
          zoom=1.5, elev=30, azim=-60)

# Right-handed coordinate system: x (red), y (green), z (blue)
x = Vector(np.array([0, 0, 0]), np.array([2, 0, 0]), linewidth=2, color="r")
y = Vector(np.array([0, 0, 0]), np.array([0, 0, 2]), linewidth=2, color="g")
z = Vector(np.array([0, 0, 0]), np.array([0, -2, 0]), linewidth=2, color="b")

theta = np.deg2rad(-30)
# Rotation matrix around y-axis
R_z = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)],
])

# Rotated axes
x_rot = Vector(np.array([0, 0, 0]), R_z @ np.array([2, 0, 0]), linewidth=2, color="r", linestyle="--")
y_rot = Vector(np.array([0, 0, 0]), R_z @ np.array([0, 0, 2]), linewidth=2, color="g", linestyle="--")

am = ArcMeasure(v1=0.5*np.array([1,0,0]),v2=R_z @ (np.array([1, 0, 0])*0.5),vn=np.array([0,-1,0]),color="r",scale=1)
am2 = ArcMeasure(v1=0.5*np.array([0,0,1]),v2=R_z @ (np.array([0, 0, 1])*0.5),vn=np.array([0,-1,0]),color="g",scale=1)

ax.annotate3D(r'$x$', xyz=np.array([2.1, 0.0, 0.0]), xytext=(0,0))
ax.annotate3D(r'$y$', xyz=np.array([0, 0.0, 2.1]), xytext=(0,0))
ax.annotate3D(r'$z$', xyz=np.array([0, -2.3, 0.0]), xytext=(0,0))

ax.annotate3D(r'$\psi$', xyz=np.array([0, -0.5, 1.5]), xytext=(0,0))
ax.annotate3D(r'$\psi$', xyz=np.array([1.1, 0, 0.15]), xytext=(0,0))

os.chdir(save_file_path)
save_svg_tex('CoordTransformation3',scour=False,fontsize=32)
# save_svg('CoordTransformation')
os.chdir(Path.cwd().parent)  
plt.show()
