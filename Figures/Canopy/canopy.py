import os

import matplotlib.pyplot as plt
def plot_canopy(N_x,N_y,L_x,L_y,scaling,border_beam):


    color = "blue"
    linewidth = 1
    coords = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio for x, y, z
    ax.set_axis_off()
    ax.view_init(elev=30, azim=-135)  # Set elevation and azimuth angles for 3D view

    id_list = []
    id_list_le = []
    idx=0
    for y in range(N_y):
        id_list_x = []
        for x in range(N_x):
            if x == 0:
                id_list_le.append(idx)
            coords.append([x/L_x*(1-scaling*y),y/L_y,0])
            id_list_x.append(idx)
            idx += 1
        id_list.append(id_list_x)


    for coord in coords:
        ax.scatter(coord[0],coord[1],coord[2],color = "red",zorder=20,s=15)

    all_sections_1 = id_list[0:-1]
    all_sections_2 = id_list[1:]

    connectivity = []
    for section1,section2 in zip(all_sections_1,all_sections_2):
            # Connect corresponding nodes between adjacent struts with squares and diagonals
            # Skip the first connection (section1[0] to section2[0])
            for i in range(1, len(section1)):
                if i < len(section2):
                    connectivity.append([section1[i], section2[i]])

                    
                    # Diagonal connections: section1[i] to section2[i-1] (if i > 0)
                    if i > 0:
                        connectivity.append([section1[i], section2[i-1]])

                    
                    # Diagonal connections: section1[i-1] to section2[i] (if i > 0)
                    if i > 0:
                        connectivity.append([section1[i-1], section2[i]])


    for connection in connectivity:
        coord1 = coords[connection[0]]
        coord2 = coords[connection[1]]
        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color='blue',linewidth = linewidth)


    for i in range(1,N_y-1):
        ids1 = id_list[i][0:-1]
        ids2 = id_list[i][1:]
        for id1,id2 in zip(ids1,ids2):
            coord1 = coords[id1]
            coord2 = coords[id2]
            ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color='blue',linewidth = linewidth)


    if border_beam:
        color = "green"
        linewidth = 1.5
    else:
        color = "blue"

    for i in [0,-1]:
        ids1 = id_list[i][0:-1]
        ids2 = id_list[i][1:]
        for id1,id2 in zip(ids1,ids2):
            coord1 = coords[id1]
            coord2 = coords[id2]
            ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color=color, linewidth=linewidth)

    ids1 = id_list_le[0:-1]
    ids2 = id_list_le[1:]
    for id1,id2 in zip(ids1,ids2):
        coord1 = coords[id1]
        coord2 = coords[id2]
        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color=color,linewidth=linewidth)
    
    return ax,fig

N_x = 2
N_y = 2
L_x = 1
L_y = 1
scaling =0
border_beam = False
ax,fig = plot_canopy(N_x,N_y,L_x,L_y,scaling,border_beam)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ax.plot([], [], color='blue', linewidth=1, label='Non-compressive spring')
ax.plot([], [], color='green', linewidth=1.5, label='Inflatable beam')
ax.scatter([], [], color='red', s=15, label='Node')
ax.legend()
fig.savefig('canopy.svg', format='svg', bbox_inches='tight')

plt.show()

N_x = 2
N_y = 2
L_x = 1
L_y = 1
scaling =-0.5
border_beam = False
ax,fig = plot_canopy(N_x,N_y,L_x,L_y,scaling,border_beam)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ax.plot([], [], color='blue', linewidth=1, label='Non-compressive spring')
ax.plot([], [], color='green', linewidth=1.5, label='Inflatable beam')
ax.scatter([], [], color='red', s=15, label='Node')
ax.legend()
fig.savefig('canopy_crooked.svg', format='svg', bbox_inches='tight')

plt.show()

N_x = 5
N_y = 4
L_x = 1
L_y = 1
scaling =0

border_beam = True
ax,fig = plot_canopy(N_x,N_y,L_x,L_y,scaling,border_beam)
ax.plot([], [], color='blue', linewidth=1, label='Non-compressive spring')
ax.plot([], [], color='green', linewidth=1.5, label='Inflatable beam')
ax.scatter([], [], color='red', s=15, label='Node')
ax.legend()
fig.savefig('canopy_section.svg', format='svg', bbox_inches='tight')

plt.show()

N_x = 5
N_y = 4
L_x = 3
L_y = 1
scaling =-0.25

border_beam = True
ax,fig = plot_canopy(N_x,N_y,L_x,L_y,scaling,border_beam)
ax.plot([], [], color='blue', linewidth=1, label='Non-compressive spring')
ax.plot([], [], color='green', linewidth=1.5, label='Inflatable beam')
ax.scatter([], [], color='red', s=15, label='Node')
ax.legend()
fig.savefig('canopy_section_crooked.svg', format='svg', bbox_inches='tight')

plt.show()