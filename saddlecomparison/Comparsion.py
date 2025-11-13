from FEM_saddle import fem_saddle
from PSM_saddle import psm_saddle
import numpy as np
import csv
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


grid_length = 10
grid_height = 11
grid_sizes = range(3,15,2)

nodes_lst = []
fem_runtimes = []
psm_runtimes = []
max_diffences = []

attempts_for_time_avg = 5


for i in range(attempts_for_time_avg):
    psm_runtimes_nested = []
    fem_runtimes_nested = []
    for grid_size in grid_sizes:
        nodes = 2*grid_size**2-2*grid_size+1
        femcoords, femruntime = fem_saddle(grid_size,grid_length,grid_height)
        psmcoords, psmruntime = psm_saddle(grid_size,grid_length,grid_height)
        psm_runtimes_nested.append(psmruntime)
        fem_runtimes_nested.append(femruntime)
        if i == attempts_for_time_avg-1:
            max_difference = np.linalg.norm(np.max(femcoords-psmcoords))*1000 #maximum nodal misallignment in mm
            nodes_lst.append(nodes)
            max_diffences.append(max_difference)
    psm_runtimes.append(psm_runtimes_nested)
    fem_runtimes.append(fem_runtimes_nested)

# Calculate averages across attempts
psm_runtimes_avg = [sum(times)/len(times) for times in zip(*psm_runtimes)]
fem_runtimes_avg = [sum(times)/len(times) for times in zip(*fem_runtimes)]

# Update the runtime lists with averages
psm_runtimes = psm_runtimes_avg
fem_runtimes = fem_runtimes_avg


with open('saddle_comparison_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Grid size', 'Number of nodes', 'FEM solver time [s]', 'PSM solver time [s]', 'Maximum difference [mm]'])
    for i, grid_size in enumerate(grid_sizes):
        grid_size_str = f"{grid_size}x{grid_size}"
        writer.writerow([grid_size_str, nodes_lst[i], f"{fem_runtimes[i]:.2e}", f"{psm_runtimes[i]:.2e}", f"{max_diffences[i]:.2e}"])
