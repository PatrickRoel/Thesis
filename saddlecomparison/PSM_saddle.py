"""
Script for PS framework validation, benchmark case where saddle form of self stressed network is sought
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from PSS.particleSystem import ParticleSystem


def psm_saddle_init(grid_size,grid_length,grid_height):


    def connectivity_matrix(grid_size: int, params: dict):
        n = grid_size**2 + (grid_size - 1) ** 2
        top_edge = [i for i in range(grid_size)]
        bottom_edge = [n - grid_size + i for i in range(grid_size)]
        left_edge = [(grid_size * 2 - 1) * i for i in range(1, grid_size - 1)]
        right_edge = [left_edge[i] + grid_size - 1 for i in range(grid_size - 2)]
        fixed_nodes = top_edge + bottom_edge + left_edge + right_edge

        connections = []

        # inner grid connections
        for i in range(n):
            if i not in fixed_nodes:
                connections.append([i, i - grid_size, params["k"], params["c"]])
                connections.append([i, i - grid_size + 1, params["k"], params["c"]])
                connections.append([i, i + grid_size - 1, params["k"], params["c"]])
                connections.append([i, i + grid_size, params["k"], params["c"]])

        # Filtering duplicates
        filtered_connections = []
        for link in connections:
            inverted_link = [link[1], link[0], *link[2:]]
            if (
                inverted_link not in filtered_connections
                and link not in filtered_connections
            ):
                filtered_connections.append(link)
        print(
            f"Filtered connections from {len(connections)} down to {len(filtered_connections)}"
        )
        return filtered_connections, fixed_nodes


    def initial_conditions(
        g_size: int, m_segment: float, fixed_nodes: list, g_h: float, g_l: float
    ):
        conditions = []

        orthogonal_distance = g_l / (g_size - 1)
        dl = g_h / g_l * orthogonal_distance
        even = [i * orthogonal_distance for i in range(g_size)]
        uneven = [
            i * orthogonal_distance + 0.5 * orthogonal_distance for i in range(g_size - 1)
        ]
        x_y = [[i * orthogonal_distance, 0] for i in range(g_size)]

        for i in range(g_size - 1):
            x_y.extend(
                list(
                    zip(
                        uneven,
                        [
                            i * orthogonal_distance + 0.5 * orthogonal_distance
                            for j in range(len(uneven))
                        ],
                    )
                )
            )
            x_y.extend(
                list(zip(even, [(i + 1) * orthogonal_distance for j in range(len(even))]))
            )

        z = [i * dl for i in range(g_size)]
        temp = z.copy()
        z.extend(reversed(temp))
        z.extend(temp[1:-1])
        z.extend(reversed(temp[1:-1]))

        n = g_size**2 + (g_size - 1) ** 2

        for i in range(n):
            if i in fixed_nodes:
                conditions.append(
                    [list(x_y[i]) + [z[fixed_nodes.index(i)]], [0, 0, 0], m_segment, True]
                )
            else:
                conditions.append([list(x_y[i]) + [g_h / 2], [0, 0, 0], m_segment, False])

        return conditions


    # dictionary of required parameters
    params = {
        # model parameters
        "n": 10,  # [-]       number of particles
        "k_t": 1,  # [N/m]     spring stiffness
        "c": 1,  # [N s/m] damping coefficient
        "L": 10,  # [m]       tether length
        "m_block": 100,  # [kg]     mass attached to end of tether
        "rho_tether": 0.1,  # [kg/m]    mass density tether
        # simulation settings
        "dt": 0.1,  # [s]       simulation timestep
        "t_steps": 1000,  # [-]      number of simulated time steps
        "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
        "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
        "max_iter": 1e5,  # [-]       maximum number of iterations
        # physical parameters
        "g": 9.807,  # [m/s^2]   gravitational acceleration
        "v_w": [5, 0, 0],  # [m/s]     wind velocity vector
        "rho": 1.225,  # [kg/ m3]  air density
        "c_d_bridle": 1.05,  # [-]       drag-coefficient of bridles
        "d_bridle": 0.02,  # [m]       diameter of bridle lines
    }

    # calculated parameters
    params["l0"] = 0  # np.sqrt( 2 * (grid_length/(grid_size-1))**2)
    params["m_segment"] = 1
    params["k"] = params["k_t"] * (params["n"] - 1)  # segment stiffness
    params["n"] = grid_size**2 + (grid_size - 1) ** 2


    # instantiate connectivity matrix and initial conditions array
    connections, f_nodes = connectivity_matrix(grid_size, params)
    init_cond = initial_conditions(
        grid_size, params["m_segment"], f_nodes, grid_height, grid_length
    )
    return (connections, init_cond, params)


# def instantiate_ps():
#     return ParticleSystem(connections, init_cond, params)


def solve(ps):
    connections,init_cond,params = ps
    psystem = ParticleSystem(connections, init_cond, params)
    psystem.stress_self()

    n = params["n"]
    t_vector = np.linspace(
        params["dt"],
        params["t_steps"] * params["dt"],
        params["t_steps"],
    )

    x = {}
    for i in range(n):
        x[f"x{i + 1}"] = np.zeros(len(t_vector))
        x[f"y{i + 1}"] = np.zeros(len(t_vector))
        x[f"z{i + 1}"] = np.zeros(len(t_vector))


    position = pd.DataFrame(index=t_vector, columns=x)
    final_step = 0

    n = params["n"]
    f_ext = np.array([[0, 0, 0] for i in range(n)]).flatten()

    start_time = time.time()
    for (
        step
    ) in t_vector:  # propagating the simulation for each timestep and saving results
        position.loc[step], _ = psystem.kin_damp_sim(f_ext)
        final_step = step
        if np.linalg.norm(psystem.f_int) <= 1e-3:
            print("Kinetic damping PS converged")
            break
    stop_time = time.time()

    kinetic_runtime = stop_time - start_time

    print(f"PS kinetic: {kinetic_runtime:.4f} s")

    # plotting & graph configuration
    # Data from layout after 1 iteration step
    X = []
    Y = []
    Z = []
    for i in range(n):
        X.append(position[f"x{i + 1}"].iloc[0])
        Y.append(position[f"y{i + 1}"].iloc[0])
        Z.append(position[f"z{i + 1}"].iloc[0])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # data from final timestep
    X_f = []
    Y_f = []
    Z_f = []
    for i in range(n):
        X_f.append(position[f"x{i + 1}"].loc[final_step])
        Y_f.append(position[f"y{i + 1}"].loc[final_step])
        Z_f.append(position[f"z{i + 1}"].loc[final_step])

    coords = [[X_f[i], Y_f[i], Z_f[i]] for i in range(n)]

    return coords, kinetic_runtime

def psm_saddle(grid_size,grid_length,grid_height):
    ps2 = psm_saddle_init(grid_size,grid_length,grid_height)
    coords,kinetic_runtime = solve(ps2)
    coords = np.array(coords)
    return coords,kinetic_runtime

if __name__ == "__main__":
    grid_size = 12
    grid_length = 10
    grid_height = 11
    coords,runtime = psm_saddle(grid_size,grid_length,grid_height)