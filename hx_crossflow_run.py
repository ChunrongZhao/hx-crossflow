
"""
Run file for heat exchanger crossflow solver

# testing by Chunrong

Author: Andrew Lock
Date edited: October 2023

Note This is mostly old and poorly written code.
One day it will be re-written with best practices.
"""

import csv
import os
import datetime
import pandas as pd
import numpy as np
from itertools import zip_longest
from matplotlib import pyplot as plt

import hx_crossflow_solver as hx_solver


# -----------Model Inputs------------------------------------------

def hx_inputs(G, F, M, CT):
    # Heat exchanger geometric inputs
    G.HX_length = 12 # (m) heat exchanger length
    G.n_rows = 4  # number of rows (deep) of heat exchanger
    G.n_passes = 2 # number of passes of hot fluid over air
    G.pitch_longitudal = 0.0635 * np.cos(np.pi / 6)  # (m) longitudal tube pitch
    G.pitch_transverse = 0.0635  # (m) Transverse tube pitch
    G.ID = 0.0212  # (m) ID of HX pipe
    G.t_wall = 0.0021  # (m) thickness of pipe wall
    G.dx_i = 2.4  # (m) initial length of pipe discretisation
    G.k_wall = 40  # (W/m) thermal conductivity of pipe wall
    G.D_fin = 0.05715  # (m) maximum diameter of fin
    G.pitch_fin = 0.0028  # (m) fin pitch
    G.t_fin = 0.0004  # (m) mean thickness of pipe fin
    G.k_fin = 200  # (m)
    # These are the model inputs (correlations, etc).
    # set correlation for heat transfer in H channel
    # (1)-Yoon f(T_b),
    # (2)-Yoon f(T_b,T_w),
    # (3)-Gnielinski
    # (4)-Pitla
    # (5)-Wang
    # + more (check solver code)
    M.Nu_CorrelationH = "Gnielinski"
    M.alpha_CorrelationC = 1  # set correlation for heat transfer in C channel
    M.row_correction = 1 # correct for different air turbulence for each row
    M.f_CorrelationH = 3  # set correlation for friction factor in H channel
    M.f_CorrelationC = []  # set correlation for friction factor in C channel (usually ignored)
    M.consider_bends = 1  # consider bends in pressure loss? 1-yes, 0-no
    M.bend_loss_coefficient = 1  # bend friction factor (from Chen 2018, for CO2)

    # The type of model
    # (0) - forced convection (specify air velocity)
    # (1) - NDDCT model (includes draft equation)
    M.cooler_type = 0

    # Solver type
    # (0) - Fixed hot fluid pressure difference, solve for mass flow rate
    # (1) - Fixed mass flow rate, solve for all temperatures
    # (3) - Fixed hot fluid intlet and outlet temperature - solve for mass flow rate
    # NOTE: Mode (1) is the normal mode of operation
    M.solver_type = 1

    # Fluid properties. A decent initialisation guess is often required for the
    # solver. Ensure you don't go into two-phase region (solver won't work).
    F.PF_fluid = "H2O"
    F.T_PF_in = 273.15 + 59.96  # (K)  process fluid inlet temperature
    F.T_PF_out = 273.15 + 33  # (K) process fluid outlet temp (initialisation)
    F.P_PF_in = 10e5  # (Pa) process fluid inlet pressure
    F.mdot_PF = 0.1005  # (kg/s) process fluid
    F.P_PF_dp = 2000  # (Pa) pressure drop for model type (0)

    F.Amb_fluid = "Air"
    F.T_air_in = 273.15 + 23  #  (K) air inllet temperature
    F.vC_in = 3  # (m/s) air velocity (starting guess for NDDCT model)
    F.P_amb_in = 101325  # (Pa) air pressure in
    F.T_air_outlet_guess = 273.15 + 40.1  # (K) air out temp guess

    # Inputs for cooling tower model only (ignored for forced convection)
    #
    # These are the cooling tower inputs. Refer to my paper (and Sam Duniam's
    # paper/Kroger) for explanation of the paremters.
    CT.R_Hd = 1.2  # Aspect ratio H5/d3
    CT.R_dD = 0.7  # Diameter ratio d5/d3
    CT.R_AA = 0.65  # Area coverage of heat exchangers Aft/A3
    CT.R_hD = 1 / 6.5  # Ratio of inlet height to diameter H3/d3
    CT.R_sd = 60 // 82.93  # Ratio of number of tower supports to d3
    CT.R_LH = 15.78 / 13.67  # Ratio of height support to tower height L/H3
    CT.D_ts = 0.2  # Diameter of tower supports
    CT.C_Dts = 2  # Drag coefficeint of tower supports
    CT.K_ct = 0.1  # Loss coefficient of cooling tower separation
    CT.sigma_c = 0.725  # Sigma_c, per Kroger
    CT.dA_width = []  # (m2) Frontal area of section of HX (calculated within code)

    # Solver type
    # (0) -  fixed diameter: use solve HX model type (1)
    # (1) - fixed mass flow rate, solve NDDCT diameter: use HX model type (2)
    CT.solver_type = 0
    if M.cooler_type == 1:
        F.vC_in = 1.5  # (m/s) air velocity initial guess
    CT.d3 = 45.5575  # (m) cooling tower inlet diameter (initialisation)
    CT.mdot_PF_total = 221.4  # (kg/s) total cooling tower hot fluid mass flow rate

# --------------------Plot function------------------------------------


def plot(result_dict, G):
    fig, ax = plt.subplots(G.n_rows, 1, sharex=True, figsize=(6, 8))
    ax = ax.flatten()

    pf_dict = {k: result_dict[k] for k in ("row_pf", "x_pf", "T_pf", "p_pf")}
    air_dict = {k: result_dict[k] for k in ("row_air", "x_air", "T_air")}
    wall_dict = {
        k: result_dict[k] for k in ("row_wall", "x_wall", "T_w_h", "T_w_c", "alpha_pf")
    }

    pf_df = pd.DataFrame.from_dict(pf_dict)
    air_df = pd.DataFrame.from_dict(air_dict)
    wall_df = pd.DataFrame.from_dict(wall_dict)

    for i, ax_i in enumerate(ax):
        ax_i.set_ylabel(r"Row " + str(i) + ", T [K]", fontsize=10)
        (l1,) = ax_i.plot(
            pf_df.loc[pf_df["row_pf"] == i]["x_pf"],
            pf_df.loc[pf_df["row_pf"] == i]["T_pf"],
            color="r",
            marker=".",
            linewidth=1,
        )
        (l2,) = ax_i.plot(
            air_df.loc[air_df["row_air"] == i]["x_air"],
            air_df.loc[air_df["row_air"] == i]["T_air"],
            color="darkblue",
            marker=".",
            linewidth=1,
        )
        (l3,) = ax_i.plot(
            air_df.loc[air_df["row_air"] == i + 1]["x_air"],
            air_df.loc[air_df["row_air"] == i + 1]["T_air"],
            color="deepskyblue",
            linewidth=1,
            marker=".",
        )
        (l4,) = ax_i.plot(
            wall_df.loc[wall_df["row_wall"] == i]["x_wall"],
            wall_df.loc[wall_df["row_wall"] == i]["T_w_h"],
            color="darkorange",
            linewidth=1,
        )
        (l5,) = ax_i.plot(
            wall_df.loc[wall_df["row_wall"] == i]["x_wall"],
            wall_df.loc[wall_df["row_wall"] == i]["T_w_c"],
            color="gold",
            linewidth=1,
        )

    ax[-1].set_xlabel("Position [m]", fontsize=10)
    ax[-1].legend(
        [l1, l2, l3, l4, l5],
        [r"Hot fluid", r"Air in", r"Air out", r"Wall hot", r"Wall cold"],
        fontsize=8,
    )
    plt.tight_layout()
    plt.show()


# --------------------Main run script------------------------------------

def main(X0=None, mod=None, verbosity=1):
    ret = hx_solver.solve(hx_inputs, X0=X0, fig_switch=0, mod=mod, verbosity=verbosity)

    result_dict = {
        "row_pf": ret["G"].row_pf,
        "x_pf": ret["G"].x_pf,
        "T_pf": ret["T_pf"],
        "p_pf": ret["p_pf"],
        "row_air": ret["G"].row_air,
        "x_air": ret["G"].x_air,
        "T_air": ret["T_air"],
        "row_wall": ret["G"].row_wall,
        "x_wall": ret["G"].x_wall,
        "T_w_h": ret["T_w_h"],
        "T_w_c": ret["T_w_c"],
        "alpha_pf": ret["alpha_pf"],
        "q_cell": ret["q_cells"],
        "T_pf_out": ret["T_pf_out"],
        "Q_total": ret["Q_total"],
        "mdot_vec": ret["mdot_vec"],
        "dp_pf": ret["dp_pf"],
        "dp_air": ret["dp_air"],
    }

    # Construct filenames
    date = str(datetime.datetime.now().date())
    timestamp = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")
    filename = "hx_results__" + timestamp + ".csv"

    # Create a new results directory for the day if one doesn't already exist
    directory = "results_" + date
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)

    data_lists = [
        [a] if isinstance(a, float) else list(a) for a in result_dict.values()
    ]
    write_data = zip_longest(*data_lists, fillvalue="")

    with open(filepath, "w", newline="") as result_file:
        writer = csv.writer(result_file)
        writer.writerow(result_dict.keys())
        for row in write_data:
            writer.writerow(row)

    # Plot results. Comment out if you don't want to plot.
    plot(result_dict, ret["G"])


# ---------------------------------------------------------------------------
#
if __name__ == "__main__":
    main()
