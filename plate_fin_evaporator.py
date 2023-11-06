# plate_fin_evaporator.py
#
# Created: Nov 2023, C.R. Zhao
"""
The code is to construct a coolant-refrigerant heat exchanger under counterflow scenario

multilayer stack with each other;
plate fin structures are considered to form micro-channels with each layer

https://repository.tudelft.nl/islandora/object/uuid:9a16e0f8-722d-4a17-9f79-96cea2de6906
"""

# ---------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
import numpy                        as np
import CoolProp.CoolProp            as CoolProp
import scipy                        as sci
from scipy                          import optimize
import matplotlib.pyplot            as plt
from statistics                     import mean
from scipy.interpolate              import interp1d


# ---------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------
def main():
    pass


# ---------------------------------------------------------------------
#   Components
# ---------------------------------------------------------------------
class plate_fin_evaporator:
    pass


# ---------------------------------------------------------------------
#  heat transfer coefficients
# ---------------------------------------------------------------------
# single phase liquid coolant
def cal_HTF_liq(k_liq, mu_liq, mu_liq_w, D_liq, Re_liq, Pr_liq):
    # liquid coolant
    if Re_liq > 10:
        C_liq       = 0.348
        n           = 0.663
    else:
        C_liq       = 0.718
        n           = 0.349

    Nu_liq      = C_liq * np.power(Re_liq, n) * np.power(Pr_liq, 1/3) * np.power(mu_liq / mu_liq_w, 0.17)
    h_liq       = k_liq * Nu_liq / D_liq

    return h_liq


# two-phase flow-boiling refrigerant
def cal_HTF_ref_2phase_cond(rho_liq, rho_vap, k_2phase, D_2phase_cond, Re_2phase_cond, x_ref, Pr_ref_liq):

    Nu_2phase_cond       = 0.0125 * np.power(Re_2phase_cond * np.sqrt(rho_liq, rho_vap), 0.9) * \
                           np.power(x_ref / (1 - x_ref), 0.1 * x_ref + 0.8) * np.power(Pr_ref_liq, 0.63)
    h_2phase_cond       = k_2phase * Nu_2phase_cond / D_2phase_cond

    return h_2phase_cond


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
