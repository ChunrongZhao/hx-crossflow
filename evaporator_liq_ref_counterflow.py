# evaporator_liq_ref_counterflow.py
#
# Created: Nov 2023, C.R. Zhao
"""
The code is to construct a coolant-refrigerant heat exchanger under counterflow scenario

multilayer stack with each other;
plate fin structures are considered to form micro-channels with each layer

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
#   Components
# ---------------------------------------------------------------------
class Evaporator_Counterflow:
    """
    Inputs:
    refrigerant_name:           -               for example, R134a
    m_dot_refrigerant           kg/s            refrigerant mass flow rate
    h_refrigerant_in            J/kg            refrigerant inlet enthalpy
    P_refrigerant_in            Pa              refrigerant inlet pressure
    SH_degree                   K               refrigerant degree of superheating exiting the evaporator
    m_dot_liq                   kg/s            liquid coolant mass flow rate
    T_liq_in                    K               liquid coolant inlet temperature
    P_liq_in                    Pa              liquid coolant inlet pressure

    Outputs:
    T_liq_out                   K               liquid coolant outlet temperature
    P_liq_out                   Pa              liquid coolant outlet pressure
    effectiveness               -               evaporator effectiveness
    Q_evap                      W               heat transferred by the evaporator
    h_refrigerant_out           J/kg            refrigerant outlet enthalpy
    P_refrigerant_out           Pa              refrigerant outlet pressure
    T_refrigerant_out           K               refrigerant outlet temperature
    x_refrigerant_out           -               refrigerant outlet vapor quality
    x_refrigerant_in            -               refrigerant inlet vapor quality
    L_1                         m               two-phase region length of the evaporator
    L_2                         m               superheated region length of the evaporator
    """

    def __init__(self, refrigerant, coolant, T_coolant_inlet, m_dot_coolant, SH_degree, m_dot_ref, P_ref_in, h_ref_in):
        # working fluid types
        self.refrigerant_name               = refrigerant
        self.heat_transfer_fluid            = coolant                   # liquid coolant
        # air side parameters
        self.T_liq_in                       = T_coolant_inlet           # K, inlet coolant temperature              273
        self.m_dot_liq                      = m_dot_coolant             # kg/s, coolant mass flow rate              0.4
        # refrigerant side parameters
        self.superheating_degree            = SH_degree                 # 11.2 K, level of refrigerant superheating
        self.m_dot_refrigerant              = m_dot_ref                 # kg/s, refrigerant mass flow rate      0.02
        # inlet refrigerant states
        self.P_refrigerant_in               = P_ref_in                       # Pa, inlet pressure
        self.h_refrigerant_in               = h_ref_in                       # K,  inlet temperature
        self.T_refrigerant_in               = CoolProp.PropsSI('T', 'P', self.P_refrigerant_in, 'H', self.h_refrigerant_in, self.refrigerant_name)
        self.x_refrigerant_in               = CoolProp.PropsSI('Q', 'P', self.P_refrigerant_in, 'H', self.h_refrigerant_in, self.refrigerant_name)
        self.rho_refrigerant_in             = CoolProp.PropsSI('D', 'P', self.P_refrigerant_in, 'H', self.h_refrigerant_in, self.refrigerant_name)

    def geometric_calculation(self):
        # liquid coolant side
        self.d_H_liq                        = 5e-3
        self.AP_chan_liq                    = 3
        # # refrigerant side
        # self.d_H_ref                        = 1e-3
        # self.AP_chan_ref                    = 1
        self.b_ref                          = 1e-3

        # other parameters
        self.t_w                            = 5e-4
        self.t_f                            = 1e-4
        self.L_evap                         = 2.
        self.W_evap                         = 1.
        self.N_p_evap                       = 10.
        self.n_rows                         = int(self.N_p_evap)
        self.n_passes                       = 1

        """geometrical parameters"""
        # self.m_dot_ref_channel              = self.m_dot_refrigerant / self.N_p_evap
        self.m_dot_liq_channel              = self.m_dot_liq / self.N_p_evap
        # fin height
        self.b_liq                          = self.d_H_liq * (1 + self.AP_chan_liq) / 2
        # self.b_ref                          = self.d_H_ref * (1 + self.AP_chan_ref) / 2
        # finned area by overall area
        self.A_f_by_A_liq                   = self.AP_chan_liq / (self.AP_chan_liq + 1)
        # self.A_f_by_A_ref                   = self.AP_chan_ref / (self.AP_chan_ref + 1)
        # area density
        self.beta_liq                       = 4 * (1 + self.AP_chan_liq) / (self.d_H_liq * (1 + self.AP_chan_liq) + 2 * self.AP_chan_liq * self.t_f)
        # self.beta_ref                       = 4 * (1 + self.AP_chan_ref) / (self.d_H_ref * (1 + self.AP_chan_ref) + 2 * self.AP_chan_ref * self.t_f)
        # total height
        self.H_evap                         = self.N_p_evap * (self.b_liq + self.b_ref + 2 * self.t_w)

    def discretisation(self, dx_i):
        self.dx_i                           = dx_i
        self.n_CPR                          = int(self.L_evap // self.dx_i + 1)
        self.dx                             = self.L_evap / self.n_CPR
        self.n_cells                        = self.n_CPR * self.n_rows
        self.n_CP                           = int(self.n_rows / self.n_passes)

        self.A_cs_liq                       = self.b_liq * self.W_evap
        self.A_cs_ref                       = self.b_ref * self.W_evap

    def thermodynamic_equations(self):
        """Equations to solve energy balance for each cell in the model, and return
        the energy balance error"""
        # Initialise error vector: TW_H, TW_C, T_liq, T_ref,
        error                       = np.zeros((4 * self.n_cells) + self.n_rows)

        # ------------------------------------------------------------------------------
        # coolant pressure drop calculations
        # ------------------------------------------------------------------------------



        # ------------------------------------------------------------------------------
        # refrigerant pressure drop calculations
        # ------------------------------------------------------------------------------



        # ------------------------------------------------------------------------------
        # heat transfer calculations
        # ------------------------------------------------------------------------------
        for i in range(G.n_cells):
            # Amb node [i] is the outlet to the current cell, i=0,1,2,3,..,23
            node_amb_i1                     = i + G.n_CPR           # Ambient fluid inlet node to current cell; 6,7,8,...,29
            node_amb_i2                     = i + (2 * G.n_CPR)     # Ambient node two prior to current cell; 12,13,14,...,35
            node_amb_i3                     = i - G.n_CPR           # Ambient node following current cell; -6,-5,-4,...,17

            row                             = abs(i) // G.n_CPR     # row = 0, 1, 2, 3
            x                               = (i % G.n_CPR) * G.dx  # [0,1,2,3,4,5]*dx

            if DV[row] == 1:
                x1                          = x                     # row=0,1; [0,1,2,3,4,5]*dx
                x2                          = x + G.dx              # row=0,1; [1,2,3,4,5,6]*dx
            else:
                x1                          = G.HX_length - x - G.dx   # row=2,3; L-[1,2,3,4,5,6]*dx = [5,4,3,2,1,0]*dx
                x2                          = G.HX_length - x          # row=2,3; L-[0,1,2,3,4,5]*dx = [6,5,4,3,2,1]*dx

            if x1 == 0:
                x1                          = 0.001
            if x2 == 0:
                x2                          = 0.001

            # i=0-5,row=0, end at (TH[5]+TH[5])/2; i=6-11,row=1, end at (TH[12]+TH[13])/2;
            # i=12-17,row=2, end at (TH[19]+TH[20])/2; i=18-23,row=3, end at (TH[26]+TH[27])/2;
            k_PF            = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, PH[i], (0.5 * (TH[i + row] + TH[i + 1 + row]))], [CPCP.iconductivity]) # k at centre of i cell
            km_PF           = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, PH[i], TH[i + row]], [CPCP.iconductivity])    # k at left side of i cell
            kp_PF           = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, PH[i], TH[i + 1 + row]], [CPCP.iconductivity])    # k at right side of i cell
            k_amb           = I.k_f(0.5 * (T_air[i] + T_air[node_amb_i1]))  # k_air at centre of i cell
            km_amb          = I.k_f(T_air[node_amb_i1])  # k_air at lower side of i cell
            kp_amb          = I.k_f(T_air[i])  # k_air at upper side of i cell
            # temperature at the cell centre
            Tb_PF           = 0.5 * (TH[i + row] + TH[i + 1 + row])  # bulk temperature at the centre of i cell
            Pb_PF           = 0.5 * (PH[i + row] + PH[i + 1 + row])  # bulk pressure
            Tb_amb          = 0.5 * (T_air[node_amb_i1] + T_air[i])  # bulk temperature of air at the centre of i cell
            # Prandtl number
            Pr_PF           = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, Pb_PF, Tb_PF], [CPCP.iPrandtl])
            Pr_amb          = I.Pr_f(Tb_amb)
            # density
            rho_PF_b        = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, Pb_PF, Tb_PF], [CPCP.iDmass]) # centre of the cell
            rho_PF_w        = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, Pb_PF, TW_H[i]], [CPCP.iDmass])   # hot wall side
            rho_amb         = I.rho_f(Tb_amb)
            # velocity
            U_PF            = abs(mdot_vec[row] / (rho_PF_b * G.A_CS))  # A_CS=pi*ID^2/4
            # viscosity
            mu_PF           = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, Pb_PF, Tb_PF], [CPCP.iviscosity])
            # Reynolds number at i cell of the row
            Re_PF           = rho_PF_b * U_PF * G.ID / mu_PF
            # Nusselt number
            Nu_PF           = calc_Nu(G, F, Re=Re_PF, Pr=Pr_PF, P=Pb_PF, Tb=Tb_PF, Tw=TW_H[i], rho_b=rho_PF_b, rho_w=rho_PF_w,
                                      Correlation=M.Nu_CorrelationH, mu_b=mu_PF, x1=x1, x2=x2, Dh=G.ID)
            # heat transfer coefficient at outer surface of the finned tube, air with wall
            alpha_o         = calc_alpha_amb(G, F, I, Pr=Pr_amb, Tb=Tb_amb, rho_b=rho_amb,
                                             Correlation=M.alpha_CorrelationC,K_c=K_c[row], row_correction=M.row_correction)
            # heat transfer coefficient at inner surface of the tube, PF with wall
            alpha_i         = Nu_PF * k_PF / G.ID
            # fin efficiency based on Schmidt and Zeller, Andrew Lock's thesis, Page 46, Eq. (4.12)
            b               = ((2 * alpha_o) / (G.k_fin * G.t_fin)) ** 0.5
            n_f             = np.tanh(b * G.tube_OD * G.eta_phi / 2) / (b * G.tube_OD * G.eta_phi / 2)

            # ----------------------------------------------------------------------
            # q3, q4    TODO: actually q5 and q6 of Table 4.1, at Page 43 of Andrew Lock's PHD thesis
            # ----------------------------------------------------------------------
            # Convective fluid-wall heat transfer hot side
            q3              = -(TW_H[i] - ((TH[i + row] + TH[i + 1 + row]) / 2)) * alpha_i * G.A_tube_i

            # Convective fluid-wall heat transfer cold side
            q4              = (-(((T_air[node_amb_i1] + T_air[i]) / 2) - TW_C[i]) * alpha_o * ((G.A_f * n_f) + G.A_r))
            # A_r is root area not covered by fins; A_f is fin area

            # ----------------------------------------------------------------------
            # q1, q2
            # ----------------------------------------------------------------------
            # PF side bulk convection
            if DV[row] > 0: # row 0 and 1
                q1_conv     = (GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, PH[i + row], TH[i + row]], [CPCP.iHmass]) * mdot_vec[row] * DV[row]) # left of the cell, enter
                q2_conv     = (GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, PH[i + row + 1], TH[i + row + 1]], [CPCP.iHmass]) * mdot_vec[row] * DV[row] * -1)  # right of the cell, exit

                if i in inlet_cells or header_outlet_cells:
                    q1_cond = (-km_PF * G.A_CS / (G.dx / 2) * (0.5 * (TH[i + row] + TH[i + 1 + row]) - TH[i + row]))      # todo: should it be (*2) instead of (/2)?
                else:
                    q1_cond = (-km_PF * G.A_CS / G.dx * (0.5 * (TH[i + row] + TH[i + 1 + row]) - 0.5 * (TH[i + row] + TH[i - 1 + row])))    # left of the cell, noting the (-) sign

                if i in outlet_cells or header_inlet_cells:
                    q2_cond = (-kp_PF * G.A_CS / (G.dx / 2) * (TH[i + 1 + row] - 0.5 * (TH[i + row] + TH[i + 1 + row])))     # todo: should it be (*2) instead of (/2)?
                else:
                    q2_cond = (-kp_PF * G.A_CS / G.dx * (0.5 * (TH[i + row] + TH[i + 1 + row]) - 0.5 * (TH[i + 1 + row] + TH[i + 2 + row])))  # right of the cell
            else:
                q2_conv     = (GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, PH[i + row], TH[i + row]], [CPCP.iHmass]) * mdot_vec[row] * DV[row]) # DV[row], -1 means reverse direction, left, exit
                q1_conv     = (GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, PH[i + row + 1], TH[i + row + 1]], [CPCP.iHmass]) * mdot_vec[row] * DV[row] * -1)    # right, enter the cell

                if i in inlet_cells or header_outlet_cells:
                    q2_cond = (-km_PF * G.A_CS / G.dx / 2 * (0.5 * (TH[i + row] + TH[i + 1 + row]) - TH[i + row]))
                else:
                    q2_cond = (-km_PF * G.A_CS / G.dx * (0.5 * (TH[i + row] + TH[i + 1 + row]) - 0.5 * (TH[i + row] + TH[i - 1 + row])))

                if i in outlet_cells or header_inlet_cells:
                    q1_cond = (-kp_PF * G.A_CS / G.dx / 2 * (TH[i + 1 + row] - 0.5 * (TH[i + row] + TH[i + 1 + row])))
                else:
                    q1_cond = (-kp_PF * G.A_CS / G.dx * (0.5 * (TH[i + 1 + row] + TH[i + 2 + row]) - 0.5 * (TH[i + row] + TH[i + 1 + row])))

            q1              = q1_cond + q1_conv # enter
            q2              = q2_cond + q2_conv # exit

            # ----------------------------------------------------------------------
            # q5, q6        todo: actually q3, q4 of Andy's thesis
            # ----------------------------------------------------------------------
            # Amb side bulk convection
            q5_conv         = I.h_f(T_air[node_amb_i1]) * F.mdot_C  # upper stream, bottom to the cell; airflow direction, bottom to top; enter
            q6_conv         = I.h_f(T_air[i]) * F.mdot_C    # down stream, top to the cell; exit

            if i >= (G.n_cells - G.n_CPR):  # 18
                q5_cond     = (-km_amb * G.A_amb / (G.pitch_longitudal / 2) * (0.5 * (T_air[node_amb_i1] + T_air[i]) - T_air[node_amb_i1]))       # todo: should it be (*2) instead of (/2)?
            else:
                q5_cond     = (-km_amb * G.A_amb / G.pitch_longitudal * (0.5 * (T_air[node_amb_i1] + T_air[i]) - 0.5 * (T_air[node_amb_i1] + T_air[node_amb_i2])))  # enter the cell

            if i < G.n_CPR: # 6
                q6_cond     = (-km_amb * G.A_amb / (G.pitch_longitudal / 2) * (T_air[i] - 0.5 * (T_air[node_amb_i1] + T_air[i])))      # todo: should it be (*2) instead of (/2)?
            else:
                q6_cond     = (-km_amb * G.A_amb / G.pitch_longitudal * (0.5 * (T_air[i] + T_air[node_amb_i3]) - 0.5 * (T_air[node_amb_i1] + T_air[i])))    # upper-down, -, exit

            q5              = q5_cond + q5_conv # enter
            q6              = q6_cond + q6_conv # exit

            # ----------------------------------------------------------------------
            # q7, q8
            # ----------------------------------------------------------------------
            # Wall conduction
            if i in inlet_cells or header_outlet_cells:
                q7          = 0
            else:
                q7          = (-(((TW_H[i] + TW_C[i]) / 2) - ((TW_H[i - 1] + TW_C[i - 1]) / 2)) * G.k_wall * G.A_WCS / G.dx)

            if i in outlet_cells or header_inlet_cells:
                q8          = 0
            else:
                q8          = (-(((TW_H[i + 1] + TW_C[i + 1]) / 2) - ((TW_H[i] + TW_C[i]) / 2)) * G.k_wall * G.A_WCS / G.dx)

            # ----------------------------------------------------------------------
            # q9
            # ----------------------------------------------------------------------
            q9              = (-(TW_C[i] - TW_H[i]) * G.k_wall * 2 * np.pi * G.dx / np.log((G.ID + (2 * G.t_wall)) / G.ID))

            # ----------------------------------------------------------------------
            # q1-9 errors
            # ----------------------------------------------------------------------
            error[i]                    = q1 + q2 - q3  # * (100/G.n_cells) # 0-23
            error[G.n_cells + i]        = q4 + q5 - q6  # * (100/G.n_cells) # 24-47
            error[2 * G.n_cells + i]    = q3 - q4 + q7 - q8  # * (100/G.n_cells) # 48-71
            error[3 * G.n_cells + i]    = q9 - (min(q3, q4))  # * (100/G.n_cells)   # 72-95

            # flag=0
            if flag == 1:
                Gr                      = 9.81 * (rho_PF_w - rho_PF_b) * rho_PF_b * (G.ID**3) / (mu_PF**2)

                Bu_c                    = Gr / (Re_PF**2)
                Bu_c_array.append(Bu_c)

                Bu_j                    = Bu_c * (rho_PF_b / rho_PF_w) * ((x1 / G.ID) ** 2)
                Bu_j_array.append(Bu_j)

                Gr_p                    = Gr * Nu_PF
                Bu_p                    = Gr_p / (Re_PF**2.75 * Pr_PF**0.5 * (1 + 2.4 * Re_PF ** (-1 / 8) * (Pr_PF ** (2 / 3) - 1)))
                Bu_p_array.append(Bu_p)

                Q1.append(q1)
                Q2.append(q2)
                Q3.append(q3)
                Q4.append(q4)
                Q5.append(q5)
                Q6.append(q6)
                Q7.append(q7)
                Q8.append(q8)
                Q9.append(q9)
                Alpha_i.append(alpha_i)
                Alpha_o.append(alpha_o)
                dT.append(Tb_PF - Tb_amb)

                Ui                      = alpha_i * G.A_tube_i
                Uo                      = alpha_o * ((G.A_f * n_f) + G.A_r)
                Uw                      = G.k_wall * 2 * np.pi * G.dx / np.log((G.ID + (2 * G.t_wall)) / G.ID)
                UA.append((Ui**-1 + Uo**-1 + Uw**-1) ** -1)




# ---------------------------------------------------------------------
#  functions
# ---------------------------------------------------------------------
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


def cal_HTF_ref_2phase_cond(rho_liq, rho_vap, k_2phase, D_2phase_cond, Re_2phase_cond, x_ref, Pr_ref_liq):

    Nu_2phase_cond       = 0.0125 * np.power(Re_2phase_cond * np.sqrt(rho_liq, rho_vap), 0.9) * \
                           np.power(x_ref / (1 - x_ref), 0.1 * x_ref + 0.8) * np.power(Pr_ref_liq, 0.63)
    h_2phase_cond       = k_2phase * Nu_2phase_cond / D_2phase_cond

    return h_2phase_cond
