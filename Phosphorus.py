import SurplusElement.mathematics.spectral as spec
import numpy as np
import scipy.linalg as sp_linalg
from fontTools.varLib.mutator import prev
import matplotlib.pyplot as plt
import time as time
import pandas as pd
import scipy.interpolate as interp
"""Numerical parameters, matrices and evaluation nodes"""
l = 0.5
T = 1
n_c = 50
n_d = 500
D = spec.chebDiffMatrix(matrixSize=n_c, a=0, b=l)
D2 = D @ D

defectD = spec.chebDiffMatrix(matrixSize=n_d, a=0, b=200)
defectNodes = spec.chebNodes(pointsAmount=n_d, a=0, b=200)
defectD2 = defectD @ defectD

nodes = spec.chebNodes(pointsAmount=n_c, a=0, b=l)
I = np.eye(n_c)
defectI = np.eye(n_d)
diffC = D.copy()
diffCV = D.copy()
diffCI = D.copy()

"""Main constants"""
"""N := EN from fortran code"""
N = 1e+3
"""n_e := ENI from fortran code"""
ENIO = 3.87298*1e+16
ENIE = 0.605
ENIF = 1.5
EKB = 8.61709715681519*1e-5
TE = 900
TE_K = TE + 273.15
n_e = 1e-12*ENIO*np.exp(-ENIE/(EKB*TE_K))*TE_K**ENIF
K_S = 1.0
"""no idea"""
C_enh = 1e-12*1.65*1e+23*np.exp(-0.88/(EKB*TE_K))


"""Defects constants"""
PLI = 10.0
# V0 = 200.0
RP = 0.0108
DRP = 0.0068
l_V = 10.0
l_I = l_V
GIM = 2.7*1e+4
def read_data():
    df_c_0 = pd.read_csv('C_T0.dat', sep='\s+', header=None,
                          converters={i: lambda x: float(x.replace('D', 'E'))
                                      for i in range(2)})
    c_0 = df_c_0.values

    df_c_i = pd.read_csv('C_I.dat', sep='\s+', header=None,
                         converters={i: lambda x: float(x.replace('D', 'E'))
                                     for i in range(2)})
    c_i = df_c_i.values

    df_c_v = pd.read_csv('C_V.dat', sep='\s+', header=None,
                         converters={i: lambda x: float(x.replace('D', 'E'))
                                     for i in range(2)})
    c_v = df_c_v.values

    df_c_ph = pd.read_csv('C_PH.dat', sep='\s+', header=None,
                      converters={i: lambda x: float(x.replace('D', 'E'))
                                 for i in range(2)})
    c_ph = df_c_ph.values


    df_ph_0 = pd.read_csv('ph-impl_o.dat', sep='\s+', header=None,
                      converters={i: lambda x: float(x.replace('D', 'E'))
                                 for i in range(2)})
    ph_0 = df_ph_0.values

    df_ph = pd.read_csv('ph-impl.dat', sep='\s+', header=None,
                          converters={i: lambda x: float(x.replace('D', 'E'))
                                      for i in range(2)})
    ph = df_ph.values

    return c_ph, c_0, c_i, c_v, ph_0, ph

beta_F_1 = 4.0
beta_F_2 = 0.1
beta_E_1 = 0.0076
beta_E_2 = 0.0001
D_E_i = 4.0*1e-7
D_F_i = 1.5*1e-7
"""There are two variants of the following constants in the fortran code
alpha, beta, ga are included into the boundary conditions for defects equations
ga * dc/dx = alpha * c + beta"""
# V0 = 500.0
alpha_1 = 1.0
beta_1_I = -1e-3
ga_1 = 0.0
alpha_2 = 1.0

# V0 = 3200.0
alpha_1 = 0.0
beta_1_V = -10.0
ga_1 = 1.0
alpha_2 = 1.0
beta_2 = -1.0
ga_2 = 0.0
V0_I = 200.0
V0_V = 0.0
def chi(C):
    return (C - N + np.sqrt((C - N)**2 + 4 * n_e**2))/(2 * n_e)

def D_E(chi):
    return D_E_i * (1.0 + beta_E_1 * chi + beta_E_2 * chi**2)/(1.0 + beta_E_1 + beta_E_2)

def D_F(chi):
    return D_F_i * (1.0 + beta_F_1 * chi + beta_F_2 * chi ** 2) / (1.0 + beta_F_1 + beta_F_2)

def D_N(chi, C, C_V, C_I):
    return C * (D_E(chi) * C_V + D_F(chi) * C_I) / np.sqrt((C - N)**2 + 4 * n_e**2)

def d_V(x):
    return x*0.0 + 1.0
def d_I(x):
    return x*0.0 + 1.0

def k_V(x):
    return x*0.0 + 1.0
def k_I(x):
    return x*0.0 + 1.0

tau = 0.01

np.set_printoptions(precision=3, suppress=True)
data = read_data()
"""1e-12 is a conversion from 1/cm3 -> 1/mum3"""
# c0 = data[1]*1e-12
c0 = data[1]
# prevC = interp.interp1d(x=c0[:, 0], y=c0[:, 1], kind="linear", fill_value=0.0)(nodes)
prevC = np.interp(x=nodes, xp=c0[:, 0], fp=c0[:, 1], right=0.0)
# plt.plot(data[0][:, 0], data[0][:, 1], label="C")
# plt.legend()
# plt.show()
# plt.plot(data[2][:, 0], data[2][:, 1], label="c_i")
# plt.plot(data[3][:, 0], data[3][:, 1], label="c_v")
# plt.legend()
# plt.show
# plt.plot(nodes, prevC)
# plt.show()

# def defectsCalculation():
#         prevChi = chi(prevC)
#         """Vacancy concentration calculations"""
#         v_V = V0_V*np.exp(-(nodes-RP)**2/(2*DRP**2))
#         g_V = 1.0 + GIM * np.exp(-(nodes - RP)**2/(2 * DRP**2))
#         diffCV = (np.diag(d_V(prevChi)) @ D2 -
#                   np.diag(D @ v_V) @ I - np.diag(v_V) @ D
#                   - np.diag((k_V(prevChi) / (l_V) ** 2))) @ I
#
#         diffCV[0, :] = ga_1 * D[0, :] - alpha_1 * I[0, :]
#         # print(diffCV)
#         diffCV[-1, :] = ga_2 * D[-1, :] - alpha_2 * I[-1, :]
#
#         C_V_RHS = - g_V / (l_V) ** 2
#         C_V_RHS[0] = beta_1_V
#         C_V_RHS[-1] = beta_2
#         # print(C_V_RHS)
#
#         C_V = sp_linalg.solve(diffCV, C_V_RHS)
#
#
#         """Interatomic? concentration calculations"""
#         v_I = V0_I * np.exp(-(nodes - RP) ** 2 / (2 * DRP ** 2))
#         g_I = 1.0 + GIM * np.exp(-(nodes - RP) ** 2 / (2 * DRP ** 2))
#         diffCI = (np.diag(d_I(prevChi)) @ D2 -
#                   np.diag((D @ v_I)) @ I - np.diag(v_I) @ D -
#                   I @ np.diag(k_I(prevChi) / (l_I) ** 2))
#
#         diffCI[0, :] = ga_1 * D[0, :] - alpha_1 * I[0, :]
#         diffCI[-1, :] = ga_2 * D[-1, :] - alpha_2 * I[-1, :]
#
#         C_I_RHS = - g_I / (l_I) ** 2
#         C_I_RHS[0] = beta_1_I
#         C_I_RHS[-1] = beta_2
#
#         C_I = sp_linalg.solve(diffCI, C_I_RHS)
#         return C_I, C_V

def defectsCalculation():
    """Vacancy concentration calculations"""
    v_V = V0_V * np.exp(-(defectNodes - RP) ** 2 / (2 * DRP ** 2))
    g_V = 1.0 + GIM * np.exp(-(defectNodes - RP) ** 2 / (2 * DRP ** 2))
    diffCV = (defectD2 -
              np.diag(defectD2 @ v_V) @ defectI - np.diag(v_V) @ defectD2
              - np.diag((np.ones(defectD2.shape[0]) / (l_V) ** 2))) @ defectI

    diffCV[0, :] = ga_1 * defectD[0, :] - alpha_1 * defectI[0, :]
    # print(diffCV)
    diffCV[-1, :] = ga_2 * defectD[-1, :] - alpha_2 * defectI[-1, :]

    C_V_RHS = - g_V / (l_V) ** 2
    C_V_RHS[0] = beta_1_V
    C_V_RHS[-1] = beta_2
    # print(C_V_RHS)

    C_V = sp_linalg.solve(diffCV, C_V_RHS)

    """Interatomic? concentration calculations"""
    v_I = V0_I * np.exp(-(defectNodes - RP) ** 2 / (2 * DRP ** 2))
    g_I = 1.0 + GIM * np.exp(-(defectNodes - RP) ** 2 / (2 * DRP ** 2))
    diffCI = (defectD2 -
              np.diag((defectD @ v_I)) @ defectI - np.diag(v_I) @ defectD -
              np.diag(np.ones(defectD2.shape[0]) / (l_I) ** 2) @ defectI)

    diffCI[0, :] = ga_1 * defectD[0, :] - alpha_1 * defectI[0, :]
    diffCI[-1, :] = ga_2 * defectD[-1, :] - alpha_2 * defectI[-1, :]

    C_I_RHS = - g_I / (l_I) ** 2
    C_I_RHS[0] = beta_1_I
    C_I_RHS[-1] = beta_2

    C_I = sp_linalg.solve(diffCI, C_I_RHS)
    return C_I, C_V

C_I_large_grid, C_V_large_grid = defectsCalculation()
C_I = np.interp(x=nodes, xp=defectNodes, fp=C_I_large_grid)
C_V = np.interp(x=nodes, xp=defectNodes, fp=C_V_large_grid)

plt.plot(nodes, C_I)
plt.plot(nodes, C_V)
plt.show()

plt.plot(defectNodes, C_I_large_grid)
plt.plot(defectNodes, C_V_large_grid)
plt.show()
for i in range(int(T/tau)):
    prevChi = chi(prevC)
    # plt.plot(nodes, prevChi)
    # plt.show()
    """Concentration equation"""
    L = D_E(prevChi) * (D @ C_V) + D_F(prevChi) * (D @ C_I)
    R = D_E(prevChi) * C_V + D_F(prevChi) * C_I + D_N(prevChi, prevC, C_V, C_I)

    diffC = (np.diag(L) @ I + np.diag(R) @ D) @ D
    # print(diffC.shape)
    # diffC[0, :] *= 0
    # diffC[-1, :] *= 0



    A_implicit = I - tau * diffC
    # A_implicit[0, :] = (I @ np.diag(L) + np.diag(R) @ D - 1e+21*K_S * I)[0, :]
    A_implicit[0, :] = (np.diag(L) @ I + np.diag(R) @ D - K_S * I)[0, :]
    A_implicit[-1, :] = (np.diag(L) @ I + np.diag(R) @ D)[-1, :]
    RHS_implicit = prevC
    RHS_implicit[0] = 0
    RHS_implicit[-1] = 0
    nextC = sp_linalg.solve(A_implicit, RHS_implicit)
    # plt.plot(nodes, C_I)
    # plt.plot(nodes, C_V)
    # plt.show()
    plt.plot(nodes, prevC)
    plt.plot(nodes, nextC)
    plt.show()
    prevC = nextC

    # time.sleep(500)
plt.plot(nodes, nextC)
plt.show()


