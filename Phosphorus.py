import SurplusElement.mathematics.spectral as spec
import numpy as np
import scipy.linalg as sp_linalg
from fontTools.varLib.mutator import prev
import matplotlib.pyplot as plt
import time as time
import pandas as pd

"""Numerical parameters, matrices and evaluation nodes"""
l = 10
T = 1
n = 20

D = spec.chebDiffMatrix(matrixSize=n, a=0, b=l)
nodes = spec.chebNodes(pointsAmount=n, a=0, b=l)
I = np.eye(n)
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

"""no idea"""
C_enh = 1e-12*1.65*1e+23*np.exp(-0.88/(EKB*TE_K))


"""Defects constants"""
PLI = 10.0
V0 = 200.0
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
V0 = 500.0
alpha_1 = 1.0
beta_1 = -1e-3
ga_1 = 0.0
alpha_2 = 1.0

V0 = 3200.0
alpha_1 = 0.0
beta_1 = 0.0
ga_1 = 1.0
alpha_2 = 1.0
beta_2 = -1.0
ga_2 = 0.0
def chi(C):
    return (C - N + np.sqrt((C - N)**2 + 4 * n_e**2))/(2 * n_e)

def D_E(chi):
    return D_E_i * (1.0 + beta_E_1 * chi + beta_E_2 * chi**2)/(1.0 + beta_E_1 + beta_E_2)

def D_F(chi):
    return D_F_i * (1.0 + beta_F_1 * chi + beta_F_2 * chi ** 2) / (1.0 + beta_F_1 + beta_F_2)

def D_N(chi, C, C_V, C_I):
    return C * (D_E(chi) * C_V + D_F(chi) * C_I) / np.sqrt((C - N)**2 + 4 * n_e**2)

def d_V(x):
    return x
def d_I(x):
    return x

def k_V(x):
    return x
def k_I(x):
    return x

tau = 0.01

prevC = np.ones(n)

for i in range(int(T/tau)):
    prevChi = chi(prevC)
    """Vacancy concentration calculations"""
    v_V = V0*np.exp[-(nodes-RP)**2/(2*DRP**2)]
    g_V = 1.0 + GIM * np.exp(-(nodes - RP)**2/(2 * DRP**2))
    diffCV = D @ (np.diag(D @ d_V(prevChi)) - I @ np.diag(v_V)) + I @ np.diag((k_V(prevChi) / (l_V) ** 2))
    diffCV[0, :] = 0
    diffCV[-1, :] = 0

    diffCV[0, :] = ga_1 * D[0, :] - alpha_1 * I[0, :]
    diffCV[-1, :] = ga_2 * D[-1, :] - alpha_2 * I[-1, :]

    C_V_RHS = - g_V / (l_V) ** 2
    C_V_RHS[0] = beta_1
    C_V_RHS[-1] = beta_2

    C_V = sp_linalg.solve(diffCV, C_V_RHS)

    """Interatomic? concentration calculations"""
    v_I = V0 * np.exp[-(nodes - RP) ** 2 / (2 * DRP ** 2)]
    g_I = 1.0 + GIM * np.exp(-(nodes - RP) ** 2 / (2 * DRP ** 2))
    diffCI = D @ (D @ np.diag(d_I(prevChi)) - I @ np.diag(v_I)) + I @ np.diag(k_I(prevChi) / (l_I) ** 2)
    diffCI[0, :] = 0
    diffCI[-1, :] = 0

    diffCI[0, :] = ga_1 * D[0, :] - alpha_1 * I[0, :]
    diffCI[-1, :] = ga_2 * D[-1, :] - alpha_2 * I[-1, :]

    C_I_RHS = - g_I / (l_I) ** 2
    C_I_RHS[0] = beta_1
    C_I_RHS[-1] = beta_2

    C_I = sp_linalg.solve(diffCI, C_I_RHS)
    """Concentration equation"""
    L = D_E(prevChi) * (D @ C_V) + D_F(prevChi) * (D @ C_I)
    R = D_E(prevChi) * C_V + D_F(prevChi) * C_I + D_N(prevChi, prevC, C_V, C_I)

    diffC = D @ (I @ np.diag(L) + D @ np.diag(R))
    diffC[0, :] = 0
    diffC[-1, :] = 0

    diffC[0, :] = K_s * D[0, :]

    nextC = prevC + tau * diffC
    prevC = nextC



