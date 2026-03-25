import SurplusElement.mathematics.spectral as spec
import numpy as np
import scipy.linalg as sp_linalg
from fontTools.varLib.mutator import prev

l = 10
T = 1
n = 20

D = spec.chebDiffMatrix(matrixSize=n, a=0, b=l)
I = np.eye(n)

# n_e = 1
# N = 10**15

diffC = D.copy()
diffCV = D.copy()
diffCI = D.copy()

def chi(C):
    return (C - N + np.sqrt((C - N)**2 + 4 * n_e**2))/(2 * n_e)

def D_E(chi):
    return D_E_i * (1.0 + beta_E_1 * chi + beta_E_2 * chi**2)/(1.0 + beta_E_1 + beta_E_2)

def D_F(chi):
    return D_F_i * (1.0 + beta_F_1 * chi + beta_F_2 * chi ** 2) / (1.0 + beta_F_1 + beta_F_2)

def D_N(chi, C, C_V, C_I):
    return C * (D_E(chi) * C_V + D_F(chi) * C_I) / np.sqrt((C - N)**2 + 4 * n_e**2)

tau = 0.01

prevC = np.ones(n)

for i in range(int(T/tau)):
    prevChi = chi(prevC)
    """Vacancy concentration calculations"""
    diffCV = D @ (np.diag(D @ d_V(prevChi)) - I @ np.diag(v_V)) + I @ np.diag((k_V(prevChi) / (l_V) ** 2))
    diffCV[0, :] = 0
    diffCV[-1, :] = 0

    diffCV[0, 0] = alpha_V - const
    diffCV[-1, -1] = 1.0

    C_V = sp_linalg.solve(diffCV, - g_V / (l_V) ** 2)

    """Interatomic? concentration calculations"""
    diffCI = D @ (D @ np.diag(d_I(prevChi)) - I @ np.diag(v_I)) + I @ np.diag(k_I(prevChi) / (l_I) ** 2)
    diffCI[0, :] = 0
    diffCI[-1, :] = 0

    diffCI[0, 0] = alpha_I - const
    diffCI[-1, -1] = 1.0

    C_I = sp_linalg.solve(diffCI, - g_I / (l_I) ** 2)
    """Concentration equation"""
    L = D_E(prevChi) * (D @ C_V) + D_F(prevChi) * (D @ C_I)
    R = D_E(prevChi) * C_V + D_F(prevChi) * C_I + D_N(prevChi, prevC, C_V, C_I)

    diffC = D @ (I @ np.diag(L) + D @ np.diag(R))
    diffC[0, :] = 0
    diffC[-1, :] = 0

    diffC[0, :] = K_s * D[0, :]

    nextC = prevC + tau * diffC
    prevC = nextC



