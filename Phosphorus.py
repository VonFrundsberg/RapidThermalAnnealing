import SurplusElement.mathematics.spectral as spec
import numpy as np
from fontTools.varLib.mutator import prev

l = 10
T = 1
n = 20

D = spec.chebDiffMatrix(matrixSize=n, a=0, b=l)
I = np.eye(n)

n_e = 1
N = 10**15

def chi(C):
    return (C - N + np.sqrt((C - N)**2 + 4 * n_e**2))/(2 * n_e)

def D_E(chi):
    return D_E_i * (1.0 + beta_E_1 * chi + beta_E_2 * chi**2)/(1.0 + beta_E_1 + beta_E_2)

def D_F(chi):
    return D_F_i * (1.0 + beta_F_1 * chi + beta_F_2 * chi ** 2) / (1.0 + beta_F_1 + beta_F_2)

def D_N(chi, C, C_V, C_I):
    return C * (D_E(chi) * C_V + D_F(chi) * C_I) / np.sqrt((C - N)**2 + 4 * n_e**2)

tau = 0.01

prevChi = chi(prevC)
nextC = prevC + tau * D @ (
        D_E(prevChi) * D @ (prevC_V * prevC) +
        D_F(prevChi) * D @ (prevC_i * prevC) +
        D_N(prevChi, prevC, prevC_V, prevC_I) * D @ prevC
)

D @ (d_V(prevChi) * D @ C_V - v_V * C_V) - k_V(prevChi) * C_V / (l_V)**2 + g_V / (l_V)**2 = 0
D @ (d_I(prevChi) * D @ C_I - v_I * C_I) - k_I(prevChi) * C_I / (l_I)**2 + g_I / (l_I)**2 = 0