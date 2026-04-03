import SurplusElement.mathematics.spectral as spec
import SurplusElement.GalerkinMethod.Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils


import numpy as np
import scipy.linalg as sp_linalg
from fontTools.varLib.mutator import prev
import matplotlib.pyplot as plt
import time as time
import pandas as pd
import scipy.interpolate as interp


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
beta_F_1 = 4.0
beta_F_2 = 0.1
beta_E_1 = 0.0076
beta_E_2 = 0.0001
D_E_i = 4.0*1e-7
D_F_i = 1.5*1e-7

"""Defects constants"""
PLI = 10.0
V0 = 200.0
RP = 0.0108
DRP = 0.0068
l_V = 10.0
l_I = l_V
GIM = 2.7*1e+4
RPDRP = 0.015
X_STAR = 0.02

alpha_1 = 1.0
ga_1 = 0.0
alpha_2 = 1.0
beta_2 = -1.0
beta_1_I = -1e-3
beta_1_V = -10.0
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

defectsC_I = galerkin.GalerkinMethod1d("LS")
defectsC_V = galerkin.GalerkinMethod1d("LS")
defectsMesh = MeshClass.mesh(1)
defectsApproximationOrder: int = 100
integrationPointsAmount: int = 500

def setDefectsMesh():
    fileElements = open("elementsDataDefects.txt", "w")
    fileNeighbours = open("neighboursDataDefects.txt", "w")
    fileElements.write("0.0 0.1 " + str(defectsApproximationOrder) + " 0.0" + "\n")
    fileElements.write("0.1 inf " + str(defectsApproximationOrder) + " 1.0")
    fileElements.close()
    fileNeighbours.write("0 1")
    defectsMesh.fileRead("elementsDataDefects.txt", "neighboursDataDefects.txt")

setDefectsMesh()
def defectForms(v, g, l):
    innerForm1 = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * 0.0 - 1.0, integrationPointsAmount)

    def boundaryForm11(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: x*0.0 + 1.0)

    def boundaryForm12(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x*0.0 + 1.0)

    innerForm2 = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
        trialElement, testElement, weight=lambda x: v(x), integrationPointsAmount=integrationPointsAmount)

    boundaryForm2 = lambda trialElement, testElement: elem1dUtils.evaluateDG_ErrorComponent(
        trialElement, testElement, weight=lambda x: -v(x))

    innerForm3 = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
        trialElement, testElement, lambda x: x * 0.0 - 1/l**2, integrationPointsAmount)

    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: -g(x)/l**2, weight=lambda x: x * 0.0 + 1.0,
        integrationPointsAmount=integrationPointsAmount)

    return innerForm1, boundaryForm11, boundaryForm12, innerForm2, boundaryForm2, innerForm3, functional

boundaryConditionsC_I = ['{"boundaryPoint": "0.0", "boundaryValue": 1e-3}',
                      '{"boundaryPoint": "np.inf", "boundaryValue": 1}']
boundaryConditionsC_V = ['{"boundaryPoint": "0.0", "boundaryValue": 10}',
                      '{"boundaryPoint": "np.inf", "boundaryValue": 1}']

def defectsCalculation():
    """Vacancy concentration calculations"""
    # v_V = V0_V * np.exp(-(defectNodes - RP) ** 2 / (2 * DRP ** 2))

    # g_V = 1.0 + GIM * np.exp(-(defectNodes - RP) ** 2 / (2 * DRP ** 2))
    def v_V(x):
        x = np.atleast_1d(x)
        result = np.zeros(x.shape)
        lessThanXSTAR = np.where(x <= X_STAR)
        moreThanXSTAR = np.where(x >= X_STAR)
        result[lessThanXSTAR] = V0_V
        result[moreThanXSTAR] = V0_V * np.exp(-(x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2))
        return result
    def g_V(x):
        x = np.atleast_1d(x)
        result = np.zeros(x.shape)
        lessThanXSTAR = np.where(x <= X_STAR)
        moreThanXSTAR = np.where(x >= X_STAR)
        result[lessThanXSTAR] = 1.0 + GIM
        result[moreThanXSTAR] = 1.0 + GIM * np.exp(-(x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2))
        return result
    C_V_Forms = defectForms(v_V, g_V, l_V)
    defectsC_V.setBilinearForm(innerForms=[C_V_Forms[0], C_V_Forms[3], C_V_Forms[5]],
                               boundaryForms=[C_V_Forms[1], C_V_Forms[2], C_V_Forms[4]])
    defectsC_V.initializeMesh(defectsMesh)
    defectsC_V.initializeElements()
    defectsC_V.recalculateRHS(functionals=[C_V_Forms[6]])
    defectsC_V.calculateElements()
    sol = defectsC_V.solveSLAE()
    # defectsC_V.
    plt.plot(sol)
    plt.show()

    # return C_I, C_V

defectsCalculation()