import SurplusElement.mathematics.spectral as spec
import SurplusElement.GalerkinMethod.Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass
import GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils


import numpy as np
import scipy.linalg as sp_linalg
from fontTools.varLib.mutator import prev
import matplotlib.pyplot as plt
import time as time
import pandas as pd
import scipy.interpolate as interp

defectsC_I = galerkin.GalerkinMethod1d("LE")
defectsC_V = galerkin.GalerkinMethod1d("LE")
defectsMesh = MeshClass.mesh(1)
defectsApproximationOrder = 10

def generateDefectsMesh():
    file = open("elementsDataDefects.txt", "w")
    file = open("neighboursDataPoisson.txt", "w")
    file.write("0.0 inf " + str(defectsApproximationOrder) + " 1.0")
    file.close()
    defectsMesh.fileRead("elementsDataPoisson.txt", "neighboursDataPoisson.txt")


def poissonOperator():
    gradForm = "-integral x * x grad(u) @ grad(v)"
    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: -x * x, integrationPointsAmount)
    return gradForm