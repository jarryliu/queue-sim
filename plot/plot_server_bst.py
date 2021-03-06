import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, var, sqrt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from file_process import readLatency, readServerLatency

distrNameA = "bstexp10"
distrNameB = "cst"
dirName = "../run/"+distrNameA+"/"

mu = 2000.0
pRate = 1000.0
bRateList = np.array([1 + 0.05*(i+1) for i in range(20)])*pRate
bucketSizeList = [5*(i+1) for i in range(8)]

test = "servers_test"
readLatency(test, [dirName], [distrNameA], bRateList, bucketSizeList, pRate, distrNameB, mu)
