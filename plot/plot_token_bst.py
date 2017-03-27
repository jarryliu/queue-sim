import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, var, sqrt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from file_process import readLatency

distrNameA = "bstexp10"
dirName = "../run/"+distrNameA+"/"

pRate = 1000.0
bRateList = np.array([1 + 0.05*(i+1) for i in range(20)])*pRate
bucketSizeList = [5*(i+1) for i in range(8)]

test = "token_test"
readLatency(test, [dirName], [distrNameA], bRateList, bucketSizeList)
