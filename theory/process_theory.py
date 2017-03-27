#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import re
from math import sqrt, ceil
from scipy.optimize import curve_fit


# get G/G/1 bounds
def getLatencyGG1(lam, var_a, mu, var_s):
    c_a = sqrt(var_a)*lam
    c_s = sqrt(var_a)*lam
    lower = (lam*var_s - 1/mu*(2-lam/mu))/2/(1-lam/mu) + 1/mu
    upper = lam*(c_a**2 + c_s**2)/2/(1-lam/mu) + 1/mu
    return lower, upper

# get D/D/1 latency
def getLatencyDD1(k, r):
    return (k+1)/2/r

# get D/D/m latency
def getLatencyDDM(k, r, m):
    return (k-1)/2/r/m

# get batched periodic arrival
def bPerVar(m, k):
    return (k-1)*m**2

def getBurstByVar(m, v):
    return ceil(v*m**2 + 1)

# get G/G/m bounds
def getLatencyGGM(lam, var_a, mu, var_s, m):
    xmean = 1/mu*m
    xvar = var_s*m**2
    xsqrt = xmean**2 + xvar
    rho = lam/xmean
    lower = (lam*var_s - 1/mu*(2-lam/mu))/2/(1-lam/mu) + 1/mu - (m-1)*xsqrt/2/m*xmean
    upper = lam*(var_a +xvar/m + (m-1)*xsqrt/m**2)/2/(1-rho/m)
    return lower, upper

def modelPub1(n, a, c):
    return (n+1)/2*a + c

def fitPub1(x, y):
    A, B = curve_fit(modelPub1, x, y)[0]
    return A, B

def modelPub8(n, a, c):
    return (n+1)/2/8*a + c

def fitPub8(x, y):
    A, B = curve_fit(modelPub8, x, y)[0]
    return A, B

def modelBatch1(k, a, c):
    return (k+1)/2*a + c

def fitBatch1(x, y):
    A, B = curve_fit(modelBatch1, x, y)[0]
    return A, B

def modelBatch8(k, a, b, c):
    return ((k+1)/2*a + b)/8 + c

def fitBatch8(x, y):
    A, B, C = curve_fit(modelBatch8, x, y)[0]
    return A, B, C

def modelSimple(k, s, v):
    return s*k + v

def fitSimple(x, y):
    popt, pcov = curve_fit(modelSimple, x, y, bounds = (0.0, [0.1, 1]))
    s, v = popt
    return s, v

def modelSetup(k, s, v):
    # b is the processing time, t is the setup time.
    return s*(k+1)/2 + v

def fitSetup(x, y):
    popt, pcov = curve_fit(modelSetup, x, y)
    s, v = popt
    return s, v

def MXG1B(lam, x, s, v, b):
    return (2*v + lam * v**2 + v*(x-1))/2/(1+lam*v) + (lam * s**2 + s*(x-1))/2/(1-lam*s) + s + b

def modelRateMXG1B(lam, s, v, b):
    return MXG1B(lam, 1, s, v, b)

def modelXMXG1B(x, s, v, b):
    return MXG1B(10, x, s, v, b)

def fitModelXMXG1B(x, y):
    popt, pcov = curve_fit(modelXMXG1B, x, y, p0=[0.01, 0.01, 0.01], bounds=(0, [1, 1, 1]))
    s, v, b = popt
    return s, v, b

def fitModelRateMXG1B(x,y):
    popt, pcov = curve_fit(modelRateMXG1B, x, y, p0=[0.1, 0.1, 0.1], bounds=(0, [1, 1, 1]))
    s, v, b = popt
    return s, v, b

def MXG1(lam, x, s, v):
    #return (2*v + lam * v**2 + v*(x-1))/2/(1+lam*v) + (lam * s**2 + s*(x-1))/2/(1-lam*s) + s
    return (2*v + lam * v**2)/2/(1+lam*v) + (lam * s**2 + s*(x-1))/2/(1-lam*s) + s

def modelRateMXG1(lam, s, v):
    return MXG1(lam, 1, s, v)
    #return (2*v + lam * v**2)/2/(1+lam*v) + (lam * s**2)/2/(1-lam*s) + s

def modelXMXG1(x, s, v):
    return MXG1(10, x, s, v)

def fitModelXMXG1(x, y):
    popt, pcov = curve_fit(modelXMXG1, x, y, p0=[0.001, 0.001], bounds = (0.0, [0.1, 1]))
    s, v = popt
    return s, v

def fitModelRateMXG1(x,y):
    popt, pcov = curve_fit(modelRateMXG1, x, y, p0=[0.00001, 0.0001], bounds = (0.0, [0.01, 1]))
    s, v = popt
    return s, v

if __name__ == "__main__":
    bList = [1, 10, 100, 200, 500, 1000]
    d = [ 0.0285051,   0.04483418,  0.10612187,  0.13303281,  0.2454595,   0.42056296]
    d = [ 0.00790332,  0.0306745,   0.10055516,  0.14903699,  0.31175821,  0.50757778]
    d = [ 0.00259253,  0.01941352,  0.06728919,  0.10172252,  0.21064543,  0.35071596]
    d = [ 0.04970462,  0.07822953,  0.20420267,  0.30418885,  1.59998225,  1.29097012]
    d = [ 0.00710161,  0.02580212,  0.06545459,  0.08921691,  0.15171784,  0.23702999]

    d = [ 0.02813178,  0.22386735,  0.98454178,  1.83693287,  4.46053578,  8.69559633]
    #d = [ 0.03156879, 0.05690863,  0.16209397,  0.28676564,  0.64994968,  1.08822955]
    plt.plot(bList, d, '.')
    a, c = fitPub1(bList, d)
    print(a, c)
    y = [modelPub1(b, a, c) for b in bList]
    plt.plot(bList, y, '-')
    plt.show()
    exit(0)


    testType = "burst"
    s = 4.09674822271e-10
    v = 0.00746867756975
    # s = 2.49535905164e-3
    # v = 1.2e-1
    # b = 3.8e-1
    # s = 8.14440153e-03
    # v = 2.15899696e-02
    # b = 9.81785133e-05

    if testType == "rate":
        #x = [10, 20, 50, 100, 500, 1000, 10000]
        x = np.array([i for i in range(1, 100, 1)])
        y = np.array([modelRateMXG1(i, s, v) for i in x])
        s, v = fitModelRateMXG1(x,y)
        print(s,v)
        plt.plot(x, y, '.')
        y = [modelRateMXG1(i, s, v) for i in x]
        # y = np.array([modelRateMXG1B(i, s, v) for i in x])
        # s, v, b = fitModelRateMXG1B(x,y)
        # print(s,v,b)
        # plt.plot(x, y, '.')
        # y = [modelRateMXG1B(i, s, v, b) for i in x]

    else:
        x = [i for i in range(1, 1002, 1)]
        y = np.array([modelXMXG1(i, s, v)  for i in x])
        s, v = fitModelXMXG1(x,y)
        print(s,v)
        plt.plot(x, y, '.')
        y = [modelXMXG1(i, s, v) for i in x]
        # y = np.array([modelXMXG1B(i, s, v, b)  for i in x])
        # s, v, b = fitModelXMXG1B(x,y)
        # print(s,v,b)
        # plt.plot(x, y, '.')
        # y = [modelXMXG1B(i, s, v, b) for i in x]
        # print(y)

    plt.plot(x, y)
    plt.show()
    exit(0)
