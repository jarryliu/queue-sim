#!/usr/local/bin/python

import numpy as np
from scipy.stats import truncnorm

def getRandomDist(distName, para):
    mean = para[0]
    if distName.startswith("bst"):
        distName = distName[3:6]
    new = float('inf')
    if distName == "exp":
        new = np.random.exponential(mean)
    elif distName == "cst":
        new = mean
    elif distName == "wei":
        scale = mean * 1.0/24
        new = scale* np.random.weibull(0.25) # mean, 24
    elif distName == "binorm":
        if np.random.rand() < para[0]:
            new = 1500
        else:
            new = 60
    elif distName == "norm":
        new = int(para[0].rvs(1))
    return new

def setDistr(distName, para):
    if distName == "exp" or distName == "cst" or distName == "wei":
        return [distName, para]
    elif distName == "binorm":
        mean = para[0]
        p = (mean-60.0)/(1500.0-60.0)
        return [distName, [p]]
    elif distName == "norm":
        mean = para[0]
        sigma = para[1]
        x = stats.truncnorm( (60.0 - mean)/sigma, (1500.0 - mean)/sigma, loc = mean, scale = sigma)
        return [distName, [x]]
    elif distName == "markov":
        mean = para[0]
        return [distName, [mean, 0.7*mean, 1.3*mean]]
    elif distName.startswith('bst'):
        burst = int(distName[6:])
        para = [p*burst for p in para]
        return [distName, para + [burst]]
