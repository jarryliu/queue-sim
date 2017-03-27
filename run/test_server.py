#!/usr/bin/python
import os, sys

import subprocess, shlex
from multiprocessing import Pool
import numpy as np
maxPool = 8



numTest = 8
duration = 500
distrName = "exp"

#testSet = ["onebucket", "onebucket_schedule", "twobuckets"]
testSet = ["servers_test"]

if len(sys.argv) >= 2:
    numTest = int(sys.argv[1])
if len(sys.argv) >= 3:
    duration = float(sys.argv[2])
if len(sys.argv) >= 4:
    distrName = sys.argv[3]

def subprocess_cmd(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    proc_stdout = process.communicate()[0].strip()
    results = proc_stdout.split(" ")
    return [float(r) for r in results]

bRate = 1000.0
bucketSize = 10
distrName = "exp"
mu = 1000.0
pRate = 900.0

tresult = []
sresult = []

pRateList = [100, 200, 300, 400, 500, 600, 700, 800, 900]
for t in testSet:
    print "\n\ntest for ", t
    for pRate in pRateList:
        print "\nbucket size is ", bucketSize
        pNum = 0
        result = []
        while pNum < numTest:
            poolNum = min(maxPool, numTest-pNum)
            pool = Pool(poolNum)
            cmd = []
            for i in xrange(poolNum):
                cmdString = "python "+t+".py " + str(duration) + " " + str(pRate) + " " + str(bRate) + " " + str(int(bucketSize)) + " " + str(mu)
                cmd.append(cmdString)
            re = pool.map(subprocess_cmd, tuple(cmd))
            re = re[0]
            if len(result) != len(re):
                result = [[] for i in xrange(len(re))]
            for r in xrange(len(re)):
                result[r].append(re[r])
            pNum += poolNum
        np.savetxt(f, (tresult, sresult))

print tresult
print sresult

tresult = []
sresult = []

#rates = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
ratio = 2.0
bRate /= ratio
bucketSize /= ratio
for t in testSet:
    print "\n\ntest for ", t
    for pRate in pRateList:
        pRate /= ratio
        print "\nbucket size is ", bucketSize
        pNum = 0
        result = []
        while pNum < numTest:
            poolNum = min(maxPool, numTest-pNum)
            pool = Pool(poolNum)
            cmd = []
            for i in xrange(poolNum):
                cmdString = "python "+t+".py " + str(duration) + " " + str(pRate) + " " + str(bRate) + " " + str(int(bucketSize)) + " " + str(mu)
                cmd.append(cmdString)
            re = pool.map(subprocess_cmd, tuple(cmd))
            re = re[0]
            if len(result) != len(re):
                result = [[] for i in xrange(len(re))]
            for r in xrange(len(re)):
                result[r].append(re[r])
            pNum += poolNum
        tresult.append(np.mean(result[0]))
        sresult.append(np.mean(result[1]))
print tresult
print sresult
