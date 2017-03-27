#!/usr/bin/python
import os, sys

import subprocess, shlex
from multiprocessing import Pool
import numpy as np
maxPool = 30



numTest = 30
duration = 500
distrName = "exp"

#testSet = ["onebucket", "onebucket_schedule", "twobuckets"]
testSet = ["onebucket"]

if len(sys.argv) >= 2:
    numTest = int(sys.argv[1])
if len(sys.argv) >= 3:
    duration = float(sys.argv[2])
if len(sys.argv) >= 4:
    #if sys.argv[3] == "exp" or sys.argv[3] == "wei":
    distrName = sys.argv[3]

def subprocess_cmd(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    proc_stdout = process.communicate()[0].strip()
    return float(proc_stdout)

# for t in testSet:
#     print "\n\ntest for ", t
#     for u in [0.8, 0.85, 0.9, 0.95]:
#         print "\nutilization is ", u
#         pNum = 0
#         result = []
#         while pNum < numTest:
#             poolNum = min(maxPool, numTest-pNum)
#             pool = Pool(poolNum)
#             cmd = []
#             for i in xrange(poolNum):
#                 cmdString = "python "+t+".py " + str(duration) + " " + str(u) + " 10 " + distrName
#                 cmd.append(cmdString)
#             result += pool.map(subprocess_cmd, tuple(cmd))
#             pNum += poolNum
#         print result
#         print "mean: ", np.mean(result), "var: ", np.var(result)

for t in testSet:
    print "\n\ntest for ", t
    for b in range(1, 21):
        print "\nbucket size is ", b
        pNum = 0
        result = []
        while pNum < numTest:
            poolNum = min(maxPool, numTest-pNum)
            pool = Pool(poolNum)
            cmd = []
            for i in xrange(poolNum):
                cmdString = "python "+t+".py " + str(duration) + " 0.9 " + str(b) + " " + distrName
                cmd.append(cmdString)
            result += pool.map(subprocess_cmd, tuple(cmd))
            pNum += poolNum
        print result
        print "mean: ", np.mean(result), "var: ", np.var(result)

rates = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
for t in testSet:
    print "\n\ntest for ", t
    for b in rates:
        print "\n rate is ", b
        pNum = 0
        result = []
        while pNum < numTest:
            poolNum = min(maxPool, numTest-pNum)
            pool = Pool(poolNum)
            cmd = []
            for i in xrange(poolNum):
                cmdString = "python "+t+".py " + str(duration) + " 0.9 10 " + " " + distrName + " " + str(b)
                cmd.append(cmdString)
            result += pool.map(subprocess_cmd, tuple(cmd))
            pNum += poolNum
        print result
        print "mean: ", np.mean(result), "var: ", np.var(result)
