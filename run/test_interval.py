#!/usr/bin/python
import os, sys

import subprocess, shlex
from multiprocessing import Pool
import numpy as np
maxPool = 30



numTest = 30
duration = 500
distrName = "exp"

if len(sys.argv) >= 2:
    numTest = int(sys.argv[1])
if len(sys.argv) >= 3:
    duration = float(sys.argv[2])
if len(sys.argv) >= 4:
    distrName = sys.argv[3]

def subprocess_cmd(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    proc_stdout = process.communicate()[0].strip()
    return float(proc_stdout)

# for t in ["onebucket", "onebucket_schedule"]:
#     print "\n for test", t
#     pNum = 0
#     result = []
#     while pNum < numTest:
#         poolNum = min(maxPool, numTest-pNum)
#         pool = Pool(poolNum)
#         cmd = []
#         for i in xrange(poolNum):
#             cmdString = "python " + t + ".py " + str(duration) + " 0.9 20 " + distrName
#             cmd.append(cmdString)
#         result += pool.map(subprocess_cmd, tuple(cmd))
#         pNum += poolNum
#     print result
#     print "mean: ", np.mean(result), "var: ", np.var(result)

for interval in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]:
    print "\ninterval is ", interval
    pNum = 0
    result = []
    while pNum < numTest:
        poolNum = min(maxPool, numTest-pNum)
        pool = Pool(poolNum)
        cmd = []
        for i in xrange(poolNum):
            cmdString = "python twobuckets.py " + str(duration) + " 0.9 20 " + distrName + " "+ str(interval)
            cmd.append(cmdString)
        result += pool.map(subprocess_cmd, tuple(cmd))
        pNum += poolNum
    print result
    print "mean: ", np.mean(result), "var: ", np.var(result)
