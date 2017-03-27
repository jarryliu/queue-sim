#!/usr/bin/python
import os, sys

import subprocess, shlex
from multiprocessing import Pool
import numpy as np


numTest = 1
duration = 1000
distrNameA = "exp"
distrNameB = "exp"
appDir = "../app/"

#testSet = ["onebucket", "onebucket_schedule", "twobuckets"]
testSet = ["token_test"]
odirName = "./"

if len(sys.argv) >= 2:
    numTest = int(sys.argv[1])
if len(sys.argv) >= 3:
    duration = float(sys.argv[2])
if len(sys.argv) >= 4:
    #if sys.argv[3] == "exp" or sys.argv[3] == "wei":
    distrNameA = sys.argv[3]
if len(sys.argv) >= 5:
    distrNameB = sys.argv[4]
if len(sys.argv) >= 6:
    odirName = sys.argv[4]+'/'

pRate = 1000.0


tresult = []
sresult = []

#pRateList = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
bRateList = np.array([1 + 0.05*(i+1) for i in range(20)])*pRate
bucketSizeList = [i+1 for i in range(8)]
#distrNameAList = ["bstexp1", "bstexp2", "bstexp3", "bstexp4", "bstexp5", "bstexp6", "bstexp7", "bstexp8", "bstexp9", "bstexp10"]
distrNameAList = ["exp"]
pNum = 0

t = testSet[0]
while pNum < numTest:
    #print "********************* Test " + str(pNum+1) + " *********************"
    for bRate in bRateList:
        pList = []
        for distrNameA in distrNameAList:
            for b in bucketSizeList:
                dirName = odirName
                dirName += distrNameA[:3]
                print("\ntest for "+ str(t) + " bucket size " + str(b) + " bucket rate " +str(bRate) + "\n")
                fileName = dirName + "/" + distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + ".out"
                if not os.path.exists(os.path.dirname(fileName)):
                    try:
                        os.makedirs(os.path.dirname(fileName))
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                cmd = ["python", appDir+t+".py",  str(duration),  str(pRate),
                    str(bRate), str(int(b)), distrNameA, fileName]
                pList.append(subprocess.Popen(cmd))
        for p in pList:
            p.wait()
            # exec("scp " + dirName + "/* " + remote_loc + dirName + "/")
            # exec("rm " + dirName + "/*")
    pNum += 1
