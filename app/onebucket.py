#!/usr/bin/python

from scheduler import Scheduler
from queue import Queue
from genjob import GenJob
from tokenbucket import TokenBucket
import logging, sys
import numpy as np
logging.basicConfig(stream=sys.stdout, level=logging.INFO)



### one token bucket
numJob = 2
testDuration = 200
utilization = 0.9
bucketSize = 10
distrName = "exp"
pRate = 1000.0 # 1000 per second

if len(sys.argv) >= 2:
    testDuration = float(sys.argv[1])
if len(sys.argv) >= 3:
    utilization = float(sys.argv[2])
if len(sys.argv) >= 4:
    bucketSize = int(sys.argv[3])
if len(sys.argv) >= 5:
    #if sys.argv[4] == "exp" or sys.argv[4] == "wei":
    distrName = sys.argv[4]
if len(sys.argv) >= 6:
    pRate = float(sys.argv[5])

gRate = pRate * utilization / numJob

gen = []
queue = [Queue()]
scheduler = [Scheduler()]
for i in xrange(numJob) :
    gen.append(GenJob(i))
    gen[-1].setOutput(queue[-1])
    #gen[-1].setIntDistr("exp", [1.0/gRate]) # rate 2
    gen[-1].setIntDistr(distrName, [1.0/gRate])
    #gen[-1].setSizeDistr("binorm", [1000.0])

bucket = [TokenBucket(0)]
queue[-1].setOutput(bucket[-1])
bucket[-1].setParameters(pRate, bucketSize) # set rate 1000 per second, bucket size 10
time = 0

while time < testDuration:
    nextTimeList = []
    itemList = []
    for b in bucket:
        nextTime, item  = b.getNextTime()
        if nextTime in nextTimeList:
            print "error, duplicated "
        nextTimeList.append(nextTime)
        itemList.append(item)

    index = nextTimeList.index(min(nextTimeList))
    time = nextTimeList[index]
    item = itemList[index]
    logging.debug("Simulation time %f", time)
    item.runTime(time)

result = []
#for g in gen:
#    g.showStatistic()
for q in queue:
#    q.showStatistic(testDuration/1.2)
    result.append(q.showRate(testDuration/2))
print np.mean(result)

#for b in bucket:
#    b.showStatistic(testDuration/1.2)
