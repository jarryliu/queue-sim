#!/usr/bin/python

from scheduler import Scheduler
from queue import Queue
from genjob import GenJob
from tokenbucket import TokenBucket
import logging, sys
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def updateBuckets(time, bucketList, flag = True):
    size = []
    rate = []
    if flag != True :
        return
    for b in bucketList:
        size.append(b.getQueueSize(time))
        rate.append(b.getRate(time))

    # print "queue size ", size
    # print "estimated rate", rate
    # print "token number", num
    for i in xrange(len(bucketList)):
        rateAdj = -1
        sizeAdj = -1
        if np.sum(rate) != 0:
            rpercent = 1.0 * rate[i] / np.sum(rate)
            rateAdj = rpercent * pRate
            sizeAdj = round(rpercent * bucketSize)

        #print rateAdj, int(sizeAdj), int(numAdj)
        bucketList[i].updateParameters(time, rateAdj, sizeAdj)

### one token bucket
numJob = 2
testDuration = 200
utilization = 0.9
bucketSize = 10
updateInterval = float('inf')
updateFlag = False
distrName = "exp"

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
    updateInterval = float(sys.argv[5])
    updateFlag = True

pRate = 1000.0 # 1000 per second
gRate = pRate * utilization / numJob

gen = []
queue = []
bucket = []
for i in xrange(numJob) :
    gen.append(GenJob(i))
    queue.append(Queue(i))
    gen[-1].setOutput(queue[-1])
    gen[-1].setIntDistr(distrName, [1.0/gRate])
    #gen[-1].setSizeDistr("binorm", [1000.0])
    gen[-1].setSizeDistr("cst", [1])
    bucket.append(TokenBucket(i))
    queue[-1].setOutput(bucket[-1])
    #bucket[-1].setParameters(pRate/numJob, bucketSize/numJob)
    bucket[-1].setParameters(pRate/numJob, bucketSize/numJob)

time = 0
nextUpdate = updateInterval
while time < testDuration:
    nextTimeList = []
    itemList = []
    for b in bucket:
        nextTime, item  = b.getNextTime()
        nextTimeList.append(nextTime)
        itemList.append(item)

    index = [i for i in xrange(len(nextTimeList)) if nextTimeList[i] == min(nextTimeList)]
    time = nextTimeList[index[0]]

    if time >= nextUpdate:
        logging.debug("Update Token time %f", nextUpdate)
        updateBuckets(nextUpdate, bucket, updateFlag)
        nextUpdate += updateInterval
    logging.debug("Simulation time %f", time)
    for i in index :
        itemList[i].runTime(time)

result = []
for q in queue:
    #q.showStatistic(testDuration/2)
    result.append(q.showRate(testDuration/2))
print np.mean(result)




#for b in bucket:
#    b.showStatistic(testDuration/1.2)
