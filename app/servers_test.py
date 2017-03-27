#!/usr/bin/python
import sys
sys.path.append('../')
from sim import Scheduler, Queue, GenJob, TokenBucket, Server
import logging, sys
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

### one token bucket
numJob = 1
testDuration = 500.0
bRate = 1000.0
bucketSize = 20
distrNameA = "bstcst10"
distrNameB = "cst"
mu = 1000.0
pRate = 600.0
fileName = "test.out"
if len(sys.argv) >= 2:
    testDuration = float(sys.argv[1])
if len(sys.argv) >= 3:
    pRate = float(sys.argv[2]) # arrive rate
if len(sys.argv) >= 4:
    bRate = float(sys.argv[3]) # bucket limiting rate
if len(sys.argv) >= 5:
    bucketSize = int(sys.argv[4]) # bucket size
if len(sys.argv) >= 6:
    mu = float(sys.argv[5]) # service rate
if len(sys.argv) >= 7:
    distrNameA = sys.argv[6]
if len(sys.argv) >= 8:
    distrNameB = sys.argv[7]
if len(sys.argv) >= 9:
    fileName = sys.argv[8]

serviceDistr = [distrNameB, [1.0/mu]]
#serviceDistr = ["wei", [1.0/mu]]
 # 1000 per second
gRate = pRate*1.0 / numJob

gen = []
queue = []
bucket = []
server = []
for i in xrange(numJob) :
    gen.append(GenJob(i))
    queue.append(Queue(i))
    gen[-1].setOutput(queue[-1])
    gen[-1].setIntDistr(distrNameA, [1.0/gRate])
    #gen[-1].setSizeDistr("binorm", [1000.0])
    gen[-1].setSizeDistr("cst", [1])
    bucket.append(TokenBucket(i))
    queue[-1].setOutput(bucket[-1])
    #bucket[-1].setParameters(pRate/numJob, bucketSize/numJob)
    bucket[-1].setParameters(bRate/numJob, bucketSize/numJob)
    server.append(Server(i))
    bucket[-1].setOutput(server[-1])
    server[-1].setParameters(serviceDistr[0], serviceDistr[1])

time = 0
while time < testDuration:
    nextTimeList = []
    itemList = []
    for b in server:
        nextTime, item  = b.getNextTime()
        nextTimeList.append(nextTime)
        itemList.append(item)

    index = [i for i in xrange(len(nextTimeList)) if nextTimeList[i] == min(nextTimeList)]
    time = nextTimeList[index[0]]
    logging.debug("Simulation time %f", time)
    for i in index :
        itemList[i].whoAmI()
        itemList[i].runTime(time)

f = open(fileName, "w")
for q in queue:
    #q.showStatistic(testDuration/2)
    deq, enq = q.showQueueingTime(int(testDuration)*int(pRate)/2)
    np.savetxt(f, (enq, deq))
for s in server:
    #s.showStatistic(testDuration/2)
    deq, enq = s.showQueueingTime(int(testDuration)*int(pRate)/2)
    np.savetxt(f, (enq, deq))
f.close()




# print "Test for ", numJob
# print numjob, ", queue ", np.mean(q_result), q_result
# print numjob, "server ", np.mean(s_result), s_result




#for b in bucket:
#    b.showStatistic(testDuration/1.2)
