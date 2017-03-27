#!/usr/local/bin/python

from util import getRandomDist
import numpy as np
import logging
from math import floor, ceil

class Server(object):
    def __init__ (self, id = 0) :
        self.id = id
        self.input = []
        self.output = None

        self.serviceDistr  = ["exp", [1.0/1000]]
        self.queue = []
        self.nextTime = float('inf')
        self.lastUpdateTime = 0.0

        self.enqueueTime = []
        self.dequeueTime = []
        self.dropTime = []
        self.finishTime = []

    def whoAmI(self):
         logging.debug("[Server %d]", self.id)

    def setRate(self, rate):
        self.rate = rate

    def setInput(self, input):
        if isinstance(input, list):
            self.input += input
        else:
            self.input.append(input)

    def setOutput(self, output):
        self.output = output
        self.output.setInput(self)

    def getNextTime(self):
        nextTime = []
        nextItem = []
        for i in self.input:
            time, item = i.getNextTime()
            nextTime.append(time)
            nextItem.append(item)

        index = nextTime.index(min(nextTime))
        ntime = nextTime[index]
        nitem = nextItem[index]

        if self.nextTime <= ntime:
            return self.nextTime, self
        else :
            return ntime, nitem

    def runTime(self, time):
        self.updateTime(time)
        # this item triggers the change
        if self.time == self.nextTime:
            # update nextTime, push job to next
            self.pushJob(time)
                    #print "[Bucket] pull job"
        #else:
        #    logging.info("[Bucket %d] runtime error, should not be here %f %f", self.id, self.time, self.nextTime)

    def setParameters(self, name, parameters):
        self.serviceDistr = [name, parameters]


    # def updateParameters(self, name, parameters):
    #     self.updateTime(time)
    #     self.setParameters(name, parameters)
    #     logging.info("[Server %d] update service distribution, %s, %d", name, parameters[0])


    def updateTime(self, time):
        self.time = time

    def hasJob(self):
        return len(self.queue) > 0

    def hasQueue(self, time, size):
        return True

    def updateNextTime(self, time):
        if not self.hasJob():
            self.nextTime = float('inf')
        else:
            self.nextTime = time + getRandomDist(self.serviceDistr[0], self.serviceDistr[1])
        #self.nextSize = size
        #return self.empty == True

    def decreaseQueue(self):
        self.queue.pop(0)

    def increaseQueue(self, size):
        self.queue.append(size)

    # return True when success dequeue a job
    def dequeueJob(self, time):
        # no output, then dequeue and mark as finished
        if self.queue == []:
            return False
        size = self.queue[0]
        if self.output == None:
            self.finishTime.append(time)
        # output enqueueJob fail
        elif not self.output.enqueueJob(time, size):
            return False
        self.dequeueTime.append(time)
        self.decreaseQueue()
        logging.debug("[Server %d] dequeue job, queue length %d", self.id, len(self.queue))
        self.updateNextTime(time)
        return True

    # pull job from input
    # return True when pull a job
    # def pullJob(self, time):
    #     # has a job in this element, then dequeue
    #     logging.debug("[Server %d] pull job", self.id)
    #     if self.input != []:
    #         self.input[0].pullJob(time)
    #         return False #self.dequeueJob(time)
    #         # # get from one of them, with round-robin
    #         # for i in xrange(len(self.input)):
    #         #     index = (self.nextIndex + i) % len(self.input)
    #         #     if not self.input[index].pullJob(time):
    #         #         continue
    #         #     # get job from pull job, dequeue and increase nextIndex
    #         #     self.nextIndex = (i+1)% len(self.input)
    #         #     return self.dequeueJob(time)
    #     return False

    # return True if enqueue success, otherwise False
    # called by dequeueJob() of self.input
    def enqueueJob(self, time, size):
        if not self.hasQueue(time, size) :
            logging.debug("[Server %d] enqueue job fail size %f, nextTime %f, queue length %f, lastupdateTime %f", self.id, size, self.nextTime, len(self.queue), self.lastUpdateTime)
            return False
        logging.debug("[Server %d] enqueue job, with queue length %d", self.id, len(self.queue))
        updateFlag = False
        if len(self.queue) == 0:
            updateFlag = True
        self.increaseQueue(size)
        self.enqueueTime.append(time)
        if updateFlag :
            self.updateNextTime(time)
        return True

    # called by input, push job to output if possible
    def pushJob(self, time):
        if not self.dequeueJob(time):
            return False
        return True
        #return self.pushJob(time, size)

    def getQueueSize(self, time):
        size = 0
        if self.input != []:
            for q in self.input:
                size += q.getQueueSize(time)
        return size

    def getRate(self, time):
        rate = 0
        if self.input != []:
            for q in self.input:
                rate += q.getRate(time)
        return rate

    def showStatistic(self, startTime):
        print "Server", self.id
        i = 0
        for i in xrange(len(self.enqueueTime)):
            if startTime <= self.enqueueTime[i]:
                break
        j = len(self.dequeueTime)
        queueingTime = np.array(self.dequeueTime[i:j]) - np.array(self.enqueueTime[i:j])
        #print self.enqueueTime
        #print self.dequeueTime
        print "start time ", startTime, i, j
        print "Queueing Time", queueingTime
        print "with mean", np.mean(queueingTime), "and variance", np.var(queueingTime)
        print "drop number", len(self.dropTime)
        print "Finish number", len(self.finishTime)
        print "Still in queue", len(self.enqueueTime) - len(self.dequeueTime)
        print "\n\n"

    def showRate(self, startTime):
        i = 0
        for i in xrange(len(self.enqueueTime)):
            if startTime <= self.enqueueTime[i]:
                break
        j = len(self.dequeueTime)
        queueingTime = np.array(self.dequeueTime[i:j]) - np.array(self.enqueueTime[i:j])
        return np.mean(queueingTime)

    def showQueueingTime(self, num):
        queueLen = len(self.dequeueTime)
        start = 0
        if queueLen > num:
            start = queueLen-num
        return self.dequeueTime[start:], self.enqueueTime[start:queueLen]

    def run(self, stopTime):
        while self.time < stopTime:
            self.runNextTime()
        print "stop running", self.time
