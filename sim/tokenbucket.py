#!/usr/local/bin/python

from util import getRandomDist
import numpy as np
import logging
from math import floor, ceil

class TokenBucket(object):
    def __init__ (self, id = 0) :
        self.id = id
        self.input = []
        self.output = None

        self.rate = 1
        self.nextTime = float('inf')
        self.nextSize = 0.0
        self.time = 0.0
        self.tokenNum = 0
        self.bucketSize = 10
        self.nextIndex = 0
        self.queue = 0
        self.lastUpdateTime = 0.0
        self.usedToken = 0

        #self.enqueueTime = []
        #self.dequeueTime = []
        #self.dropTime = []
        self.finishTime = []

    def whoAmI(self):
         logging.debug("[TokenBucket %d]", self.id)

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
            self.tokenNum = self.nextSize
            while self.pullJob(time):
                continue
                    #print "[Bucket] pull job"
        #else:
        #    logging.info("[Bucket %d] runtime error, should not be here %f %f", self.id, self.time, self.nextTime)

    def setParameters(self, rate = -1, size = -1, tokenNum = -1):
        if rate != -1:
            self.rate = rate
        if size != -1:
            self.bucketSize = size
        if tokenNum != -1:
            self.tokenNum = tokenNum
        else:
            self.tokenNum = 0 # initial token number to half of token bucket size


    def updateParameters(self, time, rate = -1, size = -1, tokenNum = -1):
        self.updateTime(time)
        self.setParameters(rate, size, tokenNum)
        logging.info("[Bucket %d] update token bucket, %f, %d, %d", self.id, rate, size, tokenNum)


    def updateTime(self, time):
        self.time = time

    def hasJob(self):
        return self.queue > 0

    def hasQueue(self, time, size):
        if time != self.nextTime:
            # self.tokenNum = time * self.rate - self.usedToken
            # if self.tokenNum >= self.bucketSize + 1:
            #     self.usedToken += floor(self.tokenNum) - self.bucketSize
            #     self.tokenNum -= floor(self.tokenNum) - self.bucketSize
            self.tokenNum += (time - self.lastUpdateTime) * self.rate
            if self.tokenNum > self.bucketSize:
                self.tokenNum = self.bucketSize
        else:
            self.tokenNum = size
        self.lastUpdateTime = time

        if self.tokenNum >= size:
            self.nextTime = float('inf')
            logging.debug("tokenNum %f, lastupdateTime %f, nextTime %f, time%f", self.tokenNum, self.lastUpdateTime, self.nextTime, time)
            return True
        else:
            self.getNextUpdateTime(time, size)
            logging.debug("tokenNum %f, lastupdateTime %f, nextTime %f, time%f", self.tokenNum, self.lastUpdateTime, self.nextTime, time)
            return False

    def getNextUpdateTime(self, time, size):
        self.nextTime = time + (size - self.tokenNum)/self.rate
        #self.nextSize = size
        #return self.empty == True

    def decreaseQueue(self):
        self.tokenNum -= self.queue
        self.usedToken += self.queue
        self.queue = 0

    def increaseQueue(self, size):
        self.queue = size

    # return True when success dequeue a job
    def dequeueJob(self, time):
        # no output, then dequeue and mark as finished
        if self.output == None:
            self.finishTime.append(time)
        # output enqueueJob fail
        elif not self.output.enqueueJob(time, self.queue):
            return False
        #self.dequeueTime.append(time)
        self.decreaseQueue()
        logging.debug("[Bucket %d] dequeue job, %f", self.id, self.tokenNum)
        return True

    # pull job from input
    # return True when pull a job
    def pullJob(self, time):
        # has a job in this element, then dequeue
        logging.debug("[Bucket %d] pull job", self.id)
        if self.input != []:
            # get from one of them, with round-robin
            for i in xrange(len(self.input)):
                index = (self.nextIndex + i) % len(self.input)
                if not self.input[index].pullJob(time):
                    continue
                # get job from pull job, dequeue and increase nextIndex
                self.nextIndex = (i+1)% len(self.input)
                return self.dequeueJob(time)
        return False

    # return True if enqueue success, otherwise False
    # called by dequeueJob() of self.input
    def enqueueJob(self, time, size):
        if not self.hasQueue(time, size) :
            logging.debug("[Bucket %d] enqueue job fail size %f, nextTime %f, tokenNum %f, lastupdateTime %f", self.id, size, self.nextTime, self.tokenNum, self.lastUpdateTime)
            return False
        logging.debug("[Bucket %d] enqueue job, with token %f", self.id, self.tokenNum)
        #self.enqueueTime.append(time)
        self.increaseQueue(size)
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

    def getTokenNum(self):
        return self.tokenNum

    # def showStatistic(self, startTime):
    #     print "Token Bucket", self.id
    #     i = 0
    #     for i in xrange(len(self.dequeueTime)):
    #         if startTime <= self.dequeueTime[i]:
    #             break
    #     j = len(self.dequeueTime)
    #     queueingTime = np.array(self.dequeueTime[i:j]) - np.array(self.enqueueTime[i:j])
    #     #print "Queueing Time", queueingTime
    #     print "with mean", np.mean(queueingTime), "and variance", np.var(queueingTime)
    #     print "drop number", len(self.dropTime)
    #     print "Finish number", len(self.finishTime)
    #     print "Still in queue", len(self.enqueueTime) - len(self.dequeueTime)
    #     print "\n\n"


    # def run(self, stopTime):
    #     while self.time < stopTime:
    #         self.runNextTime()
    #     print "stop running", self.time
