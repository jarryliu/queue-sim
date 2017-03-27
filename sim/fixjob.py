#!/usr/local/bin/python

import numpy as np
from util import getRandomDist, setDistr
import logging

class FixJob (object):
    def __init__ (self, id=0):
        self.id = id
        self.time = 0.0
        self.nextTime = -1.0
        self.intDistr = ["bst", []]
        #self.interval = []
        self.sizeDistr = ["cst", [1]]
        #self.size = []
        self.output = None
        self.input = []
        self.queue = []
        self.nextEventTime = -1.0

        self.dropTime = []
        self.waitQueueSize = 0
        # self.enqueueTime = []
        # self.dequeueTime = []
        self.fullPolicy = "drop"
        self.finishTime = []

        self.nextMarkovTime = 0.0
        self.changeFlag = True
        self.nextIndex = 1

        self.intList = []
        self.intNum = 0

    def whoAmI(self):
         logging.debug("[FixJob %d]", self.id)

    def setIntDistr(self, name, arg):
        self.intDistr = [name, arg]

    def setIntList(self, lst):
        self.intList = lst
        self.intNum = 0
        self.updateNextTime()

    def setSizeDistr(self, name, parameters):
        self.sizeDistr = setDistr(name, parameters)

    def getInterval(self):
        if self.intNum >= len(self.intList):
            return float('inf')
        result = self.intList[self.intNum]
        self.intNum += 1
        return result

    def getJobSize(self):
        size = getRandomDist(self.sizeDistr[0], self.sizeDistr[1])
        return size

    def getNextTime(self):
        #print "get next time from genjob", self.id
        return self.nextTime, self

    def updateNextTime(self):
        self.nextTime = self.time + self.getInterval()

    def updateTime(self, time):
        self.time = time

    def runTime(self, time):
        self.updateTime(time)

        # this item triggers the change
        if self.time == self.nextTime:
            # update nextTime, push job to next
            enNum = 1
            if self.intDistr[0].startswith("bst"):
                enNum = self.intDistr[1][-1]
            for i in xrange(enNum):
                self.enqueueJob(time, self.getJobSize())
            self.updateNextTime()
            self.pushAllJob(time)
        else:
            logging.info("[runTime] error, should not be here in GenJob")

    # pull job from input
    # return True when pull a job
    def pullJob(self, time):
        # has a job in this element, then dequeue
        logging.debug("[GenJob %d] pull job", self.id)
        if self.hasJob():
            return self.dequeueJob(time)
        elif self.input != []:
            # get from one of them, with round-robin
            for i in len(self.input):
                index = (self.nextIndex + i)% len(self.input)
                if not self.input[index].pullJob(time):
                    continue
                # get job from pull job, dequeue and increase nextIndex
                self.nextIndex = (i+1)% len(self.input)
                return self.dequeueJob(time)
        return False

    def increaseQueue(self, size):
        self.waitQueueSize += size
        self.queue.append(size)

    def decreaseQueue(self):
        self.waitQueueSize -= self.queue[0]
        self.queue.pop(0)

    def hasJob(self):
        return self.waitQueueSize > 0

    # return True if enqueue success, otherwise False
    # called by dequeueJob() of self.input
    def enqueueJob(self, time, size):
        self.increaseQueue(size)
        #self.size.append(size)
        #self.enqueueTime.append(time)
        logging.debug("[GenJob %d] generate job %f", self.id, size)
        return True

    # return True when success dequeue a job
    def dequeueJob(self, time):
        # no output, then dequeue and mark as finished
        if self.queue == []:
            logging.debug("[GenJob %d] dequeue job fail: nothing in the queue", self.id)
            return False
        size = self.queue[0]
        if self.output == None :
            self.finishTime.append(time)
        # output enqueueJob fail
        elif not self.output.enqueueJob(time, size):
            logging.debug("[GenJob %d] dequeue job fail: output enqueue fail", self.id)
            return False
        #self.dequeueTime.append(time)
        self.decreaseQueue()
        logging.debug("[GenJob %d] dequeue job %f", self.id, size)
        return True

    # called by input, push job to output if possible
    def pushJob(self, time):
        if not self.dequeueJob(time):
            return False
        return self.output.pushJob(time)

    # called by input, push job to output if possible
    def pushAllJob(self, time):
        while self.dequeueJob(time):
            self.output.pushJob(time)
        return False

    def setOutput(self, output):
        self.output = output
        self.output.setInput(self)

    def showStatistic(self):
        print "Fix Job Generator", self.id
        print "Interval Distribution", self.intDistr
        #print "Mean", np.mean(self.interval), "Variance", np.var(self.interval)
        print "Size Distribution", self.sizeDistr
        #print self.size
        #print "Mean", np.mean(self.size), "Variance", np.var(self.size)
