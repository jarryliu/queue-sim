
import numpy as np
import logging

DEFAULT_QUEUE_SIZE=100
class Queue(object):
    def __init__(self, id = 0, queueSize = float("inf")):
        self.id = id
        self.queueSize = queueSize
        self.queueLen = 0
        self.input = []
        self.output = None

        self.enqueueTime = []
        self.dequeueTime = []
        self.dropTime = []
        self.finishTime = []
        self.queue = []

        self.nextIndex = 0
        self.estimatedRate = -1
        self.lastJobNum = 0
        self.lastUpdateTime = 0

    def whoAmI(self):
         logging.debug("[Queue %d]", self.id)

    # input should be an array
    def setInput(self, input):
        if isinstance(input, list):
            self.input += input
        else:
            self.input.append(input)

    def resetInput(self):
        self.input = []

    def setOutput(self, output):
        self.output = output
        self.output.setInput(self)

    def setQueueSize(self, queueSize):
        self.queueSize = queueSize

    def getNextTime(self):
        nextTime = []
        nextItem = []
        for i in self.input:
            time, item = i.getNextTime()
            nextTime.append(time)
            nextItem.append(item)

        index = nextTime.index(min(nextTime))
        self.nextTime = nextTime[index]
        self.index = index
        return nextTime[index], nextItem[index]

    def hasJob(self):
        return self.queueLen > 0

    def hasQueue(self, size):
        return self.queueSize - self.queueLen >= size

    def decreaseQueue(self):
        self.queueLen -= self.queue[0]
        self.queue.pop(0)

    def increaseQueue(self, size):
        self.queueLen += size
        self.queue.append(size)

    # return True when success dequeue a job
    def dequeueJob(self, time):
        # no output, then dequeue and mark as finished
        size = self.queue[0]
        if self.output == None :
            self.finishTime.append(time)
        # output enqueueJob fail
        elif not self.output.enqueueJob(time, size):
            return False
        self.dequeueTime.append(time)
        self.decreaseQueue()
        logging.debug("[Queue %d] dequeue job %f, queue len %d", self.id, size, self.queueLen)
        return True

    # pull job from input
    # return True when pull a job
    def pullJob(self, time):
        # has a job in this element, then dequeue
        logging.debug("[Queue %d] pull job", self.id)
        if self.hasJob():
            return self.dequeueJob(time)
        elif self.input != []:
            # get from one of them, with round-robin
            for i in xrange(len(self.input)):
                index = (self.nextIndex + i)% len(self.input)
                if not self.input[index].pullJob(time):
                    continue
                # get job from pull job, dequeue and increase nextIndex
                self.nextIndex = (i+1)% len(self.input)
                return self.dequeueJob(time)
        return False

    # return True if enqueue success, otherwise False
    # called by dequeueJob() of self.input
    def enqueueJob(self, time, size):
        # if self.lastEnqueueTime == -1:
        #     self.estimatedRate = 1.0/(time - self.lastEnqueueTime)
        # else:
        #     self.estimatedRate = 0.95*self.estimatedRate + 0.1/(time - self.lastEnqueueTime)
        if not self.hasQueue(size):
            return False
        self.lastEnqueueTime = time
        self.increaseQueue(size)
        self.enqueueTime.append(time)
        logging.debug("[Queue %d] enqueue job %f, queue length %d", self.id, size, self.queueLen)
        return True

    # called by input, push job to output if possible
    def pushJob(self, time):
        if not self.dequeueJob(time):
            return False
        return self.output.pushJob(time)

    def getQueueSize(self, time):
        return self.queueLen

    def getRate(self, time):
        if self.estimatedRate == -1:
            self.estimatedRate = 1.0*len(self.enqueueTime)/time
        else:
            currentRate = 1.0*(len(self.enqueueTime)-self.lastJobNum)/(time-self.lastUpdateTime)
            self.estimatedRate = 0.5*self.estimatedRate + 0.5*currentRate

        self.lastUpdateTime = time
        self.lastJobNum = len(self.enqueueTime)

        return self.estimatedRate

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

    def showStatistic(self, startTime):
        print "Queue", self.id
        i = 0
        for i in xrange(len(self.dequeueTime)):
            if startTime <= self.dequeueTime[i]:
                break
        j = len(self.dequeueTime)
        queueingTime = np.array(self.dequeueTime[i:j]) - np.array(self.enqueueTime[i:j])
        #print "Queueing Time", self.enqueueTime
        #print "Dequeuing Time", self.dequeueTime
        interval = np.array(self.enqueueTime[1:]) - np.array(self.enqueueTime[0:-1])
        print "arrive rate", 1.0/np.mean(interval)
        print "with mean", np.mean(queueingTime), "and variance", np.var(queueingTime)
        print "Drop number", len(self.dropTime)
        print "Finish number", len(self.finishTime)
        print "Still in queue", self.queueLen
        print "\n"
