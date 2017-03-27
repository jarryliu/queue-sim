
import logging

class Scheduler(object):
    def __init__ (self, id = 0):
        self.id = id
        self.name = "Scheduler"
        self.input = []
        self.output = None
        self.time = 0.0
        self.nextTime = float("inf")
        self.index = -1
        self.hasJob = 0.0
        self.nextIndex = 0

        self.queueLen = 0.0
        self.queue = []

        self.finishTime = []
        self.enqueueTime = []
        self.dequeueTime = []

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

    def getNextTime(self):
        nextTime = []
        nextItem = []
        for i in self.input:
            time, item = i.getNextTime()
            #print "[scheduler] get next time", i.id, time, item
            nextTime.append(time)
            nextItem.append(item)

        index = nextTime.index(min(nextTime))
        self.nextTime = nextTime[index]
        self.index = index
        return nextTime[index], nextItem[index]

    def hasJob(self):
        return self.queueLen > 0

    def hasQueue(self):
        return self.queueLen == 0 and (self.output == None or self.output.hasQueue(size))

    def decreaseQueue(self):
        self.queueLen -= self.queue[0]
        self.queue.pop(0)

    def increaseQueue(self, size):
        self.queueLen += size
        self.queue.append(size)

    # return True when success dequeue a job
    def dequeueJob(self, time):
        # no output, then dequeue and mark as finished
        if self.queue == []:
            return False
        size = self.queue[0]
        if self.output == None :
            self.finishTime.append(time)
        # output enqueueJob fail
        elif not self.output.enqueueJob(time, size):
            return False
        self.dequeueTime.append(time)
        self.decreaseQueue()
        logging.debug("[Scheduler %d] dequeue job %f", self.id, size)
        return True

    # pull job from input
    # return True when pull a job
    def pullJob(self, time):
        # has a job in this element, then dequeue
        logging.debug("[Scheduler %d] pull job", self.id)
        if self.hasJob():
            return self.dequeueJob(time, size)
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
        if not self.hasQueue(size) :
            return False
        self.increaseQueue(size)
        self.enqueueTime.append(time)
        logging.debug("[Scheduler %d] enqueue job", self.id)
        return True

    # called by input, push job to output if possible
    def pushJob(self, time):
        if not self.dequeueJob(time):
            return False
        return self.output.pushJob(time)


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

    def showStatistic(self):
        print "Scheduler", self.id
        #print "EnqueueTime", self.enqueueTime
        #print "DequeueTime", self.dequeueTime
        print "In Queue Time", self.dequeueTime - self.enqueueTime[:len(self.dequeueTime)]
        print "drop number", len(self.dropTime)
        print "Finish number". len(self.finishTime)
