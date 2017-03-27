#!/usr/local/bin/python3
# support only python3
import numpy as np
import matplotlib.pyplot as plt
import re
from math import sqrt, floor, ceil
from process_theory import *
import scipy as sp
import scipy.stats
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc

# get the all the latencies
def latencyFilter(data):
    return data[:, 3:-1]

# get the histogram of distribution
def histogram(data, title, bins=1000, maxtype = "long"):
    #print(data)
    if maxtype == "short":
        plt.hist(data, bins=bins, range = (min(data), max(data)))  # arguments are passed to np.histogram
    else:
        plt.hist(data, bins=bins, range = (min(data), max(data)))
    # plt.hist(data, bins=bins)
    plt.title(title)
    plt.show()


def mean_confidence_interval(a, k=1, confidence=0.99):
    n = len(a)/k
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2, n-1)
    return m, m-h, m+h

# get the data of single log file and process the latencies
def processFile(path, f, newFile = False):
    data = np.loadtxt(path+f)
    (x,y)=  data.shape
    # 1. get arrival rate
    # arrival = data[1:, 1]/1000/1000
    arrival = data[floor(x/5):floor(4*x/5), 1]/1000/1000
    #histogram(arrival, "arrival interval distribution")
    mean_a = np.mean(arrival)
    var_a = np.var(arrival)
    # print("Mean Arrival interval is", mean_a, "variance is", var_a)
    # 2. get end-to-end latency distribution
    # latency = data[1:, 0]/1000/1000
    latency = data[floor(x/5):, 11]/1000/1000
    # print(f,latency)
    #histogram(latency, "end-to-end latency distribution")
    m, m_l, m_h = mean_confidence_interval(latency)
    mList = np.mean(data[floor(x/5):floor(4*x/5), 3:8]/1000/1000, 0)
    if newFile:
        temp = np.mean(data[floor(x/5):floor(4*x/5), 3:11]/1000/1000, 0)
        mList[0] = temp[0]+temp[1]
        mList[1] = temp[2] + temp[3]
        mList[2] = temp[4] + temp[5]
        mList[3:] = temp[6:]
    # print(f, m, m_l, m_h)
    mean_s = [m, m_l, m_h, np.percentile(latency,5), np.percentile(latency, 99)]+list(mList)
    var_s = np.var(latency)
    # print("Average Latency is", mean_s, "variance is", var_s, "98 percentile", np.percentile(latency, 95))
    return mean_a, var_a, mean_s, var_s

def processMFile(path, f):
    data = np.loadtxt(path+f)

    (x,y)=  data.shape
    # 1. get arrival rate
    arrival = data[floor(x/5):floor(4*x/5), 1]/1000/1000
    #histogram(arrival, "arrival interval distribution")
    mean_a = np.mean(arrival)
    var_a = np.var(arrival)
    print("Mean Arrival interval is", mean_a, "variance is", var_a)
    # 2. get end-to-end latency distribution
    latency = data[floor(x/5):floor(4*x/5), 4:6]/1000/1000
    latency = np.sum(latency, 1)
    #histogram(latency, "end-to-end latency distribution")
    mean_s = np.mean(latency)
    var_s = np.var(latency)
    print("Average Latency is", mean_s, "variance is", var_s)
    return mean_a, var_a, mean_s, var_s


def printLatency(lam, var_a, mu, var_s):
    lower, upper = getLatency(lam, var_a, mu, var_s)
    print("Theoretical latency bounds are", lower, upper)

# get the first integer in a string
def getNum(string):
    r = re.compile("([0-9]*)([a-zA-Z]*)([0-9]*)")
    for s in r.match(string).groups():
        if s.isdigit():
            return int(s)
    return 0

def readFileList(path, fList, newFile = False):
    maList = []
    varaList = []
    msList = []
    varsList = []
    for f in fList:
        mean_a, var_a, mean_s, var_s = processFile(path, f, newFile=newFile)
        maList.append(mean_a)
        varaList.append(var_a)
        msList.append(mean_s)
        varsList.append(var_s)
    return np.array(maList), np.array(varaList), np.array(msList), np.array(varsList)


def burstLatency(bList, m):
    mean = []
    for b in bList:
        mean.append(getLatencyDD1(b, 1/m))
    return mean

def getBounds(mList, vList, mean, var):
    upper = []
    lower = []
    for i in range(len(mList)):
        l, u  = getLatencyGG1(1/mList[i], vList[i], 1/mean, var)
        lower.append(l)
        upper.append(u)
    return lower, upper

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)

def plotStage(mean_s, newFile = False):
    #plt.plot(bList, mean_s[:,4], "*-", label= "99 percentile")
    plt.plot(bList, mean_s[:, 5], "*-", label="stage 1")
    plt.plot(bList, mean_s[:, 6], "*-", label="stage 2")
    plt.plot(bList, mean_s[:, 7], "*-", label="stage 3")
    plt.plot(bList, mean_s[:, 8], "*-", label="stage 4")
    #plt.plot(bList, mean_s[:, 9], "*-", label="stage 5")
    print("latency ", mean_s[:,0])
    print("stage 1 ", mean_s[:,5])
    print("stage 2 ", mean_s[:,6])
    print("stage 3 ", mean_s[:,7])
    print("stage 4 ", mean_s[:,8])
    #print("stage 5 ", mean_s[:,9])

# type = {"arrival", "latency", "depart", "stage1" ... "stage5"}
def showIndividual(path, f):
    data = np.loadtxt(path+f)
    (x,y)=  data.shape
    # 1. get arrival rate
    # arrival = data[1:, 1]/1000/1000
    # arrdiff = np.
    totalLatency = np.array(data[100:floor(x/5), 0])
    totalLatency = totalLatency[1:]
    intDiff = np.array(data[100:floor(x/5), 1])
    intDiff = intDiff[1:].reshape(len(intDiff[1:]),1)
    latency = np.array(data[100:floor(x/5), 3:8])
    interval = latency[1:, :] - latency[:-1, :]
    s = intDiff.T - interval[:, 0]
    interval[:, 0] = intDiff.T
    interval[:, 1:] = np.cumsum(interval[:, 1:], axis =1) + intDiff

    # np.set_printoptions(threshold=np.inf)
    result = np.concatenate((s.T, interval, latency[1:, :], totalLatency.reshape(len(totalLatency), 1)), axis = 1)

    import pandas as pd
    df = pd.DataFrame(result[:201, :])
    with pd.option_context('display.max_rows', 200, 'display.max_columns', 12):
        print(df)


#path = "/Users/junjieliu/Development/GitHub/RTM-test/2018-2-5/"
# path = "/Users/junjieliu/Development/GitHub/RTM-test/theory_validation/"
# #
# # showIndividual(path+"poisson_1core_batch/", "b200")
# showIndividual(path+"batch_1pub_1core/", "batch100")
# exit(0)


# draw the distribution,
# type = {"arrival", "latency", "depart", "stage1" ... "stage5"}
def drawDistribution(path, f, types=["arrival"]):
    data = np.loadtxt(path+f)
    (x,y)=  data.shape
    # 1. get arrival rate
    # arrival = data[1:, 1]/1000/1000
    # arrdiff = np.
    intDiff = np.array(data[floor(x/5):floor(4*x/5), 1]/1000/1000)
    intDiff[0] = 0
    intDiff = np.cumsum(intDiff)

    for t in types:
        if t == "arrival":
            arrival = data[floor(x/5):floor(4*x/5), 9]
            arrival = np.array(arrival)
            # arrival = (arrival[1:] - arrival[:-1])/1000/1000
            histogram(arrival, "arrival")
        elif t == "interval0":
            latency = np.array(data[floor(x/5):floor(4*x/5), 3]/1000/1000)
            interval = intDiff - latency
            #print(intDiff[:10], interval[:10])
            interval = interval[1:] - interval[:-1]
            # print(interval[:100])
            histogram(interval, 'interval0', maxtype ="short")
        elif t == "interval1":
            interval = data[floor(x/5):floor(4*x/5), 1]/1000/1000
            histogram(interval, "interval1", maxtype ="short")
        elif t == "interval2":
            latency = np.array(data[floor(x/5):floor(4*x/5), 4]/1000/1000)
            interval = intDiff + latency
            #print(intDiff[:10], interval[:10])
            interval = interval[1:] - interval[:-1]
            histogram(interval, 'interval2', maxtype ="short")
        elif t == "interval3":
            latency = np.array(data[floor(x/5):floor(4*x/5), 4]/1000/1000)
            latency += np.array(data[floor(x/5):floor(4*x/5), 5]/1000/1000)
            interval = intDiff + latency
            #print(intDiff[:10], interval[:10])
            interval = interval[1:] - interval[:-1]
            histogram(interval, 'interval3', maxtype ="short")
        elif t == "interval4":
            latency = np.array(data[floor(x/5):floor(4*x/5), 4]/1000/1000)
            latency += np.array(data[floor(x/5):floor(4*x/5), 5]/1000/1000)
            latency += np.array(data[floor(x/5):floor(4*x/5), 6]/1000/1000)
            interval = intDiff + latency
            #print(intDiff[:10], interval[:10])
            interval = interval[1:] - interval[:-1]
            histogram(interval, 'interval4', maxtype ="short")
        elif t == "interval5":
            latency = np.array(data[floor(x/5):floor(4*x/5), 4]/1000/1000)
            latency += np.array(data[floor(x/5):floor(4*x/5), 5]/1000/1000)
            latency += np.array(data[floor(x/5):floor(4*x/5), 6]/1000/1000)
            latency += np.array(data[floor(x/5):floor(4*x/5), 7]/1000/1000)
            interval = intDiff + latency
            #print(intDiff[:10], interval[:10])
            interval = interval[1:] - interval[:-1]
            histogram(interval, 'interval5', maxtype ="short")
        elif t == "latency":
            latency = data[floor(x/5):floor(4*x/5), 0]/1000/1000
            histogram(latency, "latency")
        elif t == "stage1":
            latency = data[floor(x/5):floor(4*x/5), 3]/1000/1000
            histogram(latency, "stage1 latency")
        elif t == "stage2":
            latency = data[floor(x/5):floor(4*x/5), 4]/1000/1000
            histogram(latency, "stage2 latency")
        elif t == "stage3":
            latency = data[floor(x/5):floor(4*x/5), 5]/1000/1000
            histogram(latency, "stage3 latency")
        elif t == "stage4":
            latency = data[floor(x/5):floor(4*x/5), 6]/1000/1000
            histogram(latency, "stage4 latency")
        elif t == "stage5":
            latency = data[floor(x/5):floor(4*x/5), 7]/1000/1000
            histogram(latency, "stage5 latency")



def showLatency(path, bList, fList, directory, showStage = True, changeRate = False, newFile = False):
    mean_a, var_a, mean_s, var_s = readFileList(path + directory+"/", fList, newFile = newFile)
    # plot the shaded range of the confidence intervals
    #plt.fill_between(bList, mean_s[:, 3], mean_s[:, 4], alpha=.2)
    plt.fill_between(bList, mean_s[:, 1], mean_s[:, 2], alpha=.5)
    # print(mean_s[:, 0])
    #plt.plot(bList, mean_s[:, 0], "*-", label="measured"
    plt.plot(bList, mean_s[:, 0], "*-", label=directory)
    if showStage:
        plotStage(mean_s)
    # if not changeRate:
    #     # s, v, b = fitModelXMXG1B(bList, mean_s[:,0])
    #     # print("parameter fitting", s, v, b)
    #     # xlist = [i for i in range(int(bList[0]), int(bList[-1]))]
    #     # y = [modelXMXG1B(i, s, v, b) for i in xlist]
    #     # s, v = fitModelXMXG1(bList, mean_s[:,0])
    #     # print("parameter fitting", s, v)
    #     # xlist = [i for i in range(int(bList[0]), int(bList[-1]))]
    #     # y = [modelXMXG1(i, s, v) for i in xlist]
    #     # plt.plot(xlist, y, "--", label="MXD1 model")
    #
    #     s, v = fitSetup(bList, mean_s[:,0])
    #     print("parameter fitting", s, v)
    #     xlist = [i for i in range(int(bList[0]), int(bList[-1]))]
    #     y = [modelSetup(i, s, v) for i in xlist]
    #     plt.plot(xlist, y, "--", label="Setup model")
    #
    #     s, v = fitSimple(bList, mean_s[:, 4])
    #     print("simple parameter fitting", s, v)
    #     xlist = [i for i in range(int(bList[0]), int(bList[-1]))]
    #     y = [modelSimple(i, s, v) for i in xlist]
    #     plt.plot(xlist, y, "--", label="Simple model")
    # else:
    #     # s, v, b = fitModelRateMXG1B(bList, mean_s[:,0]/1000)
    #     # print("parameter fitting", s, v, b)
    #     # xlist = [i for i in range(int(bList[0]), int(bList[-1]))]
    #     # y = [modelRateMXG1B(i, s, v, b) for i in xlist]
    #     s, v = fitModelRateMXG1(bList, mean_s[:,0])
    #     print("parameter fitting", s, v)
    #     print(bList, bList[0], bList[-1])
    #     xlist = [i for i in range(1, int(bList[-1]), 5)]
    #     y = [modelRateMXG1(i, s, v) for i in xlist]
    #     print(xlist, y)
    #     plt.plot(xlist, y, "--", label="setup model")

    plt.ylabel("Latency (ms)")
    if changeRate:
        plt.xlabel("Message Rate (kmessage/s)")
    else:
        plt.xlabel("Batch Size")
    plt.legend()
    plt.show()

if __name__ == "__main__" :
    path = "/Users/junjieliu/Development/GitHub/RTM-test/theory_validation/"
    dirList = ["cores/", "burst/", "burst_1core/", "msgsize/"]

    # drawDistribution(path+"burst_1core/", "1000pub", ["interval0", "interval1", "interval2", "interval3", "interval4", "interval5", "stage1", "stage2", "stage3", "stage4", "stage5"])
    # drawDistribution(path+"burst/", "1000pub", ["interval0", "interval1", "interval2", "interval3", "interval4", "interval5", "stage1", "stage2", "stage3", "stage4", "stage5"])
    # drawDistribution(path+"batch_1pub_1core/", "batch10", ["stage1", "stage2", "stage3", "stage4", "stage5"])
    # drawDistribution(path+"batch_1pub_8core/", "batch1", ["latency"])
    # drawDistribution(path+"poisson_1pub_1core_concurrency/", "p100", ["arrival"])
    # drawDistribution(path+"poisson_1pub_1core_rate/", "nemda50", ["arrival"])
    # exit(0)


    # m = mean_s[0]
    # print ("cores")
    # # core
    # cList = [1, 2, 4, 8]
    # fList = [str(c)+"core" for c in cList]
    # mean_a, var_a, mean_s, var_s  = readFileList(path + dirList[0], fList)
    # plt.plot(cList, mean_s, label="measured")
    # plt.plot(cList, [mean_s[0]/c for c in cList], label="estimate")
    # plt.legend()
    # plt.show()

    # bList = [1,2,4,8]
    # fList = [str(b)+"core" for b in bList]
    # directory = "cores"
    # showLatency(path, bList, fList, directory)
    # exit(0)

    # bList = np.array([100+10*i for i in range(3, 9)] + [200])
    # fList = [str(b) for b in bList]
    # bList = 1000/bList
    # directory = "rate"
    # showLatency(path, bList, fList, directory, changeRate=True)
    # exit(0)

    # bList = np.array([10, 100, 200, 500, 1000, 5000, 10000])/1000
    # fList = ["latency28", "latency29", "latency30", "latency35", "latency34", "latency33", "latency36"]
    # directory = "multi-conn"
    # showLatency(path, bList, fList, directory)
    # exit(0)

    bList = np.array([10, 100, 200, 500, 1000, 5000, 10000])/1000
    fList = ["latency15", "latency14", "latency13", "latency12", "latency11", "latency26", "latency25"]
    directory = "multi-conn"
    showLatency(path, bList, fList, directory)
    exit(0)

    # bList = [1, 10, 100, 200, 500, 1000]
    # fList = [str(b)+"pub" for b in bList]
    # directory = "burst"
    # showLatency(path, bList, fList, directory)
    # exit(0)

    # # # drawDistribution(path+"burst_1core/", "20pub", ["interval0", "interval1", "interval2", "interval3", "interval4", "interval5"])#stage2", "stage2", "stage3", "stage4", "stage5"])
    # bList = [1, 20, 100, 200, 500, 1000]
    # fList = [str(b)+"pub" for b in bList]
    # directory = "burst_1core"
    # showLatency(path, bList, fList, directory)
    # exit(0)

    # A, B = fitPub1(np.array(bList), mean_s[:, 0])
    # print("model parameter a+b", A, "c", B)
    # res = [modelPub1(i, A, B) for i in bList]
    # plt.plot(bList, res, "*--", label="model pub_1core")
    # # plotStage(mean_s)
    # print("measured ", mean_s[:, 0])
    #
    # # expLatency = [ mean_s[i, 0]/(ceil(bList[i]/8)+1)*(bList[i]+1) for i in range(len(bList)) ]
    # # print("exp", expLatency)
    # # plt.plot(bList, expLatency, "*-", label="D(k)/D/M")

    # sList = [128, 256, 512, 1024, 2048, 4096, 8192]
    # fList = [str(s) for s in sList]
    # directory = "msgsize"
    # showLatency(path, bList, fList, directory)
    # exit(0)

    # drawDistribution(path+"batch_1pub_8core/", "batch100", ["interval0", "interval1", "interval2", "interval3", "interval4", "interval5", "stage1", "stage2", "stage2", "stage3", "stage4", "stage5"])
    # bList = [1, 10, 100, 200, 500, 1000]
    # fList = ["batch"+str(b) for b in bList]
    # directory = "batch_1pub_8core"
    # showLatency(path, bList, fList, directory)
    # exit(0)

    # # new batch_1pub_1core
    # path = "/Users/junjieliu/Development/GitHub/RTM-test/theory_validation/"
    # bList = [1, 10, 100, 500, 1000]
    # fList = ["batch"+str(b) for b in bList]
    # directory = "batch_1pub_1core_moreTimestamps"
    # showLatency(path, bList, fList, directory, newFile = True)
    # exit(0)

    # drawDistribution(path+"batch_1pub_1core/", "batch500", ["interval0", "interval1", "interval2", "interval3", "interval4", "interval5", "stage1", "stage2", "stage3", "stage4", "stage5"])
    #drawDistribution(path+"batch_1pub_1core/", "batch1000", ["interval0"])#, "interval1", "interval2", "interval3", "interval4", "interval5", "stage1", "stage2", "stage3", "stage4", "stage5"])
    #drawDistribution(path+"burst_1core/", "1000pub", ["interval0"])
    #drawDistribution(path+"poisson_1pub_1core_rate/", "nemda20", ["interval0", "interval1", "interval2", "interval3", "interval4", "interval5", "stage1", "stage2", "stage3", "stage4", "stage5"])
    # bList = [1, 10, 100, 200, 500, 1000]
    # fList = ["batch"+str(b) for b in bList]
    # directory = "batch_1pub_1core"
    # showLatency(path, bList, fList, directory)
    # exit(0)

    #
    # res = [modelBatch1(i, A, B+C) for i in bList]
    # plt.plot(bList, res, "*--", label="model batch_1core")
    # # A, C = fitBatch1(np.array(bList), mean_s[:,  0])
    # # print("model parameter a", A, "c", C)
    # # res = [modelBatch1(i, A, C) for i in bList]
    # # plt.plot(bList, res, "*--", label="model batch_1core")
    # # plotStage(mean_s)
    # print("measured ", mean_s)


    # # Assuming D/D/1, not working...
    # t = burstLatency(bList[1:], mean_s[0, 0])
    # plt.plot(bList[1:], t, label=r"$D^{(k)}/D/1$")
    # print(r"$D^{(k)}/D/1$: ", t)
    #print([bPerVar(lam, k)

    # rate 10 pub, 2 cores
    # bList = np.array([200, 180, 170,160, 150, 140, 130 ,125, 120, 110])
    # fList = [str(b) for b in bList]
    # directory = "rate"
    # showLatency(path, bList, fList, directory):
    # exit(0)

    # rate 1 pub 1 core
    # bList = 1000/np.array([10000, 1000, 500, 100, 50, 20, 10])
    # fList = ["p"+str(int(1000/b)) for b in bList]
    # directory = "rate_1pub_1core"
    # showLatency(path, bList, fList, directory, changeRate = True)
    # exit(0)

    # # Poisson rate 1 pub 1 core
    # drawDistribution(path+"poisson_1pub_1core_rate/", "nemda50", ["interval0", "interval1", "interval2", "interval3", "interval4", "interval5", "stage1", "stage2", "stage3", "stage4", "stage5"])
    # bList = 1000/np.array([1000, 50, 20, 10])
    # fList = ["nemda"+str(int(1000/b)) for b in bList]
    # # print(fList)
    # directory = "poisson_1pub_1core_rate"
    # showLatency(path, bList, fList, directory, changeRate = True)
    # exit(0)

    # poisson
    # drawDistribution(path+"poisson_1core_batch/", "b1000", ["interval0", "interval1", "interval2", "interval3", "interval4", "interval5", "stage1", "stage2", "stage3", "stage4", "stage5"])
    # bList = np.array([1, 10, 100, 200, 500, 1000])
    # fList = ["b"+str(b) for b in bList]
    # directory = "poisson_1core_batch"
    # showLatency(path, bList, fList, directory)
    # exit(0)
    #
    # poisson concurrent
    # bList = np.array([100, 200, 1000])
    # fList = ["p"+str(b) for b in bList]
    # directory = "poisson_1pub_1core_concurrency"
    # showLatency(path, bList, fList, directory)
    # exit(0)




    #
    # # print("burst")
    # # # burst
    # bList = [1, 10, 100, 200, 500, 1000]
    # fList = [str(b)+"pub" for b in bList]
    # mean_a, var_a, mean_s, var_s = readFileList(path + "burst/", fList)
    # # measured result
    # #plt.fill_between(bList, mean_s[:, 3], mean_s[:, 4], alpha=.2)
    # plt.fill_between(bList, mean_s[:, 1], mean_s[:, 2], alpha=.5)
    # plt.plot(bList, mean_s[:, 0], "*-", label="pub_8core")
    # #A, B = fitPub8(bList, mean_s[:,0])
    # #print("model parameter a+b", A, "c", B)
    # res = [modelPub8(i, A, B) for i in bList]
    # plt.plot(bList, res, "*--", label="model pub_8core")
    # #plotStage(mean_s)
    # print("measured 8 cores", mean_s[:, 0])

    # # Assuming D/D/1, not working...
    # t = burstLatency(bList[1:], mean_s[0, 0])
    # plt.plot(bList[1:], t, label="D/D/1")
    # print("D/D/1: ", t)
    # #print([bPerVar(lam, k)

    # # Assuming D/D/1 with modified burst
    # bListP = [getBurstByVar(mean_a[i], var_a[i]) for i in range(len(mean_a))]
    # #print(bListP)
    # t = burstLatency(bListP, mean_s[0])
    # plt.plot(bList, t, label="D/D/1 modified")

    # #Assuming G/G/1
    # #plt.plot(bList, mean_a)
    # lower, upper = getBounds(mean_a, var_a, mean_s[0], var_s[0])
    # plt.plot(bList, lower, label="lower G/G/1")
    # #plt.plot(bList, upper, label="upper G/G/1")
    # print("G/G/1 lower bound: ", lower)
    # #print("G/G/1 upper bound: ", upper)

    # plt.ylabel("Latency (ms)")
    # plt.xlabel("# of publishers")
    # plt.legend()
    # plt.ylim(0, 10)
    # plt.show()
    # exit(0)
