#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from numpy import mean, var, sqrt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def loadfile(fileName):
    return np.loadtxt(fileName)


def getTotalDelay(a):
    t1 = a[0] - a[1]
    t2 = a[2] - a[3]
    t1m = t1.mean()
    t2m = t2.mean()
    return  t1m + t2m, t1m, t2m

def getSepDelay(a):
    t = a[1] - a[0]
    return t.mean()

def getMV(a):
    t = a[1] - a[0]
    return len(a[0]), t.mean(), t.var()

def readFileMeanDelay(fileName, testType="server_test", num=100000):
    a = loadfile(fileName)
    readLen = min([len(a), num])
    testLen = 4
    if testType == "token_test":
        testLen = 2
    tblist = []
    serlist = []
    for i in range(readLen//testLen):
        tb = getSepDelay(a[i*testLen:i*testLen +2])
        tblist.append(tb)
        if testType != "token_test":
            ser = getSepDelay(a[i*testLen +2: i*testLen +4])
            serlist.append(ser)
    m_serList = 0.0
    if len(serlist) != 0:
        m_serList = mean(serlist)
    return mean(tblist), m_serList

def readFileCI(fileName):
    a = loadfile(fileName)
    # testLen = 4
    # if testType == "token_test":
    #     testLen = 2
    tbn, tbm, tbv = getMV(a[0:2])
    # sern, serm, serv = 0
    # if testType != "token_test":
    #     sern, serm, serv = getMV(a[2:4])
    return tbn, tbm, tbv

def calCI(n, m, v, conf = 0.99):
    tvalue = 0.0
    if conf == 0.99:
        tvalue = 2.576
    d = 1.0*sqrt(v)/sqrt(n)*tvalue
    return m+d, m, m-d

def readDepartureInterval(testType, fileName, num=1000):
    a = loadfile(fileName)
    readLen = min([len(a), num])
    tblist = []
    serlist = []
    testLen = 4
    if testType == "token_test":
        testLen = 2
    intlist = np.array([])
    for i in range(readLen/testLen):
        intlist = np.append(intlist, a[i*testLen+1:i*testLen +2])
    intl = intlist[1:] - intlist[:-1]
    n, bins, patches = plt.hist(intl, bins=1000)
    #l = plt.plot(bins, y, 'r--', linewidth=1)
    print(str(intl), str(mean(intl)))
    plt.show()


def getDelayProb(a):
    b = a[1]- a[0]
    return 1.0*sum(b > 0.000000001)/len(b)

def readFileDelayProb(fileName, num):
    a = loadfile(fileName)
    readLen = min([len(a), num])
    pList = []
    for i in range(readLen/4):
        p = getDelayProb(a[i*4:i*4+2])
        pList.append(p)
    return mean(pList)

def plotLatency(test, dirList, distrNameAList, bRateList, bList, pRate=1000.0, distrNameB="", mu=1000.0, tbDelay = [], serDelay = []):
    X = []
    if len(dirList) == 1 and len(bList) > 1:
        X = np.array(bList)
    else:
        X = np.array([int(d[6:]) for d in distrNameAList])
    Y = np.array(bRateList)

    Y, X = np.meshgrid(Y,X)

    Z1 = np.array(tbDelay)
    print(str(Z1.tolist()))

    if test != "token_test":
        Z2 = np.array(serDelay)
        print(str(Z2.tolist()))

    fig = plt.figure()
    if test != "token_test":
        ax = fig.add_subplot(211, projection='3d')
        ax.plot_wireframe(X, Y, Z1)
        ax = fig.add_subplot(212, projection='3d')
        ax.plot_wireframe(X, Y, Z2)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X,Y,Z1)
    ax.set_xlabel('bucket size')
    ax.set_ylabel('bucket rate (k)')
    ax.set_zlabel('access latency')
    ax.view_init(30, 30)
    fig.savefig(test+'_'+distrNameA+'.png')
    plt.close(fig)

def readLatency(test, dirList, distrNameAList, bRateList, bList, pRate=1000.0, distrNameB="", mu=1000.0):
    tbDelay = []
    serDelay = []
    print(distrNameAList, distrNameB, mu)

    for distrNameA in distrNameAList:
        for b in bList:
            tbDelay.append([])
            serDelay.append([])
            #for b in bucketSizeList:
            for bRate in bRateList:
                tblist = []
                serlist = []
                print("read "+ str(distrNameA)+ ", r: "+str(int(bRate))+ ", b: "  +str(b))
                for d in dirList:
                    if test == "token_test":
                        fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + ".out"
                    else:
                        fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + "_" + distrNameB + "_" + str(int(mu)) + ".out"
                    tb, ser = readFileMeanDelay(fileName, test)
                    tblist.append(tb)
                    serlist.append(ser)
                tbDelay[-1].append(mean(tblist))
                serDelay[-1].append(mean(serlist))

    X = []
    if len(dirList) == 1 and len(bList) > 1:
        X = np.array(bList)
    else:
        X = np.array([int(d[6:]) for d in distrNameAList])
    Y = np.array(bRateList)/1000.0

    Y, X = np.meshgrid(Y,X)

    Z1 = np.array(tbDelay)
    print(str(Z1.tolist()))

    if test != "token_test":
        Z2 = np.array(serDelay)
        print(str(Z2.tolist()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z1)
    ax.set_xlabel('bucket size')
    ax.set_ylabel('bucket rate (k)')
    ax.set_zlabel('access latency')
    ax.view_init(30, 30)
    fig.savefig(test+'_'+distrNameA+'_tb.png')
    plt.clf()
    plt.close(fig)

    if test != "token_test":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X,Y,Z2)
        ax.set_xlabel('bucket size')
        ax.set_ylabel('bucket rate (k)')
        ax.set_zlabel('processing latency')
        ax.view_init(30, 30)
        fig.savefig(test+'_'+distrNameA+'_ser.png')
        plt.close(fig)

def readTokenLatency(test, dirList, distrNameAList, bRateList, bList, pRate=1000.0, distrNameB="", mu=1000.0):
    tbDelay = []
    serDelay = []
    print(distrNameAList, distrNameB, mu)

    for distrNameA in distrNameAList:
        for b in bList:
            tbDelay.append([])
            serDelay.append([])
            #for b in bucketSizeList:
            for bRate in bRateList:
                tblist = []
                serlist = []
                print("read "+ str(distrNameA)+ ", r: "+str(int(bRate))+ ", b: "  +str(b))
                for d in dirList:
                    if test == "token_test":
                        fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + ".out"
                    else:
                        fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + "_" + distrNameB + "_" + str(int(mu)) + ".out"
                    tb, ser = readFileMeanDelay(fileName, test)
                    tblist.append(tb)
                    serlist.append(ser)
                tbDelay[-1].append(mean(tblist))
                serDelay[-1].append(mean(serlist))

    X = []
    if len(dirList) == 1 and len(bList) > 1:
        X = np.array(bList)
    else:
        X = np.array([int(d[6:]) for d in distrNameAList])
    Y = np.array(bRateList)/1000.0

    Y, X = np.meshgrid(Y,X)

    Z = np.array(tbDelay)
    print(str(Z.tolist()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z)
    ax.set_xlabel('bucket size')
    ax.set_ylabel('bucket rate (k)')
    ax.set_zlabel('access latency')
    ax.view_init(30, 30)
    fig.savefig(test+'_'+distrNameA+'_tb.png')
    plt.close(fig)

def readServerLatency(test, dirList, distrNameAList, bRateList, bList, pRate=1000.0, distrNameB="", mu=1000.0):
    tbDelay = []
    serDelay = []
    print(distrNameAList, distrNameB, mu)

    for distrNameA in distrNameAList:
        for b in bList:
            tbDelay.append([])
            serDelay.append([])
            #for b in bucketSizeList:
            for bRate in bRateList:
                tblist = []
                serlist = []
                print("read "+ str(distrNameA)+ ", r: "+str(int(bRate))+ ", b: "  +str(b))
                for d in dirList:
                    if test == "token_test":
                        fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + ".out"
                    else:
                        fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + "_" + distrNameB + "_" + str(int(mu)) + ".out"
                    tb, ser = readFileMeanDelay(fileName, test)
                    tblist.append(tb)
                    serlist.append(ser)
                tbDelay[-1].append(mean(tblist))
                serDelay[-1].append(mean(serlist))

    X = []
    if len(dirList) == 1 and len(bList) > 1:
        X = np.array(bList)
    else:
        X = np.array([int(d[6:]) for d in distrNameAList])
    Y = np.array(bRateList)/1000.0

    Y, X = np.meshgrid(Y,X)

    Z = np.array(serDelay)
    print(str(Z.tolist()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z)
    ax.set_xlabel('bucket size')
    ax.set_ylabel('bucket rate (k)')
    ax.set_zlabel('processing latency')
    ax.view_init(30, 30)
    fig.savefig(test+'_'+distrNameA+'_ser.png')
    plt.close(fig)


def readDelayProb(dirList, distrNameAList, bRateList, bList, pRate=1000.0, distrNameB="exp", mu=1000.0):
    delayPList = []
    for distrNameA in distrNameAList:
        for b in bList:
            delayPList.append([])
            for bRate in bRateList:
                pList = []
                print("read "+ str(pRate)+ " "+ str(distrNameA))
                for d in dirList:
                    fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + "_" + distrNameB + "_" + str(int(mu)) + ".out"
                    p = readFileDelayProb(fileName, 40)
                    pList.append(p)
                delayPList[-1].append(mean(pList))
            print(str(delayPList[-1]))

    print(str(delayPList))


def readCI(dirList, distrNameAList, bRateList, bList, pRate=1000.0, distrNameB="exp", mu=1000.0):
    tbDelay = []
    serDelay = []
    d = dirList[0]
    v1List = []
    v2List = []
    v3List = []
    for distrNameA in distrNameAList:
        for b in bList:
            v1List.append([])
            v2List.append([])
            v3List.append([])
            #for b in bucketSizeList:
            for bRate in bRateList:
                tblist = []
                serlist = []
                print("read "+ str(pRate) + " " + str(distrNameA)+ " " +str(b))
                fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + ".out"
                n, m, v = readFileCI(fileName)
                v1, v2, v3 = calCI(n,m,v)
                v1List[-1].append(v1)
                v2List[-1].append(v2)
                v3List[-1].append(v3)

    X = []
    if len(dirList) == 1 and len(bList) > 1:
        X = np.array(bList)
    else:
        X = np.array([int(d[6:]) for d in distrNameAList])
    Y = np.array(bRateList)

    Y, X = np.meshgrid(Y,X)

    Z1 = np.array(v1List)
    Z2 = np.array(v2List)
    Z3 = np.array(v3List)
    print(str((Z3-Z2).tolist()))
    print(str(Z2.tolist()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z1)
    ax.plot_wireframe(X,Y,Z2)
    ax.plot_wireframe(X,Y,Z3)
    plt.show()


def readDelayProb(dirList, distrNameAList, bRateList, bList, pRate=1000.0, distrNameB="exp", mu=1000.0):
    delayPList = []
    for distrNameA in distrNameAList:
        for b in bList:
            delayPList.append([])
            for bRate in bRateList:
                pList = []
                print("read "+ str(bRate)+ " "+ str(distrNameA))
                for d in dirList:
                    fileName = d+distrNameA + "_" + str(int(pRate)) + "_" + str(int(bRate)) + "_" + str(int(b)) + "_" + distrNameB + "_" + str(int(mu)) + ".out"
                    p = readFileDelayProb(fileName, 40)
                    pList.append(p)
                delayPList[-1].append(mean(pList))
            print(str(delayPList[-1]))

    print(str(delayPList))

if __name__ == "__main__":
    dirName = "../run/exp/"
    pRate = 1000.0
    distrNameA = "exp"
    distrNameB = "exp"

    rList = np.array([1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5])*pRate
    bList = [1, 2, 3, 4, 5, 6, 7, 8]
    test = "token_test"
    readLatency(test, [dirName], [distrNameA], rList, bList)
