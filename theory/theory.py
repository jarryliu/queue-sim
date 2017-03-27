#!/usr/local/bin/python
# import matplotlib
# matplotlib.use('Agg')
from math import factorial, exp, sqrt, floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

from sympy import Symbol, solve


def probP(n, rho):
    if n < 1 :
        return [1]
    else:
        p = probP(n-1, rho)
        return [sum(p)] + [-rho*exp(-rho)*p[i]/(i+1) for i in range(len(p))]

## Pr(v > n)
def probMD(rho, n):
    p = probP(floor(n), rho)
    p = [p[i]*(n-int(n))**i for i in range(len(p))]
    return 1.0 - (1-rho)*exp(rho*n)*sum(p)

def drawPdf(rho):
    n = np.linspace(1,10,1000)
    cdfV = []
    for i in n :
        cdfV.append(probMD(rho,i))
    cdfV = np.array(cdfV)
    y = cdfV[:-1] - cdfV[1:]
    plt.plot(n[1:],y)
    plt.show()

# token bucket access delay for poisson
def tbDelay(rho, r, b, k=1):
    p = probMD(rho, b-1)
    return p/2/r/(1-rho)

def PoiTBDelay(lam, r, b):
    if lam > r:
        return float('inf')
    rho = 1.0*lam/r
    p = probMD(rho, b-1)
    return p/2/r/(1-rho)

# token bucket access delay for batched periodic
def batchPerTBDelay(lam, k, r, b):
    if k > b:
        return 1.0*(k-b)*(k-b+1)/2/r/k
    else:
        return 0

# token bucket access delay for batched poisson
def batchPoiTBDelay(lam, k, r, b):
    rho = 1.0*lam/r
    # p = 0.0
    # if k > b:
    #     p = probMD(rho, b/k)
    # return p*(1.0*k/2/r/(1-rho) + 1.0*(k+1)/2/r)
    delay = 0.0
    w = rho/2/(1-rho)*k/r
    for i in range(k):
        p = probMD(rho, (b-i)*1.0/k)
        delay += w*p
    delay /= k
    return delay


# token bucket access delay for each type
def TokenBucketDelay(type, lam, r, b ,k=1):
    if type =="poisson":
        return PoiTBDelay(lam, r, b)
    elif type == "batchperiodic":
        return batchPerTBDelay(lam, k, r, b)
    elif type =="bpoisson":
        return batchPoiTBDelay(lam, k, r, b)

# server delay for Poisson
def poiSerDelay(lam, mu, r, b):
    if lam >= r:
        return float('inf')
    cov_a = getPoiCoV(lam, r, b)
    rho = 1.0*lam/mu
    #return rho*(cov_a**2 + rho**2)/(1-rho)/(1+rho**2) + 1.0/mu
    return  rho/(1-rho)*cov_a**2/2/mu + 1.0/mu

# get the CoV for Poisson
def getPoiCoV(lam, r, b):
    lam = 1.0*lam
    r = 1.0*r
    rho = lam/r
    if b == 1:
        d2 = 2*(1-rho)/lam**2 + rho/lam**2
        var = d2 - 1/lam**2
        return sqrt(var*r**2)

    var1 = 1.0/r**2*((lam/r)**2 + 2*lam/r + 2)*exp(-lam/r) + 1/r**2 * (1-exp(-lam/r))
    # return sqrt(var*r**2)
    p1 = probMD(rho, b-1)
    p2 = probMD(rho, b)
    #print p1, p2
    var = (p2)*1/r**2 + (1-p1)*2/lam**2 - 1/lam**2 + (p1-p2)*var1
    #print lam, r, b, var
    return sqrt(var*r**2)


# get resource for poisson
def getMuPoi(tar, lam, cov_a):
    mu = Symbol('mu')
    res = solve(1.0*lam/mu/(1-1.0*lam/mu)*cov_a**2/2/mu + 1.0/mu - tar, mu) # G/D/1
    #res = solve(lam/mu/(1-lam/mu)*(cov_a**2 + 1)/2/mu + 1.0/mu - tar, mu)
    if res[-1] > lam:
        return res[-1]
    else:
        return float('inf')

# get delay for GG1
def getLatencyGG1(lam, var_a, mu, var_s):
    return (lam*var_s-mu*(2-lam/mu))/2/(1-lam/mu), lam

def bPerSerDelay(lam, k, mu, r, b):
    mu *= 1.0
    r *= 1.0
    # print lam, k, mu, r, b
    # print floor(b*r/mu), b/mu - floor(b*r/mu)/r
    # print (b-1)*b/2.0/mu/k, (floor(b*r/mu)**2-floor(b*r/mu))/r/2, floor(b*r/mu)*(b/mu - floor(b*r/mu)*1/r), 1/mu

    if (b+1)/mu <= 1/r:
        if k >= b :
            return (b-1)*b/mu/2/k + 1/mu
        else:
            return (k-1)*k/mu/2/k + 1/mu
    else:
        if k >= b:
            fl  = floor(b*r/mu)
            #return ((b-1)*b/mu/2 + ((fl**2-fl)/r/2+ fl*(b/mu-fl/r)))/1.0/k + 1/mu
            res = (b-1)*b/mu/2
            i = b
            w = (b+1)/mu
            while i < k and w-1/r >0 :
                res += w-1/r
                i += 1
                w -= 1/r - 1/mu
            return res/k + 1/mu
        else:
            return (k-1)*k/mu/2/k + 1/mu

lam = 100000.0
k = 10
mu = 150000.0
def testBPer(lam, k, mu):
    delayList = []
    rList = [lam+(mu-lam)*(i+1)/10.0 for i in range(10)]
    bList = [i+1 for i in range(k)]
    for r in rList:
        delayList.append([])
        for b in bList:
            delayList[-1].append(bPerDelay(lam, k, r, b, mu))

    X = np.array(rList)
    Y = np.array(bList)

    Y, X = np.meshgrid(Y,X)

    # print X.tolist(), Y.tolist()
    # print tbList #, serList
    print(delayList)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, np.array(delayList))
    ax.set_xlabel('token bucket rate')
    ax.set_ylabel('token bucket size')
    plt.show()

lam = 100000.0
r = 120000.0
b = 10

def testBPerTB(lam, r, b):
    delayList = []
    kList = [i+1 for i in range(40)]
    muList = [(10+i)/10.0*lam for i in range(10)]
    for mu in muList:
        delayList.append([])
        for k in kList:
            delayList[-1].append(bPerDelay(lam, k, r, b, mu))

    X = np.array(muList)
    Y = np.array(kList)

    Y, X = np.meshgrid(Y,X)

    # print X.tolist(), Y.tolist()
    # print tbList #, serList
    print(delayList)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, np.array(delayList))
    ax.set_xlabel('mu')
    ax.set_ylabel('k')
    plt.show()

def poiDelay(lam, r, b, mu):
    res1 = PoiTBDelay(lam, r, b)
    res2 = poiSerDelay(lam, mu, r, b)
    return res1 + res2

def bPerDelay(lam, k, r, b, mu):
    res1 = batchPerTBDelay(lam, k, r, b)
    res2 = bPerSerDelay(lam, k, mu, r, b)
    return res1+res2

# get resource for batched periodic
def getMuBPer(tar, lam, k, r, b):
    mu = Symbol('mu')
    if k <= b:
        return (k+1)/tar/2
    else:
    #elif (b+1)/mu <= 1/r:
        return ((b-1)*b + 2*k)/tar/2/k
    # elif (b+2)/mu <= 2/r:
    #
    # solve(bPerSerDelay(lam, k, mu, r, b) - tar, mu)
    # return res

# lam = 1000.0
# r = 1500.0
# k = 10
# b= 5
# mu = 3000.0
#
# res1 = []
# res2 = []
# res = []
# type = "poisson"
# for b in range(50):
#     if type == "batchperiodic" :
#         res1.append(batchPerTBDelay(lam, k, r, b+1))
#         res2.append(bPerSerDelay(lam, k, mu, r ,b+1))
#     elif type == "poisson" :
#         res1.append(PoiTBDelay(lam, r, b+1))
#         res2.append(poiSerDelay(lam, mu, r ,b+1))
#     res.append(res1[-1] + res2[-1])
# print res1
# print res2
# print res
# plt.plot(res1)
# plt.plot(res2)
# plt.plot(res)
# plt.show()

def testExtremeMu(r = 1000.0, b = 100):
    dList = []
    d2 = []
    for mu  in [r+r/10*i for i in range(100)]:
        if mu == r:
            dList.append(b/r)
            d2.append(b/mu)
        else:
            s = b*(b)/mu/2
            l = b/(mu - r)*r
            for i in range(int(ceil(l))):
                s += (l-i)*b/mu/l
            dList.append(s/(b+floor(l)) +1/mu)
            d2.append((b+1)/mu/4)
    plt.plot(dList)
    plt.plot(d2)
    print([dList[i] <= d2[i] for i in range(len(dList))])
    plt.show()

def testGetExtremeMu(tar = 0.0001):
    bList = [i+1 for i in range(10)]
    #rList = [10000+1000*(i+1) for i in range(20)]
    r = 10000.0
    muList = []
    for b in bList:
        mu = getExtremeMu(tar, b, r)
        muList.append(mu)
    plt.plot(muList)
    eList = [1.0*(b+1)/2/tar for b in bList]
    plt.plot(eList)
    print(muList, eList)
    plt.show()


# def testExtreme(r= 100000.0, b = 10, tar = 0.0002):
#     muList = [r+r/100000*(i+1) for i in range(100)]
#     dList = []
#     for x in muList:
#

def getExtremeMu(tar, b, r = 0):
    x = Symbol("x")
    res = solve(1.0*(b+1)*(x-r+2)/2/x/(x-r-1) - tar)
    #print res
    if res[0] >= r:
        return res[0]
    else:
        return res[-1]

def getPoiMu(lam, r, b, tar):
    delay1 = PoiTBDelay(lam, r, b)
    if tar-delay1 <= 0:
        return float('inf')
    mu = getMuPoi(tar - delay1, lam, getPoiCoV(lam, r, b))
    return mu

def getMu(lam, k, r, b, tar, type="poisson"):
    if type == "poisson":
        # print "******** getMu **********"
        # print lam ,r ,b
        mu1 = getPoiMu(lam, r, b, tar)
        if mu1 == float('inf'):
            return float('inf')
        mu2 = getExtremeMu(tar, b, r)
        # print r, b
        # print mu1, mu2, r, lam
        return max([mu1, mu2, r, lam])
    elif type == "batchperiodic":
        mu1 = getExtremeMu(tar, b, r)
        # print r, b
        # print mu1, mu2, r, lam
        return max([mu1, r, lam])

# lam = 100000.0
# alpha = 0.2
# tar = 0.00001
# mu = 198084.0
# r = 190000
# b = 2
#
# tar = 0.0001
# r = 105000.0
# b = 20
# mu = 105891.0
#
tar = 0.00005
mu = 113498.0
r = 110000
b = 10

# for poisson
def SplitTokenBucket(lam, alpha, r, b, tar, type ="poisson", per = 20):
    res = []
    kList = []
    b = int(b)
    if type == "poisson":
        for i in range(per-1):
            res.append([])
            kList.append([])
            for j in range(b-1):
                if r*(i+1)*1.0/per >= lam*alpha and r*(per-i-1)*1.0/per >= lam*(1-alpha):
                    # part1 = getMu(lam*alpha, 1, r*(i+1)/100.0, j+1, tar)
                    # part2 = getMu(lam*(1-alpha), 1, r*(100-i-1)/100.0, b-j-1, tar)
                    # res[-1].append(part1+part2)
                    total = []
                    for k in range(4*per):
                        part1 = ( getMu(lam*alpha, 1, r*(i+1)*1.0/per, j+1, tar*(k+1)*1.0/per) )
                        if part1 == float('inf'):
                            total.append(float('inf'))
                            continue
                        part2 = ( getMu(lam*(1-alpha), 1, r*(per-i-1)*1.0/per, b-j-1, tar*(1-alpha*(k+1)*1.0/per)/(1-alpha)) )
                        total.append(part1 + part2)
                        # print alpha, (i+1)*1.0/per, j+1, tar*(k+1)/per, tar*(1-alpha*(k+1)/per)/(1-alpha), tar
                        # print part1, part2, total[-1]
                    res[-1].append(min(total))
                    if min(total) == float('inf'):
                        kList[-1].append(0)
                    else:
                        kList[-1].append(np.where(np.array(total) == np.min(total))[0][-1]+1)
                    #print total
                    #print "alpha", alpha, np.where(np.array(total) == np.min(total)), np.min(total)
                    print("."),
                else:
                    res[-1].append(float('inf'))
                    kList[-1].append(0)
    res =  np.array(res)
    # print np.min(res)
    # print np.where(res == np.min(res))
    print("********************************")
    print ("SplitTokenBucket", lam, alpha, r, b, tar)
    rList, bList = np.where(res == np.min(res))
    print("r", r*(rList[-1]+1)*1.0/per, (rList[-1]+1)*1.0/per)
    print("b", bList[-1]+1)
    print("k", kList[rList[-1]][bList[-1]], "tar", tar*(kList[rList[-1]][bList[-1]]+1)*1.0/per)
    print("res", res)
    print("kList", kList)
    print(np.min(res), np.where(res == np.min(res)))
    return np.min(res), (rList[-1]+1)*1.0/per, bList[-1]+1, kList[rList[-1]][bList[-1]]





#def getBatchPCov(lam, k, r, b):



# rho = 0.9
# x = [i for i in range(20)]
# y = [probMD(rho, i) - probMD(rho, i+1) for i in x]
# z = [probMD(rho, i+1) for i in x]
# t = [1- probMD(rho, i) for i in x]
# plt.plot(x, y, label="y")
# plt.plot(x, z, label="z")
# plt.plot(x, t, label="t")
# plt.legend()
# plt.show()

def serDelay(lam, mu, s1, s2):
    rho = 1.0*lam/mu
    return rho/(1-rho)/mu*(s1+s2)/2

def getCap(t, lam, s1, s2):
    c = Symbol("c")
    res = solve(lam/(c-lam)/c*(s1+s2)/2 -t)
    if res[0] > 0:
        return res[0]
    else:
        return res[1]

def totalDelay(r, b, lam, mu):
    lam = lam*1.0
    mu = mu*1.0
    r = r*1.0
    rho = lam/r
    if lam/mu >= 0.99999:
        return 0, float('inf'), float('inf')
    alpha = probMD(rho, b)
    tb = alpha/(2*r*(1-rho))
    ## g/m/1
    # def g(x):
    #     return alpha*exp(-mu/r*(1-x)) + (1-rho)/rho*alpha*(lam*exp(rho))*lam/(lam + mu*(1-x)) + (1-alpha/rho)*lam/(lam + mu*(1-x)) - x
    # xi = optimize.bisect(g, 0, 1)
    # server = xi/mu/(1-xi)
    # print alpha, rho, xi
    ## m/m/1
    # rho = lam/mu/n
    # server = (1-alpha)*rho/mu/(1- rho)
    return tb, 0 +1.0/mu, tb+1.0/mu


# ####################### bstexp ##########################
# b = 50
# pRateList = [200, 300, 400, 500, 600, 700, 800, 900]
# distrNameAList = ["bstexp10", "bstexp20", "bstexp30", "bstexp40", "bstexp50", "bstexp60", "bstexp70", "bstexp80", "bstexp90", "bstexp100"]
# burstList = [int(d[6:]) for d in distrNameAList]
# bRate = 1000
# mu = 1000
# sigB = 1/mu
#
# delayProb = 1.0 - np.array([[0.88629400000000003, 0.815469, 0.73523000000000005, 0.64748500000000009, 0.54643299999999995, 0.43456799999999995, 0.31290899999999999, 0.16395100000000001], [0.41937100000000005, 0.37743199999999999, 0.334121, 0.28320699999999999, 0.23168900000000003, 0.180255, 0.12626499999999999, 0.066153999999999991], [0.27547999999999995, 0.24563499999999999, 0.21484699999999998, 0.18105499999999999, 0.14553699999999997, 0.114597, 0.073994000000000004, 0.038581999999999998], [0.20475300000000002, 0.18149499999999999, 0.15854099999999999, 0.132629, 0.107847, 0.078874, 0.053928000000000004, 0.025336000000000001], [0.16286299999999998, 0.14418900000000001, 0.12436499999999998, 0.104313, 0.085378000000000009, 0.064152999999999988, 0.044726000000000002, 0.020229999999999998], [0.135549, 0.11858099999999999, 0.10478399999999999, 0.087314000000000003, 0.070844999999999991, 0.05323, 0.035198, 0.017144], [0.11590899999999998, 0.10262200000000002, 0.088164000000000006, 0.072765999999999997, 0.060354000000000005, 0.044324999999999996, 0.030592000000000001, 0.013849999999999998], [0.10142699999999999, 0.088717000000000004, 0.077104000000000006, 0.065338000000000007, 0.052844999999999996, 0.039088999999999999, 0.026654000000000001, 0.012590999999999998], [0.090015999999999999, 0.078148999999999996, 0.06652799999999999, 0.056667999999999996, 0.044767000000000001, 0.035091999999999998, 0.022051000000000001, 0.012247000000000001], [0.080370999999999998, 0.070257, 0.060678999999999997, 0.051583000000000004, 0.040783, 0.030633000000000001, 0.020702999999999999, 0.011035]])
#
# maxList = []
# minList = []
# i = 0
# j = 0
# for b in burstList:
#     maxList.append([])
#     minList.append([])
#     j = 0
#     for p in pRateList:
#         lam, sigA = burstSig(p, b)
#         alpha = delayProb[i,j]
#         print alpha
#         sigA = sqrt(alpha/bRate**2 + (1-alpha)*sigA**2 - alpha/p**2)
#         ma, mi = approx(lam, sigA , mu, sigB)
#         maxList[-1].append(ma)
#         minList[-1].append(mi)
#         j += 1
#     i += 1
# X = np.array(burstList)
# Y = np.array(pRateList)
#
# Y, X = np.meshgrid(Y,X)
#
# print X.tolist(), Y.tolist(), maxList, minList
# maxList = np.array(maxList)
#
# print maxList.shape, delayProb.shape
# maxList = np.multiply(maxList, delayProb)
# fig = plt.figure()
# ax = fig.add_subplot(211, projection='3d')
# ax = fig.add_subplot(212, projection='3d')
# ax.plot_wireframe(X, Y, np.array(maxList), color="red")
# ax.plot_wireframe(X, Y, np.array(minList), color ="blue")
# plt.show()



# for b in bList:
#     t.append([])
#     for rho in rhoList:
#         t[-1].append(tbDelay(rho, r, b-1))
# X = np.array(bList)
# Y = np.array(rhoList)*r
# Y, X = np.meshgrid(Y,X)
# Z1 = np.array(t)
# print t
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(X,Y,Z1)
# ax.set_xlabel('token bucket size')
# ax.set_ylabel('arrival rate')
# ax.set_zlabel('token bucket delay')
# plt.show()

# t = []
# r = 500.0
# pRateList = [100, 150, 200, 250, 300, 350, 400, 450]
# bList = [20]
# rhoList = [i/r for i in pRateList]
# b = 20
# t = []
# for rho in rhoList:
#     t.append(tbsDelay(rho, r, b, 2))
# print t
#

def smallBurstDelay(rho, r, b, k, i=0):
    #print r,b,k,i
    pb = probMD(rho, 1.0*(b-i)/k)
    pb1 = probMD(rho, 1.0*(b-i-1)/k)
    # result = 1.0*(k-i)/k*(k-i-1)/2/r*(pb1-pb)
    # result += 1.0*(k-i)/k*(1.0*k/2/r/(1-rho)*pb1)
    # result -= 1.0*(k-i)/k*(1.0*k/2/r/(1-rho)+1.0/r)*pb
    result =  1.0*(k-i)/k*((1.0*k/2/r/(1-rho) + 1.0*(k-i-1)/2/r)*(pb1-pb) - 1.0/r*pb)
    if i == min([b,k])-1:
        return result
    else:
        return result + smallBurstDelay(rho, r, b, k, i+1)


def burstDelay(rho, r, b, k):
    pb = probMD(rho, 1.0*b/k)
    result =  (1.0*k/2/r/(1-rho) + 1.0*(k+1)/2/r)*pb
    #result2 = smallBurstDelay(rho,r,b,k,0)
    pb1 = 1.0
    if b - k > 0:
        pb1 = probMD(rho, 1.0*(b-k)/k)
    #result2 =
    if b >= k:
        pk = probMD(rho, (b-k)/k)
        result2 = (1.0*k/2/r/(1-rho) + 1.0/2/r) * (pk - pb)
    else:
        result2 = (1.0*k/2/r/(1-rho) + 1.0*(k+1-b)/2/r)*(1-pb)
    print(result, result2)
    return result


def serDelayMM1(lam, mu):
    return 1.0/mu/(1- 1.0*lam/mu)

def serDelayApprox(lam, mu, c1, c2=1):
    rho = 1.0*lam/mu
    return rho*(c1+rho**2)/mu/(1-rho)/(1+rho**2) +1/mu

# def serDelayApprox():
#     lam = 1000.0
#     mu = 1100.0
#     for i in range(100):


############################### bstexp #######################################
# t = []
# r = 1000.0
# pRateList = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
# # bList = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# #bList = [1, 2 , 5, 10, 20, 30, 40, 50]
# #bList = [5,10,15,20]
# # bList = [5, 10, 15, 20, 25, 30, 35, 40]
# # rhoList = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# distrNameAList = ["bstexp1", "bstexp2", "bstexp5", "bstexp10", "bstexp20", "bstexp30", "bstexp40", "bstexp50"]
#
# #pRateList = [100, 200, 300, 400, 500, 600, 700, 800, 900]
# #bList = [1,2,3,4, 5, 6, 7, 8]
# rhoList = [i/r for i in pRateList]
#
# b = 20
# dList = []
# #for b in bList:
# for d in distrNameAList:
#     t.append([])
#     s = int(d[6:])
#     dList.append(s)
#     for rho in rhoList:
#         t[-1].append(burstDelay(rho, r, b, s))
# X = np.array(dList)
# Y = np.array(rhoList)*r
# Y, X = np.meshgrid(Y,X)
# Z1 = np.array(t)
#
# print Z1
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(X,Y,Z1)
# ax.set_xlabel('bulk size')
# ax.set_ylabel('arrival rate')
# ax.set_zlabel('token bucket delay')
# plt.show()

# ####################### exp #########################
# bRate = 1000.0
# mu = 1000.0
#
# tbList = []
# serList = []
# p = 1000
# bRateList = [1010+i*2 for i in range(45)]
# bList = [ i*5 for i in range(20)]
# pNum = 0
#
# for b in bList:
#     tbList.append([])
#     #serList.append([])
#     #for p in pRateList:
#     for r in bRateList:
#         tb = tbDelay(1.0*p/r, r, b)
#         tbList[-1].append(tb)
#         #serList[-1].append(ser)
#
# X = np.array(bList)
# Y = np.array(bRateList)
#
# Y, X = np.meshgrid(Y,X)
#
# print X.tolist(), Y.tolist()
# print tbList #, serList
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(X, Y, np.array(tbList))
# ax.set_xlabel('token bucket size')
# ax.set_ylabel('token bucket rate')
# plt.show()

####################### exp resource #########################

def runExp(lam, tar, k = 1, type = "poisson", xsize = 200, ysize = 40):
    if type == "poisson":
        rList = [lam+(i+1)*lam/20.0 for i in range(xsize)]
        bList = [i+1 for i in range(ysize)]
        res = []

        for r in rList:
            res.append([])
            for b in bList:
                #re = PTBDelay(lam, r, b)
                re = getMu(lam, k, r, b, tar, type)
                #print lam, r, b, tar, re
                if re == float('inf'):
                    res[-1].append(100*lam)
                else:
                    res[-1].append(re)

        # print res
        # print "###", tar, "###"
        # print np.min(res)
        #print res
        res = np.array(res)
        rIndex,bIndex = np.where(res == np.min(res))
        r = lam+(rIndex[-1]+1)*lam/20.0
        b = bIndex[-1]+1
        # for i in range(len(a)):
        #     print lam+(a[i]+1)*lam/10.0, b[i]+1

        # X = np.array(rList)
        # Y = np.array(bList)
        #
        # Y, X = np.meshgrid(Y,X)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_wireframe(X, Y, np.array(res))
        # ax.set_xlabel('Token Bucket Rate')
        # ax.set_ylabel('Token Bucket Size')
        # ax.set_zlabel('Required Service Rate')
        # ax.set_zlim([0, 5*lam])
        # plt.savefig(str(int(lam))+"_"+str(int(tar*100000))+".png")

        return r, b, np.min(res), res
    elif type == "batchperiodic":
        rList = [lam+(i+1)*lam/50.0 for i in range(xsize)]
        for r in rList:
            b = k
            mu = getExtremeMu(tar, k, r)
            return mu, b, r


if __name__ == "__main__":
    #lam  = 100000.0
    #for t in [0.00001, 0.000015, 0.00002, 0.000025, 0.00003]:
    k = 1
    for t in [0.0001]:
        rList = []
        bList = []
        kList = []
        mList = []
        #lList = [1000.0*(i+1) for i in range(10)]
        lList = [10000.0+1000*(i+1) for i in range(10)]
        print(lList)
        for lam in lList:
            r, b, minRes, res = runExp(lam, t, k, "poisson", 50, 20)
            rList.append(r)
            bList.append(b)
            kList.append(k)
            mList.append(minRes)
            print("###############", lam, "#################")
            print(r, b, k, minRes)
            print(lam, r, b, t)
        print("\n###########################")
        print("rList", rList)
        print("bList", bList)
        print("kList", kList)
        print("mList", mList)
        print("lList", lList)
            #print "res", res
            # if b == 1:
            #     b = 2
            # srcList = []
            # rList = []
            # bList = []
            # delayList = []
            # for i in range(10):
            #     src, rSplit, bSplit, delaySplit = SplitTokenBucket(lam, (i+1)/20.0, r, b, t)
            #     srcList.append(src)
            #     rList.append(rSplit)
            #     bList.append(bSplit)
            #     delayList.append(delaySplit)
            #
            # srcList = np.array(srcList)
            # index = np.where(srcList == np.min(srcList))
            # print ""
            # print "############### Split for ", t, "#################"
            # print "alpha, min, r, b, delta ", (index[0][-1]+1)/20.0, np.min(srcList), rList[index[0][-1]], bList[index[0][-1]], delayList[index[0][-1]]
            # print "src, rsplit, bsplit, delaysplit", srcList, rList, bList, delayList



# plist = []
# for i in range(1,20):
#     plist.append(getDelay(0.9,i))
# print plist
# plt.plot(np.arange(1,20), plist, "-o")
# plt.show()
