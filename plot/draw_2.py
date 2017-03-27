from math import factorial, exp, sqrt, floor
import numpy as np
import matplotlib.pyplot as plt
from theory import getPoiMu, getExtremeMu


x = [10000.0*(i+1) for i in xrange(19)]
tar = 0.0001
rList = [19500.0, 28000.0, 37500.0, 46000.0, 55000.0, 66000.0, 77000.0, 84000.0, 94500.0, 105000.0, 115000.0, 126000.0, 136500.0, 147000.0, 157500.0, 168000.0, 178500.0, 189000.0, 199500.0 ]
bList = [1, 1, 2, 8, 10, 1, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
muList = [19753.7522173733, 28000.0, 37500.0, 47198.1842618215, 56813.5518752162, 66000.0, 77000.0, 86607.6073951275, 96137.9501817806, 105891.896900201, 115754.718207325, 126000.0, 136500.0, 147000.0, 157500.0, 168000.0, 178500.0, 189000.0, 199500.0]
#mList = [getPoiMu(x[i], rList[i], bList[i], tar) for i in xrange(len(rList))]
extList = [getExtremeMu(tar, b) for b in bList]

fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(x, rList)
plt.plot(x, muList)
plt.plot(x, extList)
legendList = ['service rate', 'r', r"$f(b,\delta)$"]
plt.legend(legendList, loc='lower right')
plt.ylabel("Rate (per second)")
plt.xlabel("Message Rate (per second)")

ax = fig.add_subplot(212)
plt.plot(x, bList)
plt.ylabel("Token Bucket Size")
plt.xlabel("Message Rate (per second)")
plt.show()


p1List = [23250.9005733845, 29000.0, 37500.0, 48000.0, 57500.0, 66000.0, 77000.0, 88000.0, 99000.0, 110000.0]
p1List += [126000.0, 136500.0, 147000.0, 157500.0, 168000.0, 178500.0, 189000.0, 199500.0, 210000.0]
y = [10000*(i+2) for i in xrange(10)]
fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(x, bList)
bList = [(i+1)*2 for i in xrange(len(x))]
plt.plot(x, bList)
legendList = ['optimal b', 'approximated b']
plt.legend(legendList, loc='lower right')
plt.ylabel("Token Bucket Size")
plt.xlabel("Message Rate (per second)")

ax = fig.add_subplot(212)
plt.plot(x, muList)
extList = [getExtremeMu(tar, b) for b in bList]
plt.plot(x, extList)
apMList = [getPoiMu(x[i], extList[i], bList[i], tar) for i in xrange(len(x))]
apM2List = [apMList[i]*2 for i in xrange(len(y))]
plt.plot(x, apMList)
plt.plot(x,p1List)
plt.plot(y,apM2List)
print muList
print extList, bList
print apMList
legendList = ['optimal service rate', r"approximated r, $f(b,\delta)$", 'approximated service rate', 'fix b to 1', 'with 2 servers']
plt.legend(legendList, loc='lower right')
plt.ylabel("Rate (per second)")
plt.xlabel("Message Rate (per second)")
plt.show()
