#!/usr/local/bin/python

from math import factorial, exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import sys

intList = [5000, 1000, 500, 100, 50, 10]
trate = [1000000.0/i for i in intList]
cpuList = [1.0, 4.2, 8.0, 26.9, 38.7, 52]
rate = [0.314, 1.45, 2.72, 9.42, 13.6, 21.1]
rate = [r*1000 for r in rate]
latency = [8.1924e+03, 7.9385e+03, 7.8343e+03,  8.1685e+03, 8.6729e+03, 8.6729e+03 ]
latency = [l/1000000.0 for l in latency]

#plt.plot(trate, cpuList, 'r-.')
plt.figure(1)
plt.subplot(211)
plt.plot(rate, cpuList, 'bo-')
plt.ylabel("CPU utilization (%)")
plt.xlabel("Message rate (Kbps)")
plt.subplot(212)
plt.plot(rate, latency, 'ro-')
plt.ylim(0, 0.01)
plt.ylabel("Latency (ms)")
plt.xlabel("Message rate (Kbps)")
plt.show()
sys.exit()



#from mpl_toolkits.mplot3d import Axes3D
#from theory import getDelay, getLatency, totalDelay

# bucket = np.arange(1,21)
# bresult= [0.00409438016142, 0.0033155469912, 0.00267805247694, 0.00217196080862, 0.00179592654568,
# 0.00143718393687, 0.00116060379269, 0.000978849410248, 0.000755804749056, 0.000629652721451,
# 0.000509918882204, 0.000438399316067, 0.000338310877662, 0.000280665269416, 0.000244070153101,
# 0.000172161374231, 0.000149499687789, 0.000121459034788, 9.30199199637e-05, 7.75854592678e-05]
#
# dlist = []
# for i in bucket:
#     dlist.append(getDelay(0.9,i))
# plt.plot(bucket, dlist, "-")
# plt.plot(bucket, np.array(bresult)*1000, 'o')
#
#
# legendList = ['theory', 'simulation']
# plt.legend(legendList, loc='upper right')
# plt.xlabel('bucket size')
# plt.ylabel('average latency (ms)')
# plt.show()
#
#
# rate = range(1000, 0, -100)
# rresult = [0.000644522106328, 0.000720025905961, 0.000833121678584, 0.000895596093789, 0.00101505313479, 0.00128537828299, 0.0015555967225, 0.00209048499208, 0.00313702591988, 0.00616596723663]
#
# d = getDelay(0.9,10)
# dlist = [d/(0.1*(10-i)) for i in xrange(10)]
# plt.plot(rate, dlist, "-")
# plt.plot(rate, np.array(rresult)*1000, 'o')
#
#
# legendList = ['theory', 'simulation']
# plt.legend(legendList, loc='upper right')
# plt.xlabel('bucket rate')
# plt.ylabel('average latency (ms)')
# plt.show()


# interval = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
# possion_result = [0.00131450177183, 0.0015070228446, 0.0016599388821, 0.00161004213216, 0.0015961046498, 0.00146764593642, 0.00144323696861, 0.00140161336144]

# b_result = np.array([0.000590173923748, 0.00223829675234, 0.00349507988276, 0.00554642015014,
# 0.00793513324288, 0.0117633777557, 0.0131939118183, 0.0152916625152,
# 0.0164328270268, 0.0222740491034, 0.0260078343715, 0.026809945385])*1000
#
# s_result = np.array([0.00945765245304, 0.00211677915805, 0.00153174938914, 0.00129779523745,
# 0.00117139743497, 0.00108493653043, 0.00106551896397, 0.00105197218411,
# 0.00104446798347, 0.00100978968546, 0.00100655731514, 0.00100732780158])*1000

# b_result = np.array([0.000556862018053, 0.00226279268004, 0.00373865173411, 0.00554710361537,
# 0.00823055300791, 0.0117136387434, 0.0128881523441, 0.0166177605538,
# 0.016524255912, 0.0221778073856, 0.0257723768586, 0.0267681413876])*1000
#
# s_result = np.array([0.0092905418664, 0.0021032834536, 0.00152273155381, 0.00129437599152,
# 0.00116818969581, 0.00108350271543, 0.00106527594669, 0.00105236611835,
# 0.0010370405632086788, 0.00101056378729, 0.00100803562565, 0.00100450341295])*1000

######### best result
# b_result = np.array([0, 0, 0, 1.24239805608e-06,
# 1.34584248141e-05, 4.84002550078e-05, 0.000117872470448, 0.000214928715841,
# 0.000351449322535, 0.000594727983716, 0.000975557026088, 0.00151676371671])*1000
#
# s_result = np.array([0.00980780382356, 0.00251265470871, 0.00181477766449, 0.00156341771023,
# 0.00142817810789, 0.00134093139615, 0.00128743022846, 0.00124448951586,
# 0.00121615276775, 0.00118856757796, 0.00116722571315, 0.00115158808519])*1000


# rate = 2000
# bucketSize = 200
# w_result = b_result + s_result
#
# x = range(2,14)
# b_theory = np.array([getLatency(rate/i, 0.9, bucketSize/i) for i in x])
# s_theory = np.array([1.0/(1000 - 1800.0/i) for i in x])*1000
# print b_theory
# print s_theory
# plt.plot(x, b_result, '*')
# plt.plot(x,b_theory)
# plt.plot(x, s_result, '.')
# plt.plot(x, s_theory)
# plt.plot(x, w_result, 'o')
# plt.plot(x, b_theory + s_theory)
#
# legendList = ['token_bucket_sim', 'token_bucket_theory', 'server_sim', 'server_theory', 'latency_sim', 'latency_theory']
# plt.legend(legendList, loc='upper right')
# plt.xlabel('number of servers')
# plt.ylabel('average latency (ms)')
# plt.show()


######### draw theory


# b_result = np.array([0, 0, 0, 1.24239805608e-06,
# 1.34584248141e-05, 4.84002550078e-05, 0.000117872470448, 0.000214928715841,
# 0.000351449322535, 0.000594727983716, 0.000975557026088, 0.00151676371671])*1000
#
# s_result = np.array([0.00980780382356, 0.00251265470871, 0.00181477766449, 0.00156341771023,
# 0.00142817810789, 0.00134093139615, 0.00128743022846, 0.00124448951586,
# 0.00121615276775, 0.00118856757796, 0.00116722571315, 0.00115158808519])*1000
#
# util = 0.9
# rate = 2000
# prate = 1000
# bucketSize = 200
# start = 2
# x = range(start,len(b_result)+2)
#
# b_theory = []
# s_theory = []
# for i in x:
#     b, s, t = totalDelay(rate, bucketSize, rate*util, prate, i)
#     b_theory.append(b)
#     s_theory.append(s)
#     print b, s, t
# # b_theory = np.array([getLatency(rate/i, util, bucketSize/i) for i in x])
# # s_theory = np.array([1/(prate - start*prate*0.9*1.0/i) for i in x])
# #
#
# w_result = b_result + s_result
# plt.plot(x, b_result, '*')
# plt.plot(x,np.array(b_theory)*1000)
# plt.plot(x, s_result, '.')
# plt.plot(x, np.array(s_theory)*1000)
# plt.plot(x, w_result, 'o')
# plt.plot(x, np.array(b_theory)*1000 + np.array(s_theory)*1000)
#
# legendList = ['token_bucket_sim', 'token_bucket_theory', 'server_sim', 'server_theory', 'latency_sim', 'latency_theory']
# plt.legend(legendList, loc='upper right')
# plt.xlabel('number of servers')
# plt.ylabel('average latency (ms)')
# plt.show()


######### drop load increase


# r = 500
# b = 20
# mu = 500
# opt_n = 1
# x = []
# nList = []
# bList = []
# sList = []
# tList = []
#
# for i in xrange(49):
#     lam = (i+1)*10
#     x.append(lam)
#     for j in xrange(4):
#         tb, ts, t = totalDelay(r, b, lam, mu, j+1)
#         if len(bList) < j+1:
#             bList.append([])
#             sList.append([])
#             tList.append([])
#         bList[j].append(tb)
#         sList[j].append(ts)
#         tList[j].append(t)
#
# print bList
# print sList
# print tList
# print nList
# #plt.plot(x, b_result, '*')
# plt.plot(x,np.array(tList[0])*1000)
# #plt.plot(x, s_result, '.')
# plt.plot(x, np.array(tList[1])*1000)
# #plt.plot(x, w_result, 'o')
# plt.plot(x, np.array(tList[2])*1000)
# plt.plot(x, np.array(tList[3])*1000)
#
# legendList = ['1 server', '2 servers', '3 servers', '4 servers']
# plt.legend(legendList, loc='upper left')
# plt.xlabel('arrival rate')
# plt.ylabel('average latency (ms)')
# plt.ylim(0, 15)
# plt.show()


############################

# lam = 50.0
# r = 500
# b = [2 , 4, 8, 16, 32, 128]
# bLegend = ["token bucket b="+str(i) for i in b]
# #mu = 500
# opt_n = 1
# x = []
# bList = []
# sList = []
#
# for i in xrange(100):
#     mu = lam + (i+1)*0.5
#     x.append(lam/mu)
#     for j in xrange(len(b)):
#         tb, ts, t = totalDelay(mu, b[j], lam, mu, 1)
#         if len(bList) < j+1:
#             bList.append([])
#         bList[j].append(tb*1000)
#     sList.append(lam/mu/(mu - lam)*1000)
#
# plt.plot(x,sList)
# for j in xrange(len(b)):
#     plt.plot(x, bList[j])
# legendList = ["queuing time"] + bLegend
# plt.legend(legendList, loc='upper left')
# plt.xlabel('utilization')
# plt.ylabel('average latency (ms)')
# plt.ylim(0, 400)
# plt.show()


### increase server

# lam = 500.0
# r = 500
# b = [2 , 4, 8, 16, 32, 64]
# bLegend = ["token bucket b="+str(i) for i in b]
# #mu = 500
# opt_n = 1
# x = []
# bList = []
# sList = []
# ratioA = 3
# ratioB = 4
# ratio = ratioA*1.0/ratioB
#
# for i in xrange(100):
#     mu = lam + (i+1)*5.0
#     x.append(lam/mu)
#     for j in xrange(len(b)):
#         tb1, ts1, t1 = totalDelay(mu*ratioA, b[j]*ratioA, lam*ratioA, mu, ratioA)
#         tb2, ts2, t2 = totalDelay(mu*ratioA, b[j]*ratioA, lam*ratioA, mu, ratioB)
#         if len(bList) < j+1:
#             bList.append([])
#         bList[j].append((tb2-tb1)*1000)
#         print (tb2-tb1)*1000
#     sList.append((lam/mu/(mu - lam) - ratio*lam/mu/(mu - lam*ratio))*1000)
#
# print x
# plt.plot(x,sList)
# for j in xrange(len(b)):
#     plt.plot(x, bList[j])
# legendList = ["queuing time"] + bLegend
# plt.legend(legendList, loc='upper left')
# plt.xlabel('utilization')
# plt.ylabel('change in latency (ms) when increase a server')
# plt.ylim(0, 40)
# plt.show()


####### plot

# r = 1000.0
# b = 20
# lamList = [950, 955, 960, 965, 970, 975, 980, 985, 990, 995]
# mu = 1000.0
# bList = []
# sList = []
# for lam in lamList:
#     lam *= 1.0
#     tb, ts, t = totalDelay(r, b, lam, mu, 1)
#     bList.append(tb)
#     sList.append(ts)
#
#
# plt.plot(lamList, bList)
# plt.plot(lamList, sList)
# legendList = ["token time", "server time"]
# plt.legend(legendList,loc="upper left")
# plt.ylim(-0.01, 0.3)
# plt.xlim(945, 1000)
# plt.show()


##### draw

bList = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

rList = [1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0, 1500.0]

tbList = [[0.0086655012164963772, 0.0069462937026606858, 0.0061119878316403089,
0.004497763222681749, 0.003876938261844539, 0.003390546964274696,
0.0027159118226675167, 0.0024347461060429659, 0.0019866744028636291,
0.0017593567570707523, 0.0013200001731956527, 0.00099414911505809364],
[0.0037627201922375147, 0.0025625427437552606, 0.0017829949842417827,
0.0012298620493933936, 0.00082696692817260838, 0.00058308794201233677,
0.00040019521369635792, 0.00026437935404872629, 0.00018632066836472602,
0.00012402267729263054, 7.0943063294622068e-05, 5.6494559620774469e-05],
[0.0022095713301978964, 0.0013096853587949539, 0.00072713937256503071,
0.00042894764353715976, 0.00024653261679521114, 0.00013736124194561419,
8.0361412315634121e-05, 4.4430293019744773e-05, 2.9406724790343939e-05,
1.1795146423416464e-05, 8.5039484345559113e-06, 5.9312839614103951e-06],
[0.0014611881929298993, 0.00072593918371486915, 0.00035256953645628077,
0.00017954436487361305, 8.4489334837231018e-05, 4.1318479906370685e-05,
2.1974399513942762e-05, 1.0698779145992274e-05, 4.4373112312226795e-06,
3.028488661354572e-06, 1.2635151914837706e-06, 5.3085700028457263e-07],
[0.0010424890384005199, 0.00044756733360717705, 0.00018240195443261018,
7.9531375689721311e-05, 3.580796894964474e-05, 1.3407249793421535e-05,
5.0436495208507264e-06, 2.3280473410579587e-06, 1.0953525661858181e-06,
4.1576557217018718e-07, 7.1502696550169271e-08, 3.0380263923825622e-08],
[0.0007794387364076905, 0.00028877273908869651, 0.00010307204425758113,
3.7338351682952915e-05, 1.4652217675302338e-05, 4.8590325874305792e-06,
2.377105866288889e-06, 7.8939033892313559e-07, 1.2753345779798336e-07,
1.5646413789522741e-07, 5.0947068469440634e-09, 2.7730064349452735e-08],
[0.00060260250620125587, 0.00019080787103471121, 5.7977087554087513e-05,
1.8072280771463227e-05, 6.3648178513343291e-06, 1.9166132329105377e-06,
6.420684786416018e-07, 2.2753073556394841e-07, 5.0865510844266733e-08,
9.5950108645411091e-09, 1.7806682401669604e-09, 4.703149038959964e-10],
[0.00047860166509120731, 0.00013136002069690563, 3.6838882146894813e-05,
9.6232019150645455e-06, 3.0314451320358898e-06, 1.0411160334375608e-06,
1.7512695237192022e-07, 6.5362352172974166e-08, 5.7878174140796546e-09,
3.1298298001729565e-10, 0.0, 0.0],
[0.0003868298345192014, 9.373483983780517e-05, 2.4924532266800483e-05,
5.2633050377738303e-06, 1.3950417193645079e-06, 2.6167633881354963e-07,
8.4777204153101606e-08, 1.3302193317463208e-08, 1.5399734173206525e-08,
0.0, 0.0, 0.0],
[0.00031714521472453683, 6.7876044345209876e-05, 1.5430425620576841e-05,
2.8363864016281357e-06, 7.2926797369432278e-07, 1.1011910837496543e-07,
7.7841931393777485e-09, 1.3981584637986088e-08, 0.0,
7.8820269800417015e-11, 0.0, 0.0]]

tsList = [[0.006679677862655292, 0.0068770308868411735, 0.0074732966659507918,
0.0077348077227535148, 0.0078105416045624043, 0.008147963937665325,
0.0084921141776806743, 0.008752305338601777, 0.0088621115063590317,
0.0090566327780958918, 0.0093905065648900807, 0.0094743977123601664],
[0.0076441088658002893, 0.0081786353435035122, 0.0087498405194113942,
0.0090029671774246641, 0.0092297928778259427, 0.0093696536701099262,
0.0096617572741030684, 0.0096833025293018727, 0.010003607981588163,
0.0098724565038900442, 0.0098445482952155567, 0.0098002479005328443],
[0.0086956475242956251, 0.0090666151686463504, 0.0093720184341949571,
0.0094531775361485718, 0.0098398963059626848, 0.0098212355945071234,
0.010018041874352037, 0.0098921459096796907, 0.0098956955424670603,
0.010029270355956356, 0.010064604268816387, 0.01001023740313349],
[0.0090904504424792146, 0.009225412608549784, 0.0096016456056585951,
0.0099027595356318918, 0.010039369303912821, 0.0097721368289364549,
0.010042447619923751, 0.010045292325722056, 0.010007482265982762,
0.0099870953803561369, 0.010184912443106139, 0.0098858368161917395],
[0.0093137423055012838, 0.0095619384206493182, 0.0096557424523883665,
0.0099299592347444968, 0.010063250392674448, 0.010127057969903762,
0.0098904826150556166, 0.010036861438288495, 0.0099961991171080636,
0.0099088390440836595, 0.0096536991934565494, 0.010030348539790601],
[0.0093682707003560229, 0.010176501383261838, 0.0098165119959769571,
0.0097205414379321099, 0.010006320447941785, 0.0099182604435972422,
0.010001961172821086, 0.0098252164378607853, 0.0099495692669901714,
0.010102707098157179, 0.010090222760704749, 0.0099789223025760522],
[0.0096336326797271388, 0.0097238686533284747, 0.0098371166194679786,
0.0097904040711137234, 0.0099297341641229348, 0.010001390250069974,
0.0099266848307628282, 0.0098179879154293419, 0.0098578389481048211,
0.0098189810593029593, 0.010100181267139989, 0.0099267782464376418],
[0.0095568714523684116, 0.009885090780846607, 0.0097968008289410768,
0.0097222136568735906, 0.0099612086330636024, 0.010063981692023737,
0.010186485114693453, 0.010036024516736682, 0.0099838449228713509,
0.010130933882378523, 0.010193518255552692, 0.0099776912059497298],
[0.0098415813407483066, 0.0097824395111458257, 0.009936011172877738,
0.010052051864369575, 0.010126848886467584, 0.010142662759735766,
0.010290573689306257, 0.0099869683348446474, 0.0098433343622829003,
0.0098570165778807204, 0.010013374979903155, 0.010064330226453103],
[0.0097614194737020623, 0.009815994410360249, 0.0099672335642590013,
0.0099349179582449675, 0.0098621461642761175, 0.010137879445556835,
0.009970959157126022, 0.010194055612801445, 0.0099125417813472286,
0.0098741304370536624, 0.0099527508964485801, 0.009803767794647502]]

X = np.array(rList)
Y = np.array(bList)

Y, X = np.meshgrid(Y,X)

#
# fig = plt.figure()
#
# ax = Axes3D(fig) #<-- Note the difference from your original code...
#
# cset = ax.contour(X, Y, np.array(tbList))
# ax.clabel(cset, fontsize=9, inline=1)
# plt.show()
# #


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, np.array(tbList))
ax.set_xlabel('token bucket size')
ax.set_ylabel('token bucket rate')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, np.array(tsList))
ax.set_xlabel('token bucket size')
ax.set_ylabel('token bucket rate')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, np.array(tbList)+np.array(tsList))
ax.set_xlabel('token bucket size')
ax.set_ylabel('token bucket rate')
plt.show()
