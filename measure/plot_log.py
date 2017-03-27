#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, floor, ceil
import scipy as sp
import scipy.stats
import scipy as sp


def mean_confidence_interval(a, k=1, confidence=0.99):
	n = len(a)/k
	m, se = np.mean(a), sp.stats.sem(a)
	h = se * sp.stats.t._ppf((1+confidence)/2, n-1)
	return m, m-h, m+h


# get the data of single log file and process the latencies
def processFile(path, f, newFile = False):
	print(path+f)
	data = np.loadtxt(path+f)
	(x,y)=  data.shape
	# 1. get arrival rate
	# arrival = data[1:, 1]/1000/1000
	arrival = data[floor(x/5):, 1]/1000/1000
	#histogram(arrival, "arrival interval distribution")
	mean_a = np.mean(arrival)
	var_a = np.var(arrival)
	# print("Mean Arrival interval is", mean_a, "variance is", var_a)
	# 2. get end-to-end latency distribution
	# latency = data[1:, 0]/1000/1000
	latency = data[floor(x/5):, 0]/1000/1000
	# print(f,latency)
	#histogram(latency, "end-to-end latency distribution")
	m, m_l, m_h = mean_confidence_interval(latency)
	mList = np.mean(data[floor(x/5):, 3:8]/1000/1000, 0)
	if newFile:
		temp = np.mean(data[floor(x/5):, 3:11]/1000/1000, 0)
		mList[0] = temp[0]+temp[1]
		mList[1] = temp[2] + temp[3]
		mList[2] = temp[4] + temp[5]
		mList[3:] = temp[6:]
	# print(f, m, m_l, m_h)
	mean_s = [m, m_l, m_h, np.percentile(latency,5), np.percentile(latency, 99)]+list(mList)
	var_s = np.var(latency)
	# print("Average Latency is", mean_s, "variance is", var_s, "98 percentile", np.percentile(latency, 95))
	return mean_a, var_a, mean_s, var_s


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



def plotStage(bList, mean_s):
	#plt.plot(bList, mean_s[:,4], "*-", label= "99 percentile")
	plt.plot(bList, mean_s[:, 5], "*-", label="stage 1")
	plt.plot(bList, mean_s[:, 6], "*-", label="stage 2")
	plt.plot(bList, mean_s[:, 7], "*-", label="stage 3")
	plt.plot(bList, mean_s[:, 8], "*-", label="stage 4")
	plt.plot(bList, mean_s[:, 9], "*-", label="stage 5")
	print("latency ", mean_s[:,0])
	print("stage 1 ", mean_s[:,5])
	print("stage 2 ", mean_s[:,6])
	print("stage 3 ", mean_s[:,7])
	print("stage 4 ", mean_s[:,8])
	print("stage 5 ", mean_s[:,9])
	print(mean_s[:, 9])

def showLatency(path, bList, fList, directory, label = "", showStage = True):
	mean_a, var_a, mean_s, var_s = readFileList(path + directory+"/", fList)

	plt.fill_between(bList, mean_s[:, 1], mean_s[:, 2], alpha=.5)
	
	plt.plot(bList, mean_s[:, 0], "*-", label=directory)
	if showStage:
		plotStage(bList, mean_s)

	plt.ylabel("Latency (ms)")
	plt.xlabel(label)
	plt.legend()
	plt.show()



def plotPoiRate(pubList, rate):
	directory = "poiRate_"+str(rate)
	sTest = "poi"
	batch = 1
	fList = []
	for pub in pubList:
		fList.append("latency_{:s}_{:d}_{:d}_{:d}".format(sTest, pub, batch, rate))
	showLatency("./", np.array(pubList)*rate/1000, fList, directory, label = "Message Rate (kmessages/s)")


def plotPoiBatch(batchList, pub):
	directory = "poiBatch_"+str(pub)
	sTest = "poi"
	rate = 100
	fList = []
	for batch in batchList:
		fList.append("latency_{:s}_{:d}_{:d}_{:d}".format(sTest, pub, batch, rate))
	showLatency("./", batchList, fList, directory, label = "Batch Size (messages)")



if __name__ == "__main__":
	rate = 100
	pubList = [10, 20, 50, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500]
	plotPoiRate(pubList, rate)
	
	rate = 500
	pubList = [10, 20, 50, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500]
	plotPoiRate(pubList, rate)
	
	pub = 100
	batchList = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	plotPoiBatch(batchList, pub)









