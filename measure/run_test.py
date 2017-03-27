#!/usr/bin/python
import os
from time import sleep

pubServer = "ssh root@cqos4.seas.wustl.edu"
subServer = "ssh root@cqos5.seas.wustl.edu"
nsqdServer = "ssh root@cqos3.seas.wustl.edu"
lookupdServer = "ssh root@cqos6.seas.wustl.edu"

directory = "/home/RTM/run-test/"
nsqdStart = "sh daemon_create.sh"
lookupdStart = "sh lookupd_create.sh"


def restartService():
	os.system(" ".join([subServer, "pkill", "subscriber"]))
	os.system(" ".join([pubServer, "pkill", "publisher"]))
	sleep(1)
	os.system(" ".join([nsqdServer, "pkill", "nsqd"]))
	sleep(1)
	os.system(" ".join([lookupdServer, "pkill", "nsqlookupd"]))
	sleep(1)
	os.system("{:s} \"cd {:s}; {:s}  > /dev/null 2>&1\"".format(lookupdServer, directory, lookupdStart))
	sleep(1)
	os.system("{:s} \"cd {:s}; {:s}  > /dev/null 2>&1\"".format(nsqdServer, directory, nsqdStart))
	sleep(1)
	
def TestBurst(pub, burst = 1, batch = 1, rate = 1000, newDir = 'testBurst/', conn = 2):
	test = "Burst"
	sTest = "bst"
	script = "sh burst_base.sh"
	topic = "0"+str(pub)
	interval = 1000000//rate//batch

	#os.system(" ".join([subServer, "mkdir", newDir]))
	file = "latency"+topic
	newFile = newDir + "latency_{:s}_{:d}_{:d}_{:d}_{:d}".format(sTest, pub, burst, batch, rate)
	print("Exp for {:d} pubs burst {:d} with batch {:d} and rate {:d} in {:s}".format(pub, burst, batch, rate, test))
	pubCmd = "{:s} \"cd {:s}; {:s} 0 0 {:d} {:s} {:d} {:d} {:d}  > /dev/null 2>&1\"".format(pubServer, directory, script, pub, topic, burst, batch, interval, batch)
	subCmd = "{:s} \"cd {:s}; {:s} 0 {:s} {:d}  > /dev/null 2>&1\"".format(subServer, directory, script, topic, conn)
	sleep(1)
	os.system(pubCmd)
	sleep(1)
	os.system(subCmd)

	waitTime = 50000//rate//pub + 5
	while waitTime >= 10:
		sleep(2)
		os.system("{:s} \"ps -C nsqd2 -o %cpu,%mem\"".format(nsqdServer))
		sleep(waitTime-2)
		waitTime -= 10

	sleep(waitTime)

	# wait the subscriber to finish
	sleepTime = 0
	while os.system(" ".join([subServer, "pgrep", "subscriber"]))== 0:
		if sleepTime < 60:
			sleep(10)
			sleepTime += 10
		else:
			restartService()
			return -1
		
	# kill the publisher 
	os.system(" ".join([pubServer, "pkill", "publisher"]))
	os.system(" ".join([subServer, "mv", directory+file, directory+newFile]))
	sleep(5)
	return 0


def TestPoisson(pub, batch = 1, rate = 1000, newDir = "testPoisson/", conn = 2):
	test = "Poisson"
	sTest = "poi"
	script = "sh bpi_base.sh"
	topic = "x"+str(pub)
	interval = 1000000//rate//batch

	#os.system(" ".join([subServer, "mkdir", newDir]))
	file = "latency"+topic
	newFile = newDir + "latency_{:s}_{:d}_{:d}_{:d}".format(sTest, pub, batch, rate)
	print("Exp for {:d} pubs with batch {:d} and rate {:d} in {:s}".format(pub, batch, rate, test))
	pubCmd = "{:s} \"cd {:s}; {:s} 0 {:d} {:s} {:d} {:d}  > /dev/null 2>&1\"".format(pubServer, directory, script, pub, topic, interval, batch)
	subCmd = "{:s} \"cd {:s}; {:s} 0 {:s} {:d}  > /dev/null 2>&1\"".format(subServer, directory, script, topic, conn)
	sleep(1)
	os.system(pubCmd)
	sleep(1)
	os.system(subCmd)

	waitTime = 5000//rate + 5
	while waitTime >= 10:
		sleep(4)
		os.system("{:s} \"ps -C nsqd2 -o %cpu,%mem\"".format(nsqdServer))
		sleep(6)
		waitTime -= 10

	sleep(waitTime)

	# wait the subscriber to finish
	sleepTime = 0
	while os.system(" ".join([subServer, "pgrep", "subscriber"]))== 0:
		if sleepTime < 30:
			sleep(5)
			sleepTime += 5
		else:
			restartService()
			return -1
	
	# kill the publisher 
	os.system(" ".join([pubServer, "pkill", "publisher"]))
	os.system(" ".join([subServer, "mv", directory+file, directory+newFile]))
	sleep(5)
	return 0


# increase the rate, fix burst
def testPoissonRate(pubList, rate = 100, newDir = "poiRate/", conn = 2):
	batch = 1
	os.system("{:s} \"cd {:s}; mkdir -p {:s}\"".format(subServer, directory, newDir))
	for pub in pubList:
			if TestPoisson(pub, batch, rate, newDir, conn == conn) != 0:
				print("Error Running this instance")


def testPoissonBatch(BatchList, pub = 10, newDir = "poiBatch/", conn = 2):
	rate = 100
	os.system("{:s} \"cd {:s}; mkdir -p {:s}\"".format(subServer, directory, newDir))
	for batch in batchList:
			if TestPoisson(pub, batch, rate, newDir, conn = conn) != 0:
				print("Error Running this instance")

if __name__ == "__main__":
	restartService()

	rate = 200
	pubList = [10, 20, 50, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500]
	testPoissonRate(pubList, rate, "poiRate_"+str(rate)+"_2/", conn = 2)

	rate = 200
	pubList = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	testPoissonRate(pubList, rate, "poiRate_"+str(rate)+"_2/", conn = 2)

	rate = 500
	pubList = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	testPoissonRate(pubList, rate, "poiRate_"+str(rate)+"_2/", conn = 2)


	# pub = 100
	# batchList = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	# testPoissonBatch(batchList, pub, "poiBatch_"+str(pub)+"/")





