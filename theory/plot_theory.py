#!/usr/bin/python3
import numpy as np
from theory import batchPerTBDelay, PoiTBDelay, batchPoiTBDelay, batchPerSerDelay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def token_bstper():
    pRate = 1000.0
    rList = np.array([1 + 0.05*(i+1) for i in range(20)])*pRate
    k = 20
    bList = np.array([i+1 for i in range(20)])
    tb = []

    for b in bList:
        tb.append([])
        for r in rList:
            rho = pRate*1.0/r
            tb[-1].append(batchPerTBDelay(pRate, k, r, b))
    X = np.array(bList)
    Y = np.array(rList)/1000.0
    Y, X = np.meshgrid(Y,X)
    Z = np.array(tb)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    # plt.xticks(np.arange(len(kList)), [str(kList[i])+'|'+str(bList[i]) for i in range(len(kList))])
    ax.set_xlabel('bucket size')
    ax.set_ylabel('bucket rate (k)')
    ax.set_zlabel('access latency')
    ax.view_init(30, 30)
    plt.show()
    fig.savefig("theory_token_bstper.png")
    plt.close(fig)
    print(list(Z))


def token_bstper2():
    pRate = 1000.0
    rList = np.array([1 + 0.05*(i+1) for i in range(20)])*pRate
    k = 20
    b = 10
    bList = np.array([i+1 for i in range(20)])
    tb = []

    for b in bList:
        tb.append([])
        for r in rList:
            tb[-1].append(batchPerTBDelay(pRate, k, r, b))
    X = np.array(bList)
    Y = np.array(rList)
    Y, X = np.meshgrid(Y, X)
    Z = np.array(tb)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('bucket size (msg)')
    ax.set_ylabel('bucket rate (msg/s)')
    ax.set_zlabel('access latency (s)')
    ax.view_init(30, 30)
    plt.show()
    fig.savefig("theory_token_bstper.png")
    plt.close(fig)
    print(list(Z))

def token_exp():
    pRate = 1000.0
    rList = np.array([1 + 0.05*(i+1) for i in range(20)])*pRate
    bList = [i+1 for i in range(20)]
    tb = []
    for b in bList:
        tb.append([])
        for r in rList:
            rho = pRate*1.0/r
            tb[-1].append(PoiTBDelay(pRate, r, b))
    X = np.array(bList)
    Y = np.array(rList)/1000.0
    Y, X = np.meshgrid(Y,X)
    Z = np.array(tb)
    print(str(Z.tolist()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z)
    ax.set_xlabel('bucket size')
    ax.set_ylabel('bucket rate (k)')
    ax.set_zlabel('access latency')
    ax.view_init(30, 30)
    plt.show()
    fig.savefig("theory_token_exp.png")
    plt.close(fig)

def token_bstexp():
    pRate = 1000.0
    k = 20
    rList = np.array([1 + 0.05*(i+1) for i in range(20)])*pRate
    bList = [5*(i+1) for i in range(20)]
    tb = []
    for b in bList:
        tb.append([])
        for r in rList:
            rho = pRate*1.0/r
            tb[-1].append(batchPoiTBDelay(pRate,k, r, b))
    X = np.array(bList)
    Y = np.array(rList)/1000.0
    Y, X = np.meshgrid(Y,X)
    Z = np.array(tb)
    print(str(Z.tolist()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z)
    ax.set_xlabel('bucket size')
    ax.set_ylabel('bucket rate (k)')
    ax.set_zlabel('access latency')
    ax.view_init(30, 30)
    plt.show()
    fig.savefig("theory_token_bstexp10.png")
    plt.close(fig)


def server_bstper():
    pRate = 1000.0
    mu = 1000.0
    rList = np.array([1 - 0.05*(i+1) for i in range(10)])*mu
    k = 20
    bList = np.array([i+1 for i in range(40)])
    tb = []

    for b in bList:
        tb.append([])
        for r in rList:
            tb[-1].append(batchPerSerDelay(pRate, k, r, b, pRate))
    X = np.array(bList)
    Y = np.array(rList)/1000.0
    Y, X = np.meshgrid(Y, X)
    Z = np.array(tb)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    # plt.xticks(np.arange(len(kList)), [str(kList[i])+'|'+str(bList[i])
    # for i in range(len(kList))])
    ax.set_xlabel('bucket size')
    ax.set_ylabel('bucket rate (k)')
    ax.set_zlabel('access latency')
    ax.view_init(30, 30)
    plt.show()
    fig.savefig("theory_token_bstper.png")
    plt.close(fig)
    print(list(Z))


server_bstper()
