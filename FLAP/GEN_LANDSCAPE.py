# -*- coding: utf-8 -*-
# author:nutty

import numpy as np
import random
import itertools
import gc

def gen_items(N):
    items = np.zeros((N, 2))
    for i in range(N):
        items[i] = (random.randint(1, 10), random.uniform(0, 100))
    return items

def gen_matrix(N, K): #エピスタシスを決定するマトリックスを生成
    matrix = np.identity(N, dtype=int)
    for i in range(N):
        while np.sum(matrix[i]) < K + 1:
            rnd = random.randrange(0, N)
            matrix[i,rnd] = 1
    return matrix

def knapsack(N, comb1, items, max): #全個体の適応度を算出
    weight = np.sum(comb1 * items[:,0])
    value = np.sum(comb1 * items[:,1])
    if weight > max:
        return 0             # Ensure overweighted bags are dominated
    return value

def NK(N, landscape, matrix, Comb1, key): #全個体の適応度を算出
    vector = np.zeros(N)
    for i in np.arange(N):
        vector[i] = landscape[np.sum(Comb1 * matrix[i] * key), i]
    return(vector)

def calc_knapsack(N, key, items, max): #全個体の遺伝子，適応度，局所的最適解であるか，大域的最適解であるか，インデックスを格納
    value = np.zeros((2**N, N+4))  
    c1 = 0
    index = 0
    for c2 in itertools.product(range(2), repeat=N):
        Comb1 = np.array(c2)  
        fit = knapsack(N, Comb1, items, max)
        value[c1, :N] = Comb1 #0からN-1番目までは個体の遺伝子，N=3なら000, 001, 010, ..., 111を格納
        value[c1, N] = int(fit) #N番目にはその個体の適応度を格納

        value[c1, N+3] = index #N+3番目にはインデックスを格納
        index += 1
        c1 = c1 + 1
    for c3 in np.arange(2**N): 
        peak = 1 
        for c4 in np.arange(N):  
            new_comb = value[c3, :N].copy().astype(int)
            new_comb[c4] = abs(new_comb[c4] - 1)
            if ((value[c3, N] <= value[np.sum(new_comb * key), N])):
                peak = 0  
        value[c3, N+1] = peak #それが局所的最適解であるならN+1番目は1を格納
    max = np.argmax(value[:, N])
    value[max, N+2] = 1 #それが大域的最適解であるならN+2番目は1を格納
    return(value)

def calc_NK(N, key, landscape, matrix): #全個体の遺伝子，適応度，局所的最適解であるか，大域的最適解であるか，インデックスを格納
    value = np.zeros((2**N, N+4))  
    c1 = 0
    index = 0
    for c2 in itertools.product(range(2), repeat=N):
        Comb1 = np.array(c2)  
        fit = NK(N, landscape, matrix, Comb1, key)

        value[c1, :N] = Comb1 #0からN-1番目までは個体の遺伝子，N=3なら000, 001, 010, ..., 111を格納
        value[c1, N] = np.mean(fit).astype(np.float16) #N番目にはその個体の適応度を格納
        value[c1, N+3] = index #N+3番目にはインデックスを格納
        index += 1
        c1 = c1 + 1
    for c3 in np.arange(2**N): 
        peak = 1 
        for c4 in np.arange(N):  
            new_comb = value[c3, :N].copy().astype(int)
            new_comb[c4] = abs(new_comb[c4] - 1)
            if ((value[c3, N] < value[np.sum(new_comb * key), N])):
                peak = 0  
        value[c3, N+1] = peak #それが局所的最適解であるならN+1番目は1を格納
    max = np.argmax(value[:, N])
    value[max, N+2] = 1 #それが大域的最適解であるならN+2番目は1を格納
    return(value)

def proposal(N, data): #ある局所解に確率的探索手法で遷移し得る解の個数を算出
    basin = np.identity(2**N)
    for i in np.argsort(data[:,N], axis=0):
        neighbor = data[np.sum(abs(data[:,:N] - data[i,:N]),axis=1)==1]
        foots = neighbor[neighbor[:,N]- data[i,N]>0]
        if len(foots) > 0:
            basin[i] /= len(foots)
            for foot in foots[:,N+3]:
                basin[int(foot)] += basin[i]
    for i in data[data[:, N+1]==1][:,N + 3]:
        data[int(i),N+1] = np.sum(basin[int(i)] > 0)  #局所的最適解の部分を解の個数に更新

    del neighbor
    del foots
    gc.collect()

    return data, basin

def proposal2(N, data): #ある局所解に確率的探索手法で遷移し得る解の個数を算出
    basin = np.identity(2**N)
    for i in np.argsort(data[:,N], axis=0):
        neighbor = data[np.sum(abs(data[:,:N] - data[i,:N]),axis=1)==1]
        foots = neighbor[neighbor[:,N]- data[i,N]>0]
        if len(foots) > 0:
            roulette = foots[:, N] / np.sum(foots[:, N])
            c = 0
            for foot in foots[:,N+3]:
                basin[int(foot)] += basin[i] * roulette[c]
                c += 1
    for i in data[data[:, N+1]==1][:,N + 3]:
        data[int(i),N+1] = np.sum(basin[int(i)] > 0)  #局所的最適解の部分を解の個数に更新

    del neighbor
    del foots
    gc.collect()

    return data, basin

def proposal3(N, data, D): #ある局所解に確率的探索手法で遷移し得る解の個数を算出
    basin = np.identity(2**N)
    for i in np.argsort(data[:,N], axis=0):
        neighbor = data[np.sum(abs(data[:,:N] - data[i,:N]),axis=1)==1]
        foots = neighbor[neighbor[:,N]- data[i,N]>0]
        if len(foots) > 0:
            a = np.argsort(foots[:, N])
            a = a[::-1]
            foots = foots[a[:D]]
            roulette = foots[:, N] / np.sum(foots[:, N])
            c = 0
            for foot in foots[:,N+3]:
                basin[int(foot)] += basin[i] * roulette[c]
                c += 1
    for i in data[data[:, N+1]==1][:,N + 3]:
        data[int(i),N+1] = np.sum(basin[int(i)] > 0)  #局所的最適解の部分を解の個数に更新

    del neighbor
    del foots
    gc.collect()

    return data, basin

def existing(N, data): #ある局所解に山登り法で遷移し得る解の個数を算出
    basin = np.identity(2**N)
    for i in np.argsort(data[:,N], axis=0):
        neighbor = data[np.sum(abs(data[:,:N] - data[i,:N]),axis=1)==1]
        foots = neighbor[neighbor[:,N]- data[i,N]>0]
        if len(foots) > 0:
            basin[int(foots[np.argmax(foots[:,N]), N+3])] += basin[i]
     
    for i in data[data[:, N+1]==1][:,N + 3]:
        data[int(i),N+1] = np.sum(basin[int(i)] > 0) #局所的最適解の部分を解の個数に更新
    return data, basin

def gen_landscape(N, K, trial, problem, seed): #指定されたNとKの値に基づくランドスケープをtrialの数だけ生成
    N = N
    trial = trial
    key = np.power(2, np.arange(N - 1, -1, -1))
    data = np.zeros((trial, 2**N, N+4))
    random.seed(seed)
    np.random.seed(seed)
    for i in range(trial):
        if problem == 'knapsack':
            max = 10 * (K + 1) 
            items = gen_items(N)
            data[i] = calc_knapsack(N, key, items, max)
            del items
            gc.collect()
        elif problem == 'NK':
            matrix = gen_matrix(N, K)
            landscape = np.random.rand(2**N, N).astype(np.float16)
            data[i] = calc_NK(N, key, landscape, matrix)
            del matrix
            del landscape
            gc.collect()
    return data

