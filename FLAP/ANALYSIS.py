# -*- coding: utf-8 -*-
# author:nutty

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

import GEN_LANDSCAPE
#import ALGORITHMS
import PLOT
import ALGORITHM

def analysis(N, trial, problem, seed): #N*trialの数だけNKランドスケープを生成し，Kを変えながら局所的最適解の数と大域的最適解のbasinの解空間を占める割合を算出する
    N = N
    trial = trial
    num_lopts = np.zeros((N ,trial))
    gopt = np.zeros((N,trial))
    for K in tqdm(range(N)):
        landscape = GEN_LANDSCAPE.gen_landscape(N, K, trial, problem, seed) #calc_basinのコメントアウトを解除
        landscape, m = GEN_LANDSCAPE.existing(N, landscape)
        del m
        gc.collect()
        #gopt[K] = landscape[landscape[:,:,N+2] >0][:,N+1]/ np.sum(landscape[landscape[:,:,N+1] >0][:,N+1],axis=0)
        gopt[K] = landscape[landscape[:,:,N+2] >0][:,N+1]/ 2**N
        num_lopts[K] = len(landscape[:,:, N + 1]>0, axis=1)
    PLOT.to_dataframe(N=N, trial=trial, data=gopt.astype(np.float16), column='normalized grobal optimum basin', file_name='Nomalized_Gopt_hill_N'+str(N))
    PLOT.to_dataframe(N=N, trial=trial, data=num_lopts, column='number of local optima', file_name='Num_Lopt_hill_N'+str(N))

def gen_LON_p(N, K, problem, seed):
    #for K in tqdm(range(N)):
    landscape = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    landscape, basin_matrix = GEN_LANDSCAPE.proposal(N, landscape)
    
    lopts = landscape[landscape[:,N+1]>0, N+3].astype(int)
    attr = landscape[landscape[:,N+1]>0, N:N+2]

    lopts_network = np.zeros((len(lopts), len(lopts)))
    basin_matrix = basin_matrix[lopts]
    c = 0
    for i in range(len(lopts)):
        feeder = basin_matrix[i] * basin_matrix
        lopts_network[c] = np.sum(np.where(feeder == 0, feeder, 1) * basin_matrix, axis=1) / landscape[lopts[0], N + 1]
        c += 1
    lopts_network = lopts_network.astype(np.float16)

    del basin_matrix
    gc.collect()
    file_name = problem + '_p_N' + str(N) + 'K' + str(K) + 'seed' + str(seed) 
    np.save(file_name + '.npy', lopts_network)
    PLOT.to_gml(file_name, lopts_network, attr)
    analysis2(N, K, landscape, lopts_network, file_name)

def gen_LON_p2(N, K, problem, seed):
    #for K in tqdm(range(N)):
    landscape = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    landscape, basin_matrix = GEN_LANDSCAPE.proposal2(N, landscape)

    lopts = landscape[landscape[:,N+1]>0, N+3].astype(int)
    lopts_network = np.zeros((len(lopts), len(lopts)))
    attr = landscape[landscape[:,N+1]>0, N:N+2]
    basin_matrix = basin_matrix[lopts]

    c = 0
    for i in range(len(lopts)):
        feeder = basin_matrix[i] * basin_matrix
        lopts_network[c] = np.sum(np.where(feeder == 0, feeder, 1) * basin_matrix, axis=1) / landscape[lopts[0], N + 1]
        c += 1
    lopts_network = lopts_network.astype(np.float16)

    del basin_matrix
    gc.collect()
    file_name = problem + '_p2_N' + str(N) + 'K' + str(K) + 'seed' + str(seed) 
    np.save(file_name + '.npy', lopts_network)
    PLOT.to_gml(file_name, lopts_network, attr)
    analysis2(N, K, landscape, lopts_network, file_name)

def gen_LON_p3(N, K, D, problem, seed):
    #for K in tqdm(range(N)):
    landscape = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    landscape, basin_matrix = GEN_LANDSCAPE.proposal3(N, landscape, D)

    lopts = landscape[landscape[:,N+1]>0, N+3].astype(int)
    lopts_network = np.zeros((len(lopts), len(lopts)))
    attr = landscape[landscape[:,N+1]>0, N:N+2]
    basin_matrix = basin_matrix[lopts]

    c = 0
    for i in range(len(lopts)):
        feeder = basin_matrix[i] * basin_matrix
        lopts_network[c] = np.sum(np.where(feeder == 0, feeder, 1) * basin_matrix, axis=1) / landscape[lopts[0], N + 1]
        c += 1
    lopts_network = lopts_network.astype(np.float16)

    del basin_matrix
    gc.collect()
    file_name = problem + '_p3_N' + str(N) + 'K' + str(K) + 'D' + str(D) + 'seed' + str(seed) 
    np.save(file_name + '.npy', lopts_network)
    PLOT.to_gml(file_name, lopts_network, attr)
    analysis2(N, K, landscape, lopts_network, file_name)

def gen_LON_e(N, K, problem, seed):
    #for K in tqdm(range(N)):
    landscape = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    landscape, basin_matrix = GEN_LANDSCAPE.existing(N, landscape)
    lopts = landscape[landscape[:,N+1]>0, N+3].astype(int) 
    lopts_network = np.zeros((len(lopts), len(lopts)))
    attr = landscape[landscape[:,N+1]>0, N:N+2]

    basin_matrix = basin_matrix[lopts]
    c1 = 0
    for i in range(len(lopts)):
        c2 = 0
        index = np.where(basin_matrix[i]>0)
        for j in range(len(lopts)):
            index2 = np.where(basin_matrix[j]>0)
            for k in index2[0]:
                a = np.sum(abs(landscape[index[0], :N] - landscape[k, :N]), axis=1)
                a = len(a[a==1])
                lopts_network[c1, c2] += a / N 
            lopts_network[c1, c2] /= int(landscape[lopts[i], N+1])
            c2 += 1
        c1 += 1
    lopts_network = lopts_network.astype(np.float16)
    del basin_matrix
    gc.collect()

    file_name = problem + '_e_N' + str(N) + 'K' + str(K) + 'seed' + str(seed) 
    np.save(file_name + '.npy', lopts_network)
    PLOT.to_gml(file_name, lopts_network, attr)
    analysis2(N, K, landscape, lopts_network, file_name)

def gen_LON_ee(N, K, D, problem, seed):
    landscape = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    landscape, basin_matrix = GEN_LANDSCAPE.existing(N, landscape)
    lopts = landscape[landscape[:,N+1]>0, N+3].astype(int)
    lopts_network = np.zeros((len(lopts), len(lopts)))
    attr = landscape[landscape[:,N+1]>0, N:N+2]

    c1 = 0
    for i in lopts:
        c2 = 0
        area = landscape[np.sum(abs(landscape[:,:N] - landscape[i,:N]),axis=1)<=D, N+3].astype(int)
        lopts_basin = basin_matrix[lopts]
        lopts_basin = lopts_basin[:, area]
        lopts_network[c1] = np.sum(lopts_basin, axis=1)/len(area)
        c1 += 1
    lopts_network = lopts_network.astype(np.float16)
    del basin_matrix
    gc.collect()

    file_name = problem + '_ee_N' + str(N) + 'K' + str(K) + 'D' + str(D) + 'seed' + str(seed) 
    np.save(file_name + '.npy', lopts_network)
    PLOT.to_gml(file_name, lopts_network, attr)
    analysis2(N, K, landscape, lopts_network, file_name)

def gen_LON_eee(N, K, D, problem, seed):
    #for K in tqdm(range(N)):
    landscape = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    landscape, basin_matrix = GEN_LANDSCAPE.proposal2(N, landscape)
    lopts = landscape[landscape[:,N+1]>0, N+3].astype(int)
    lopts_network = np.zeros((len(lopts), len(lopts)))
    attr = landscape[landscape[:,N+1]>0, N:N+2]

    basin_matrix = basin_matrix[lopts]
    c = 0
    for i in range(len(lopts)):
        area = landscape[np.sum(abs(landscape[:,:N] - landscape[lopts[i],:N]),axis=1)<=D, N+3].astype(int)
        feeder = basin_matrix[i, area] * basin_matrix[:, area]
        feeder = np.where(feeder == 0, feeder, 1) * basin_matrix[:, area]
        lopts_network[c] = np.sum(feeder, axis=1) / len(area)
        c += 1
    lopts_network = lopts_network.astype(np.float16)

    del basin_matrix
    gc.collect()
    file_name = problem + '_eee_N' + str(N) + 'K' + str(K) + 'seed' + str(seed) 
    np.save(file_name + '.npy', lopts_network)
    PLOT.to_gml(file_name, lopts_network, attr)
    analysis2(N, K, landscape, lopts_network, file_name)


def analysis2(N, K, landscape, lopts_network, file_name):
    N = N
    K = K
    LON = lopts_network
    landscape = landscape

    lopt_fit = landscape[landscape[:,N+1]>0, N]
    d = np.zeros((3, len(lopt_fit)))
    for i in range (len(lopt_fit)):
        d[2,i] = landscape[landscape[:,N+1] > 0, N+1][i]
        d[1,i] = np.sum(LON, axis = 0)[i]
        d[0,i] = lopt_fit[i]
    d = pd.DataFrame(data=d.T, columns=['fitness of local optima', 'sum of weight into local optima', 'size of basin'])

    d.to_csv(file_name + '.csv')


def fitness_cloud(N, K, problem, seed):
    #for K in tqdm(range(N)):
    landscape = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    cloud = np.zeros((2**N * N, 2))
    evolability = np.zeros(2**N)
    for i in range (2 **N):
        neighbor =landscape[np.sum(abs(landscape[:,:N] - landscape[i,:N]),axis=1)==1, N]
        for j in range(N):
            cloud[i * N + j, 0] = landscape[i, N]
            cloud[i * N + j, 1] = neighbor[j]
        
        neighbor = neighbor - landscape[i, N]
        evolability[i] = len(neighbor[neighbor > 0])

    d = pd.DataFrame(data=cloud, columns=['fitness', 'fitness of neighbors'])
    d2 = pd.DataFrame(data=evolability, columns=['Number of improving neighbors'])
    d.to_csv(problem + '_N' + str(N) + 'K' + str(K) + '_clouds.csv')
    d2.to_csv(problem + '_N' + str(N) + 'K' + str(K) + '_evolability.csv')

def dist_weight(N, K, problem, seed):
    lopt = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    lopt = lopt[lopt[:, N+1]>0, :N]
    data = np.zeros((len(lopt)**2, 3))
    e = np.load(problem + '_e_N' + str(N) + 'K' + str(K) + '.npy')
    p = np.load(problem + '_p_N' + str(N) + 'K' + str(K) + '.npy')
    c = 0
    for i in range(len(lopt)):
        for j in range(len(lopt)):
            data[len(lopt) * i + j, 0] = int(np.sum(abs(lopt[j] - lopt[i])))
            data[len(lopt) * i + j, 1] = e[i, j]
            data[len(lopt) * i + j, 2] = p[i, j]
        c += 1
    data = pd.DataFrame(data=data , columns=['hamming distance', 'Existing', 'Proposal'])
    data.to_csv('N' + str(N) + 'K' + str(K) + problem + '_dist.csv')

def fitness_cloud2(N, K, problem, seed):
   landscape = GEN_LANDSCAPE.gen_landscape(N, 6, 1, problem, seed)[0]
   cloud = np.zeros((2**N * N, 2))
   evolability = np.zeros(2**N)
   for i in range (2 **N):
       neighbor =landscape[np.sum(abs(landscape[:,:N] - landscape[i,:N]),axis=1)==1, N]
       for j in range(N):
           cloud[i * N + j, 0] = landscape[i, N]
           cloud[i * N + j, 1] = neighbor[j]
   return np.corrcoef(cloud[:, 0], cloud[:, 1])

def analysis_GA(N, K, problem, seed, trial, GEN, MU, CXPB, MUPB):
    landscape = GEN_LANDSCAPE.gen_landscape(N, K, 1, problem, seed)[0]
    for i in range(trial):
        pop, lopt_list_GA = ALGORITHM.run_GA(seed=i, N=N, landscape=landscape, NGEN=GEN, MU=MU, CXPB=CXPB, MUPB=MUPB)
        lopt_list_GA = (list(dict.fromkeys(lopt_list_GA)))
        print(lopt_list_GA)