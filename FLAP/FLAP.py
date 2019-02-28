# -*- coding: utf-8 -*-
# author:nutty

#FLAPはnuttyが作り出した適応度地形解析のフレームワークです．目的関数を設定すれば，解析が行われます．

import numpy as np
import pandas as pd
from tqdm import tqdm

import GEN_LANDSCAPE
#import ALGORITHMS
import PLOT
import ANALYSIS
import ALGORITHM

if __name__ == "__main__":

    N = 12
    K = 5
    problem = 'NK'
    seed = 0

    #ANALYSIS.gen_LON_e(N, K, problem, seed)
    #ANALYSIS.gen_LON_p3(N, K, 2, problem, seed)

    #ANALYSIS.gen_LON_p3(N, K, 3, problem, seed)

    #ANALYSIS.gen_LON_p3(N, K, 5, problem, seed)


    ANALYSIS.analysis_GA(N=N, K=K, problem=problem, seed=seed, trial=5, GEN=100, MU=4, CXPB=1.0, MUPB=0.3)

    #ANALYSIS.gen_LON_eee(15, 2, 1, 'NK', 0)
    #for i in range(2):
        #seed = i
        #ANALYSIS.gen_LON_ee(8, 2, 1, 'NK', seed)
        #ANALYSIS.gen_LON_eee(8, 2, 1, 'NK', seed)

        #ANALYSIS.gen_LON_ee(8, 5, 1, 'NK', seed)
        #ANALYSIS.gen_LON_eee(8, 5, 1, 'NK', seed)

       
        #ANALYSIS.gen_LON_ee(15, 11, 1, 'NK', seed)
        #ANALYSIS.gen_LON_eee(15, 11, 1, 'NK', seed)

        #ANALYSIS.gen_LON_ee(15, 14, 1, 'NK', seed)
        #ANALYSIS.gen_LON_eee(15, 14, 1, 'NK', seed)


    #PLOT.plot_basin_weight('LON_ee_N15K5NK')
    #PLOT.plot_basin_weight('LON_ee_N15K8NK')
    #PLOT.plot_basin_weight('NK_e_N15K5')
    #PLOT.plot_basin_weight('NK_e_N15K8')

    #PLOT.plot_fitness_cloud('NK_N12K0_clouds')
    #analysis(N=N, trial=trial
    #ANALYSIS.gen_LON_e(12, 11, 'NK')
    #ANALYSIS.gen_LON_p(12, 11, 'NK')
    #ANALYSIS.gen_LON_ee(N, K, 1, 'NK')
    #trial = 12
    #analysis(N=N, trial=trial)
    #N = 9
    #trial = 3
    #analysis(N=N, trial=trial)
    #analysis2(N=N)
    #analysis3(N=N)
    #benchmark2(N=N,trial=trial)
    #analysis4(N, 2)
    #PLOT.plot_fitness_cloud('N12K2clouds')
    #PLOT.plot_basin_weight('NK_p_N15K8')
    #PLOT.plot_joint('weight_into_K5_hill')
    #PLOT.plot_fit_basin('NK_p_N15K8')
    #PLOT.plot_line('RS_N15_2', 'rate of local optima searched')
    #PLOT.plot_line("Norm_Gopt", "relative grobal optimum's basin size")
    #PLOT.plot_line2("trial", "best fitness")
    #PLOT.plot_line("Num_Lopt", "number of local optima")

    #PLOT.plot_line("distance_N15R", "sum of transition probability")
    #PLOT.plot_line("GA_distance_N12", "distance of adaptive walk")
    #PLOT.plot_weight_distribution('LON_p_N15K2NK', 2, 'weight')
    #PLOT.plot_weight_distribution('LON_p_N15K5NK', 5, 'weight')
    #PLOT.plot_weight_distribution('LON_p_N15K8NK', 8, 'weight')
    #PLOT.plot_weight_distribution('LON_p_N15K11NK', 11, 'weight')
    #d1 = pd.read_csv('LON_p_N15K2NK_distribution_weight.csv')
    #d2 = pd.read_csv('LON_p_N15K5NK_distribution_weight.csv')
    #d3 = pd.read_csv('LON_p_N15K8NK_distribution_weight.csv')
    #d4 = pd.read_csv('LON_p_N15K11NK_distribution_weight.csv')
    #d = pd.concat([d1, d2, d3, d4])
    #d.to_csv('NK_p_N15_distribution_weight.csv')

    #ANALYSIS.dist_weight(12, 11, 'NK')
    #PLOT.plot_weight_cloud('N12K11NK_dist')
    #PLOT.plot_line_distribution('NK_p_N15_distribution_weight', 'p(W < Wij)')