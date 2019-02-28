# -*- coding: utf-8 -*-
# author:nutty

import random
import numpy as np

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from tqdm import tqdm

toolbox = base.Toolbox()

def evalNK(individual, N, landscape):
    gene = int(''.join(map(str, individual)),2)
    fit = np.array(landscape[gene, N])
    return fit, fit

def init_pop(N, MU):
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=MU)
    #toolbox.register("select", tools.selTournament, tournsize=4)
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop_ini = toolbox.select(pop, len(pop))
    return pop_ini

def optima_checker2(individual, landscape, lopt_list, N, min):
    gene = int(''.join(map(str, individual)),2)
    if landscape[gene, N+1] > min:
        lopt_list.append(landscape[gene, N+1])
        min = landscape[gene, N+1]
        #if landscape[gene, N+1] > 0:
            #print('success')
    else:
        lopt_list.append(min)
    #print(lopt_list)
    return lopt_list, min

def optima_checker(individual, landscape, lopt_list, N):
    gene = int(''.join(map(str, individual)),2)
    if landscape[gene, N+1] > 0:
        lopts = landscape[landscape[:,N+1]>0, N+3]
        c = 0
        for lopt in lopts:
            if lopt == landscape[gene, N+3]:
                lopt_list.append(c)
            c += 1
    #print(lopt_list)
    return lopt_list

def GA(pop, MU, CXPB, landscape, lopt_list, N):
    offspring = toolbox.select(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in offspring]
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= CXPB:
            toolbox.mate(ind1, ind2)
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
            lopt_list = optima_checker(ind, landscape, lopt_list, N)
            ind.fitness.values = fit
    return pop, lopt_list 

def run_GA(seed, N, landscape, NGEN, MU, CXPB, MUPB):
    lopt_list = []
    random.seed(seed)
    toolbox.register("evaluate", evalNK, N=N, landscape=landscape)
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox.register("mate", tools.cxUniform, indpb=0.05)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUPB)
    toolbox.register("select", tools.selRoulette)
    #toolbox.register("select", tools.selTournament, tournsize=4)
    #toolbox.register("select", tools.selNSGA2)
    pop = init_pop(N, MU)
    for gen in range(1, NGEN):
        pop, lopt_list = GA(pop, MU, CXPB, landscape, lopt_list, N)
    return pop, lopt_list
