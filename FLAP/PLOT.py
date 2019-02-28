# -*- coding: utf-8 -*-
# author:nutty

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_style('ticks', {"xtick.direction": "in","ytick.direction": "in"})
sns.set_context("paper")
rc={'axes.labelsize': 24, 'legend.fontsize': 20.0, 'axes.titlesize': 24, 'font.family':'Arial'} #15 13 15がデフォルト
#rc={'axes.labelsize': 30, 'legend.fontsize': 20.0, 'axes.titlesize': 30, 'font.family':'Arial'} #15 13 15がデフォルト

plt.rcParams.update(**rc)


def to_dataframe(N, trial, data, column, file_name): #K数を変えながら試行したデータをデータフレームとして保存
    d = np.zeros((trial * N,3))
    for i in range(N):
        for j in range(trial):
            d[i * trial + j] = [i, data[i, j], N]
    d = pd.DataFrame(data=d , columns=['K', column, 'N'])
    d['K'] = d['K'].astype(int)
    d['N'] = d['N'].astype(int)
    d.to_csv(file_name + '.csv')

def to_gml(file_name, matrix, attr):
    with open(file_name + '.gml', mode='a') as f:
        s = 'Creator "nutty " \ngraph \n['
        f.write(s)
        for i in range(len(matrix)):
            node = '\n  node\n  [\n    id {0}\n    fitness {1}\n    basin {2}\n  ]'.format(i, attr[i, 0], int(attr[i, 1]) )
            f.writelines(node)
        
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i, j] > 0:
                    edge = '\n  edge\n  [\n    source {0}\n    target {1}\n    value {2}\n  ]'.format(i, j, matrix[i, j])
                    f.writelines(edge)
        f.writelines('\n]')
    f.close()


def plot_fit_weight(file_name):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots(figsize=(10, 20))
    #fig = sns.jointplot("fitness of local optima", "sum of weight into local optima", color=sns.xkcd_rgb["pale red"], data=d, s=10, marker='+')
    fig = sns.regplot("fitness of local optima", "sum of weight into local optima", color=sns.xkcd_rgb["pale red"], data=d, marker='+',  x_estimator=np.mean, logx=True, truncate=True)
    ax = fig.ax_joint
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.tick_params(labelsize=15)
    #fig.ax_marg_x.set_xscale('log')
    fig.ax_marg_y.set_yscale('log')
    sns.set_style('ticks', {"xtick.direction": "in","ytick.direction": "in"})
    plt.show()
    #plt.savefig(file_name + '_fw.png', bbox_inches="tight")
    #plt.savefig(file_name + '_fw.pdf', bbox_inches="tight")

def plot_fit_basin(file_name):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots(figsize=(10, 20))
    fig = sns.jointplot("fitness of local optima", "size of basin", data=d, color = sns.xkcd_rgb["pale red"], s=10, marker='+')
    ax = fig.ax_joint
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=36)
    sns.set_style('ticks', {"xtick.direction": "in","ytick.direction": "in"})
    #plt.show()
    plt.savefig(file_name + '_fb.png', bbox_inches="tight")
    plt.savefig(file_name + '_fb.pdf', bbox_inches="tight")


def plot_basin_weight(file_name):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots(figsize=(8, 8))
    #fig = sns.scatterplot(x="size of basin", y="sum of weight into local optima",hue = 'D', size='fitness of local optima', sizes=(1, 200),data=d , palette='Greens_d', alpha=0.7, style='D')
    fig = sns.scatterplot(x="size of basin", y="sum of weight into local optima", size='fitness of local optima', sizes=(10, 200),data=d , color=sns.xkcd_rgb["pale red"], alpha=0.7)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    #fig.ax_marg_x.set_xscale('log')
    #fig.ax_marg_y.set_yscale('log')
    ax.tick_params(labelsize=36)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='upper left', handles=handles[1:] , labels=labels[1:])
    #plt.show()
    #plt.savefig(file_name + '_bw.png', bbox_inches="tight")
    plt.savefig(file_name + '_bw.pdf', bbox_inches="tight")

def plot_scatter(file_name):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig = sns.scatterplot(x="fitness", y="fitness of neighbor",color=sns.xkcd_rgb["pale red"],data=d)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #fig.ax_marg_x.set_xscale('log')
    #fig.ax_marg_y.set_yscale('log')
    ax.tick_params(labelsize=30)
    #plt.show()
    plt.savefig(file_name + '.png', bbox_inches="tight")
    plt.savefig(file_name + '.pdf', bbox_inches="tight")

def plot_fitness_cloud(file_name):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots(figsize=(10, 20))
    fig = sns.jointplot("fitness", "fitness of neighbors", data=d, color=sns.xkcd_rgb["denim blue"], xlim=(0.3, 0.78), ylim=(0.3, 0.78), kind='kde')
    
    ax = fig.ax_joint
    
    #ax.set_xscale('log')
    
    #ax.set_yscale('log')
    ax.tick_params(labelsize=15)
    #fig.ax_marg_x.set_xscale('log')
    #fig.ax_marg_y.set_yscale('log')
    
    sns.set_style('ticks', {"xtick.direction": "in","ytick.direction": "in"})
    #plt.show()
    plt.savefig(file_name + '.png', bbox_inches="tight")
    #plt.savefig(file_name + '.png', bbox_inches="tight")
    plt.savefig(file_name + '.pdf', bbox_inches="tight")

def plot_weight_cloud(file_name):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots(figsize=(8, 8))
    #fig = sns.jointplot("hamming distance", "Existing", data=d, color=sns.xkcd_rgb["denim blue"], xlim=(0, 9))
    #fig = sns.scatterplot(x="hamming distance", y="Existing",color=sns.xkcd_rgb["denim blue"],data=d)
    fig = sns.stripplot(x="hamming distance", y="weight", hue ='network', palette='Set1', data=d, dodge=True, jitter=0.3,alpha=.15, zorder=1)

    #fig = sns.violinplot(x="hamming distance", y="weight", hue ='network', palette='Set1',split=True, inner="quart",data=d)

    #fig = sns.violinplot(x="hamming distance", y="weight", hue ='network', data=d, palette='Set1', jitter=True)
    #ax = fig.ax_joint
    #ax.set_xscale('log')
    
    ax.set_yscale('log')
    ax.tick_params(labelsize=18)
    #fig.ax_marg_x.set_xscale('log')
    #fig.ax_marg_y.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='upper right', handles=handles[0:], labels=labels[0:])
    sns.set_style('ticks', {"xtick.direction": "in","ytick.direction": "in"})
    #plt.show()
    plt.savefig(file_name + '.png', bbox_inches="tight")
    plt.savefig(file_name + '.png', bbox_inches="tight")
    plt.savefig(file_name + '.pdf', bbox_inches="tight")


def plot_joint(file_name):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots(figsize=(10, 20))
    fig = sns.jointplot("fitness of local optima", "sum of weight into local optima", data=d, color=sns.xkcd_rgb["denim blue"], s=10)
    
    ax = fig.ax_joint
    
    #ax.set_xscale('log')
    
    ax.set_yscale('log')
    ax.tick_params(labelsize=15)
    #fig.ax_marg_x.set_xscale('log')
    #fig.ax_marg_y.set_yscale('log')
    sns.set_style('ticks', {"xtick.direction": "in","ytick.direction": "in"})
    #plt.show()
    #plt.savefig(file_name + '.png', bbox_inches="tight")
    plt.savefig(file_name + '.pdf', bbox_inches="tight")

def plot_line(file_name, column_y):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots()
    #pal = sns.cubehelix_palette(5, rot=-.20, light=.7) #最初の引数は系列数
    #pal = sns.cubehelix_palette(3, rot=.50, light=.7) #最初の引数は系列数
    
    #fig = sns.lineplot(x="K", y=column_y, hue="N", style="N", markers=True, dashes=True,  palette='Set1', data=d, ax=ax, size=('N'), sizes=(2.0, .5),)
    #fig = sns.lineplot(x="K", y=column_y, hue="Algorithm", style="N", markers=True, dashes=False, palette='Reds_d', data=d, ax=ax)
    fig = sns.lineplot(x="K", y=column_y, hue="network", style="network", markers=False, dashes=True, palette='Set1',size=('N'), sizes=(2.0, .5), data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, hue="MUPB", palette=pal , data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, palette=pal, data=d, ax=ax)
    fig.tick_params(labelsize=24)
    plt.yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(loc='upper right', handles=handles[1:4]+handles[:], labels=labels[1:3]+labels[4:])
    ax.legend(loc='upper right', handles=handles[1:3], labels=labels[1:3])
    #plt.show()
    plt.savefig(file_name + '.png',  bbox_inches="tight")
    plt.savefig(file_name + '.pdf', bbox_inches="tight")

def plot_line2(file_name, column_y):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots()
    #pal = sns.cubehelix_palette(5, rot=-.20, light=.7) #最初の引数は系列数
    #pal = sns.cubehelix_palette(3, rot=.50, light=.7) #最初の引数は系列数
    #plt.yscale('log')
    #fig = sns.lineplot(x="K", y=column_y, hue="MUPB", style="MUPB", markers=True, dashes=False, palette='Reds_d', data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, hue="Algorithm", style="N", markers=True, dashes=False, palette='Reds_d', data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, hue="Network", style="Network", markers=True, dashes=False, palette='Set1', data=d, ax=ax)
    fig = sns.lineplot(x="function evaluations", y="best fitness", hue="Algorithm",units="trial", palette='Set1', estimator=None, lw=1,data=d)
    #fig = sns.lineplot(x="K", y=column_y, hue="MUPB", palette=pal , data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, palette=pal, data=d, ax=ax)
    fig.tick_params(labelsize=13)
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(loc='upper right', handles=handles[1:4]+handles[:], labels=labels[1:3]+labels[4:])
    ax.legend(loc='lower right', handles=handles[1:], labels=labels[1:])
    #plt.show()
    plt.savefig(file_name + '.pdf')

def plot_distribution(file_name, K, column):
    data = pd.read_csv(file_name + '.csv')
    data = np.array(data[column])
    sep = int(len(data)) + 2
    #sep = 400
    d = np.zeros((sep, 3))
    d[:, 2] = K
    c = 0
    d[0, 0] = 0
    d[0, 1] = 1
    for i in np.argsort(data):
        d[c+1, 0] = data[i]
        d[c+1, 1] = len(data[data >= d[c+1, 0]]) / len(data)
        c += 1
    #d[sep - 1, 0] = max(data) + 1
    d[sep - 1, 0] = max(data) + 0.01
    d[sep - 1, 1] = len(data[data >= d[sep - 1, 0]]) / len(data)

    d = pd.DataFrame(data=d , columns=[column, 'cumulative distribution', 'K'])
    d['K'] = 'K=' + str(K)
    d.to_csv(file_name + '_distribution_' + column + '.csv')

def plot_weight_distribution(file_name, K, column):
    data = np.load(file_name + '.npy')
    data = np.reshape(data, (len(data)**2), 1)
    #sep = int(len(data)) 

    sep = 100 
    d = np.zeros((sep, 3))
    c = 0
    d[:, 0] = np.linspace(0, 1, sep)
    for i in range(sep):
        d[c, 1] = len(data[data >= d[c, 0]]) / len(data)
        c += 1

    d = pd.DataFrame(data=d , columns=[column, 'cumulative distribution', 'K'])
    d['K'] = 'K=' + str(K)
    d.to_csv(file_name + '_distribution_' + column + '.csv')


def plot_line_distribution(file_name, column_y):
    d = pd.read_csv(file_name + '.csv')
    fig, ax = plt.subplots()
    #pal = sns.cubehelix_palette(5, rot=-.20, light=.7) #最初の引数は系列数
    #pal = sns.cubehelix_palette(3, rot=.50, light=.7) #最初の引数は系列数
    
    #fig = sns.lineplot(x="weight", y='cumulative distribution', hue="K", style="K", markers=False,palette='Reds_d', data=d, ax=ax)
    fig = sns.lineplot(x="W", y='p(W < Wij)', hue="network", style="network",size=('K'), sizes=(2.0, .5), markers=False,palette='Set1', data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, hue="Algorithm", style="N", markers=True, dashes=False, palette='Reds_d', data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, hue="Network", style="Network", markers=True, dashes=False, palette='Set1', data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, hue="MUPB", palette=pal , data=d, ax=ax)
    #fig = sns.lineplot(x="K", y=column_y, palette=pal, data=d, ax=ax)
    fig.tick_params(labelsize=18)
    plt.xscale('log')
    #plt.yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='upper right', handles=handles[1:3]+handles[4:9], labels=labels[1:3]+labels[4:9])
    #ax.legend(loc='lower left', handles=handles[1:], labels=labels[1:])
    #plt.show()
    plt.savefig(file_name + '.png', bbox_inches="tight")
    plt.savefig(file_name + '.pdf',  bbox_inches="tight")

