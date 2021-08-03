""" Module for pots processing the data from csv files.
Returns three kinf of images:
    1. Hot map for each id
    2. Trajector for each id.
    3. Mean distance between subjects
All these images are going to be save in a folder.
"""
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def heat_map(identifier,x,y,path_save):
    """Returns the heat map of objects along the  frame.
    """
    #Filter Bx,By in all frame based on Identifier 
    H, xedges, yedges = np.histogram2d(y, x, bins=(10, 10))
    X, Y = np.meshgrid(xedges, yedges)
    plt.figure()
    plt.pcolormesh(X, Y, H)
    plt.colorbar()
    plt.xlabel("x meters")
    plt.ylabel("y meters")
    plt.title(f'Heat map subject {identifier}')
    plt.savefig(f'{path_save}/heat_map_subject_{identifier}.png')
    plt.close('all')


def trajectory(identifier,x,y,path_save):
    """ Returns the trajectory of Identifier"""
    plt.figure()
    plt.plot(x, y, '.-')
    plt.plot(x[0], y[0], 'ro')
    plt.plot(x[-1], y[-1], 'go')
    plt.grid()
    plt.xlabel("x meters")
    plt.ylabel("y meters")
    plt.legend(['Trayectory','Initial Point','Final Point'])
    plt.title(f'Trajectory subject {identifier}')
    plt.savefig(f'{path_save}/trajectory_subject_{identifier}.png')
    plt.close('all')

def mean_dist(DataFrame,path_save):
    """Returns mean distance"""
    sujetos_ids= DataFrame['id'].unique()
    ind = {j:{i:[] for i in sujetos_ids} for j in sujetos_ids}
    frames_appears = np.array(DataFrame['frame'].unique())
    for frame in frames_appears:
        x = np.array(DataFrame[['Bx','By']][DataFrame['frame']==frame])
        ids = np.array(DataFrame['id'][DataFrame['frame']==frame])
        dist = distance_matrix(x,x)
        for i,id1 in enumerate(ids):
            for j,id2 in enumerate(ids):
                ind[id1][id2].append(dist[i,j])
    for i,sujeto in enumerate(DataFrame['id'].unique()):
        dis_sujeto = []
        for id in sujetos_ids:
            dis__ = np.mean(np.array(ind[sujeto][id]))
            if not np.isnan(dis__):
                dis_sujeto.append(dis__)
            else:
                dis_sujeto.append(-1)
        plt.bar(sujetos_ids.astype(str),dis_sujeto,0.5)
        plt.grid()
        plt.title(f'Mean distance {sujeto}')
        plt.ylabel('Distance[mtrs]')
        plt.xlabel('Sujects')
        plt.savefig(f'{path_save}/mean_distance_subject_{sujeto}.png')
        plt.close('all')

    
def processing_csv(path_save='./output/',*,path_data='./output/2021_03_20_08:45:32/2021_03_20_08:45:32.csv'):
    """ Make all the processing using a csv as input
    """
    tail_name = os.path.split(path_data)[1].split('.')[0]
    path_save =  os.path.join(path_save,tail_name)
    DataFrame = pd.read_csv(path_data)
    identifiers = DataFrame['id'].unique()
    mean_dist(DataFrame,path_save)
    for identifier in identifiers:
        x = np.array(DataFrame['Bx'][DataFrame['id']==identifier])
        y = np.array(DataFrame['By'][DataFrame['id']==identifier])
        heat_map(identifier,x,y,path_save)
        trajectory(identifier,x,y,path_save)

if __name__=="__main__":
    processing_csv(path_data='./output/2021_03_20_08:45:32/2021_03_20_08:45:32.csv')

