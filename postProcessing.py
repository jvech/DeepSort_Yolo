""" Module for pots processing the data from csv files.
Returns three kinf of images:
    1. Hot map for each id
    2. Trajector for each id.
    3. Animation of position of ids in the place. I need work more about it :v.
All these images are going to be save in a folder.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def heat_map(identifier,x,y,path_save):
    """Returns the heat map of objects along the  frame.
    """
    #Filter Bx,By in all frame based on Identifier 
    H, xedges, yedges = np.histogram2d(x, y, bins=(10, 10))
    X, Y = np.meshgrid(xedges, yedges)
    plt.figure()
    plt.pcolormesh(X, Y, H)
    plt.colorbar()
    plt.xlabel("x meters")
    plt.ylabel("y meters")
    plt.title(f'Heat map subject{identifier}')
    plt.savefig(f'{path_save}/heat_map_sujeto_{identifier}.png')
    plt.close('all')


def trajectory(identifier,x,y,path_save):
    """ Returns the trajectory of Identifier"""
    plt.figure()
    plt.plot(x, y, '.-')
    plt.plot(x[0], y[0], 'ro')
    plt.plot(x[-1], y[-1], 'go')
    plt.xlabel("x meters")
    plt.ylabel("y meters")
    plt.legend(['Trayectory','Initial Point','Final Point'])
    plt.title(f'Trajectory subject {identifier}')
    plt.savefig(f'{path_save}/trajectory_sujeto_{identifier}.png')
    plt.close('all')

def animate_positions(DataFrame,path_save):
    """Returns animation of moves the all id
    for repair"""

    fig,ax = plt.subplots()
    def animate(i):
        ax.clear()
        x = np.array(DataFrame['Bx'][DataFrame['frame'] == 3+i])
        y = np.array(DataFrame['By'][DataFrame['frame'] == 3+i])
        identifiers = DataFrame['id'][DataFrame['frame']==i+3]
        ax.scatter(x,y)
        ax.grid()
        for i,identifier in enumerate(identifiers):
            ax.annotate(str(identifier),(x[i],y[i]))
        ax.set_title(str(i+3))
    ani = animation.FuncAnimation(fig,animate,range(len(DataFrame['frame'].unique())),interval=1)
    f = f'{path_save}/animate_positions.mp4'
    writermp4 = animation.FFMpegWriter(fps=5)
    ani.save(f, writer=writermp4)

    
def processing_csv(path_save='./output/',*,path_data='./output/2021_03_20_08:45:32/2021_03_20_08:45:32.csv'):
    tail_name = os.path.split(path_data)[1].split('.')[0]
    path_save =  os.path.join(path_save,tail_name)
    DataFrame = pd.read_csv(path_data)
    identifiers = DataFrame['id'].unique()

    for identifier in identifiers:
        x = np.array(DataFrame['Bx'][DataFrame['id']==identifier])
        y = np.array(DataFrame['By'][DataFrame['id']==identifier])
        heat_map(identifier,x,y,path_save)
        trajectory(identifier,x,y,path_save)

if __name__=="__main__":
    processing_csv(path_data='./output/2021_03_20_08:45:32/2021_03_20_08:45:32.csv')

