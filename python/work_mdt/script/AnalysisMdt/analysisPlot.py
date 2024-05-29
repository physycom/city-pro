import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

##----------------------------------- PLOT VELOCITIES -----------------------------------##

def QuiverPopulationVelocityClass(population,velocity,save_dir,day,idx,dict_name,average_all_days = False):
    '''
        Input:
            population: (np.array 1D) -> population 
            velocity: (np.array 1D) -> velocity 
            dict_idx: (dict) -> dict_idx = {'population':[],'velocity':[]}
            save_dir: (str) -> save_dir = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/BOS'
            day: (str) -> day = 'day_1'
            idx: (int) -> idx = 0
            dict_name: (dict) -> dict_name = {0:'1 slowest',1:'2 slowest'
    '''
    assert population is not None, 'population must be provided'
    assert velocity is not None, 'velocity must be provided'
    assert len(population) == len(velocity), 'population and velocity must have the same length'
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    u = [population[i+1]-population[i] for i in range(len(population)-1)]
    v = [velocity[i+1]-velocity[i] for i in range(len(velocity)-1)]
    u.append(population[len(population)-1] -population[0])
    v.append(velocity[len(velocity)-1] -velocity[0])
    ax.quiver(population,velocity,u,v,angles='xy', scale_units='xy', scale=1,width = 0.0025)
    ax.set_xlabel('number people')
    ax.set_ylabel('velocity')
    ax.set_title(str(dict_name[idx]))
    if average_all_days:
        plt.savefig(os.path.join(save_dir,'Hysteresis_Average_{0}_Class_{1}.png'.format(day,dict_name[idx])),dpi = 200)
    else:
        plt.savefig(os.path.join(save_dir,'Hysteresis_{0}_Class_{1}.png'.format(day,dict_name[idx])),dpi = 200)
    plt.show()

def MFDByClass(population,velocity,dict_name,idx,save_dir,verbose = False): 

    nx,binsPop = np.histogram(population,range = (min(population),max(population)))
    y_avg = np.zeros(len(binsPop))
    y_dev = np.zeros(len(binsPop))
    for dx in range(len(binsPop)-1):
        idx_ = np.array([True if xi>=binsPop[dx] and xi<=binsPop[dx+1] else False  for xi in x])
        y_avg[dx] += np.mean(velocity[idx_])
        y_dev[dx] = np.std(velocity[idx_])
    print('mean:\t',y_avg[:-1],'\nstd-dev:\t',y_dev[:-1],'\ndev/mean:\t',y_dev[:-1]/y_avg[:-1])
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.plot(binsPop[:-1],y_avg[:-1])
    ax.plot(binsPop[:-1],y_avg[:-1]+y_dev[:-1])
    ax.plot(binsPop[:-1],y_avg[:-1]-y_dev[:-1])
    ax.set_xlabel('number people')
    ax.set_ylabel('velocity')
    ax.set_title(str(dict_name[idx]))
    ax.legend(['mean','mean+std','mean-std'])
    plt.savefig(os.path.join(save_dir,'{0}_class_averageV_per_D_{1}.png'.format(dict_name[idx],day)),dpi = 200)
    plt.show()



##----------------------------------- PLOT TIMES -----------------------------------##




##----------------------------------- PLOT DISTANCES -----------------------------------##




##----------------------------------- PLOT FLUXES -----------------------------------##