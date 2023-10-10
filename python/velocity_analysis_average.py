from velocity_analysis import analyzer_simulation
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import json
import argparse
from itertools import combinations
import matplotlib.pyplot as plt 
import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
#from sklearn.linear_model import LinearRegression

def ifnotmkdir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    return dir_

def plot_av_histeresys(x_av,y_av,c):
    x_av[:86] = x_av[:86]/len(c['list_config_files'])
    y_av[:86] = y_av[:86]/len(c['list_config_files'])
    u = [x_av[i+1]-x_av[i] for i in range(len(x_av)-1)]
    v = [y_av[i+1]-y_av[i] for i in range(len(y_av)-1)]
    u.append(x_av[len(x_av)-1] -x_av[0])
    v.append(y[len(y_av)-1] -y_av[0])

    plt.scatter(x_av[:86],y_av[:86])
    plt.quiver(x_av[:86],y_av[:86],u[:86],v[:86],angles='xy', scale_units='xy', scale=1,width = 0.0025)
    plt.xlabel('number people')
    plt.ylabel('velocity')
    plt.title('subnetwork complete intersection average {} days'.format(len(c['list_config_files'])))
    pth = ifnotmkdir(os.path.join(c['cartout_basename'],"complete_intersection_average"))
    plt.savefig(os.path.join(pth,'flux_in_DV_complete_intersection.png'),dpi = 200)
    plt.show()
    return True

def plot_av_MFD(x_tot,y_tot,c):
    _,binsx = np.histogram(np.array(x_tot),range = (min(x_tot),max(x_tot)),bins= 14 )
    print('min and max bin x: ',min(x_tot),' ',max(x_tot)) 
    _,binsy = np.histogram(y_tot,range = (min(y_tot),max(y_tot)))

    y_avg = np.zeros(len(binsx))
    y_dev = np.zeros(len(binsx))
    for dx in range(len(binsx)):
        if dx==len(binsx)-1:
            idx_ = np.array([True if xi>=binsx[dx] else False  for xi in np.array(x_tot)])
        else:
            idx_ = np.array([True if xi>=binsx[dx] and xi<=binsx[dx+1] else False  for xi in np.array(x_tot)])
        print('interval: ',dx,'average velocity: ',np.mean(y_tot[idx_]),' number of points considered: ',len(y_tot[idx_]))
        y_avg[dx] += np.mean(np.array(y_tot)[idx_])
        if len(y_tot[idx_])==1:
            y_dev[dx] = 0
        else:
            y_dev[dx] = np.std(np.array(y_tot)[idx_])
    #print('mean:\t',y_avg[:-1],'\nstd-dev:\t',y_dev[:-1],'\ndev/mean:\t',y_dev[:-1]/y_avg[:-1])
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    print('x average:\n',len(x_tot))
    print('y average:\n',len(y_tot))
    print('y dev:\n',len(y_dev))
    if c['linear_fit']:
        bins = [(np.array(binsx)[i+1]+np.array(binsx)[i])/2 for i in range(len(np.array(binsx))-1)]
        slope1,intercept1 = np.polyfit(binsx[:6],y_avg[:6],1)
        slope2,intercept2 = np.polyfit(binsx[5:len(binsx)-1],y_avg[5:len(binsx)-1],1)       
        ax.set_xticks(bins)
        plt.scatter(binsx,y_avg)
        plt.errorbar(binsx,y_avg, yerr=y_dev, fmt='o', linewidth=2, capsize=6)
        plt.plot(binsx[:6],slope1*binsx[:6]+intercept1,linestyle='--')
        plt.plot(binsx[5:len(binsx)-1],slope2*binsx[5:len(binsx)-1]+intercept2,linestyle='--')
        #plt.scatter(binsx,y_avg+y_dev)
        #plt.scatter(binsx,y_avg-y_dev)
        plt.xlabel('number people')
        plt.ylabel('velocity')
        plt.title("subnetwork complete intersection")
        plt.legend(['mean','error','{0}x + {1}'.format(round(slope1,3),round(intercept1,3)),'{0}x + {1}'.format(round(slope2,3),round(intercept2,3))])
        pth = ifnotmkdir(os.path.join(c['cartout_basename'],"complete_intersection"))
        plt.savefig(os.path.join(pth,'DV_average.png'),dpi = 200)
        plt.show()
        
    else:
        bins = [(np.array(binsx)[i+1]+np.array(binsx)[i])/2 for i in range(len(np.array(binsx))-1)]
        ax.set_xticks(bins)
        plt.scatter(binsx,y_avg)
        plt.errorbar(binsx,y_avg, yerr=y_dev, fmt='o', linewidth=2, capsize=6)
        #plt.scatter(binsx,y_avg+y_dev)
        #plt.scatter(binsx,y_avg-y_dev)
        plt.xlabel('number people')
        plt.ylabel('velocity')
        plt.title("subnetwork complete intersection")
        plt.legend(['mean','error'])
        pth = ifnotmkdir(os.path.join(c['cartout_basename'],"complete_intersection"))
        plt.savefig(os.path.join(pth,'DV_average.png'),dpi = 200)
        plt.show()

def extract_variables_MFD(c):
    x_tot = []
    y_tot= []
    for config_file in c['list_config_files']:
        with open(config_file,'r') as f:
            jsonfile = f.read()
        config_ = json.loads(jsonfile)
        anal = analyzer_simulation(config_)
        x,y = anal.fondamental_diagram_complete_intersection()
        x_tot.extend(x)
        y_tot.extend(y)
    return np.array(x_tot),np.array(y_tot)

def extract_av_in_day_DV(c,max_ = 0):
    x_av = np.zeros(96)
    y_av = np.zeros(96)
    for config_file in c['list_config_files']:
        with open(config_file,'r') as f:
            jsonfile = f.read()
        config_ = json.loads(jsonfile)
        anal = analyzer_simulation(config_)
        x,y = anal.fondamental_diagram_complete_intersection()
        if len(x)>max_:
            max_ = len(x)
        if len(x)<max_:
            x = np.append(x,np.zeros(max_-len(x)))
            y = np.append(y,np.zeros(max_-len(y)))
        x_av += x
        y_av += y
    return np.array(x_av),np.array(y_av)

    

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='directory configuaration file', required=True)
args = parser.parse_args()
with open(args.config,'r') as f:
        jsonfile = f.read()
c = json.loads(jsonfile)
linear_fit = c['linear_fit']
#CHEATING PUTTING APART 10 POINTS (ASSUMING THEY COME AT LAST 2.30 hours)
if c['plot_hysteresis']:
    x_av,y_av = extract_av_in_day_DV(c)
    plot_av_histeresys(x_av,y_av,c)
if c['plot_av_MFD']:
    x_tot,y_tot = extract_variables_MFD(c)
    if linear_fit:
        plot_av_MFD(x_tot,y_tot,c)

