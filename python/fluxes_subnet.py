import folium
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import json
import argparse
from itertools import combinations
import matplotlib.pyplot as plt 
import datetime
import warnings
from collections import defaultdict
from multiprocessing import Pool
warnings.filterwarnings('ignore')


def _class_int2str(fcm_file):
    '''
    Creates dict_vel: {0:vel0,1:vel1,...} vel0<vel1<...
            dict_class_int2str: {0:i slowest,...,i quickest}
    '''
    df_fcm = pd.read_csv(fcm_file,';')
    dict_vel = defaultdict()
    for f,fc in df_fcm.groupby('class'):
        if f!=10:
            vel = np.mean(fc['av_speed'].to_numpy())
            if vel<50:
                dict_vel[f] = vel
                print('velocity of class {0} is: {1} m/s'.format(f,vel))
            else:
                print('velocity of class {0} is: {1} m/s'.format(f,vel))
    dict_vel = dict(sorted(dict_vel.items(),key = lambda item:item[1]))
    class_int2str = defaultdict(dict)
    number_classes = len(dict_vel.keys())
    for i in range(number_classes):
        if i<number_classes/2:
            class_int2str[list(dict_vel.keys())[i]] = '{} slowest'.format(i+1)
        elif i<number_classes==2:
            class_int2str[list(dict_vel.keys())[i]] = 'middle velocity class'
        else:
            class_int2str[list(dict_vel.keys())[i]] = '{} quickest'.format(number_classes - i)      
               
    return class_int2str,dict_vel


def complete_intersection2gdf(intersection_poly):
    mask = [True if str(x) in intersection_poly else False for x in geoj['poly_lid']]    
    gdf = geoj.loc[mask] 
    masknot = [False if str(x) in intersection_poly else True for x in geoj['poly_lid']]    
    gdfnot =  geoj.loc[masknot] 
    return gdf,gdfnot
    
def geoj2gdf(dir_file):
    '''Input: file pointing to list of roads to choose in the geojson file'''
    file_complement1 = open(dir_file,'r')        
    list_complement1 = file_complement1.read().split(" ")   
    mask = [True if str(x) in list_complement1 else False for x in geoj['poly_lid']]    
    gdf = geoj.loc[mask] 
    masknot = [False if str(x) in list_complement1 else True for x in geoj['poly_lid']]    
    gdfnot =  geoj.loc[masknot] 
    return gdf,gdfnot

def geoj2_gdf_each_class(dir_file,class_int2str):
    '''
    Output:
        class2gdf: {0:[gdf,gdfnot],1:[gdf,gdfnot],...}
    '''
    file_complement1 = open(dir_file,'r')        
    list_complement1 = file_complement1.read().split("\t")
    class2subnet = defaultdict(list)
    class2gdf = defaultdict()
    print('size list all subnets: ',len(list_complement1))
    for i in class_int2str.keys():
        for x in list_complement1:
            if "class_{}".format(i) in x:
                print("class_{}".format(i))
                class2subnet[i] = []
            else:
                class2subnet[i].append(x)
    print("class2subnet keys: ",class2subnet.keys())
    for i in class2subnet.keys():
        print('number polies subnet: ',class_int2str[i],' ',len(class2subnet[i]))
        mask = [True if str(x) in class2subnet[i] else False for x in geoj['poly_lid']]    
        gdf = geoj.loc[mask] 
        masknot = [False if str(x) in class2subnet[i] else True for x in geoj['poly_lid']]    
        gdfnot =  geoj.loc[masknot] 
        class2gdf[i] = [gdf,gdfnot]
    return class2gdf

def plot(gdf,gdfnot,string_,working_dir,classes_colors,i):
    network = gdf.to_crs({'init': 'epsg:4326'})
    networknot = gdfnot.to_crs({'init': 'epsg:4326'})    
    fig,ax = plt.subplots(1,1,figsize = (20,15))
    plt.title(string_)
    try:
        networknot.plot(ax = ax,color = "green")    
        network.plot(ax = ax,color = classes_colors[i])
        plt.savefig(os.path.join(working_dir,'plot',string_ +'.png'),dpi = 200)
        plt.show()
    except ValueError:
        print('direct ',len(gdf))
        print('negated ',len(gdfnot))
        print('empty intersection')
    return fig,ax    

def plot_complete_intersection(gdf,gdfnot,string_,working_dir):
    network = gdf.to_crs({'init': 'epsg:4326'})
    networknot = gdfnot.to_crs({'init': 'epsg:4326'})    
    fig,ax = plt.subplots(1,1,figsize = (20,15))
    plt.title(string_)
    try:
        networknot.plot(ax = ax,color = "green")    
        network.plot(ax = ax,color = "blue")
        plt.savefig(os.path.join(working_dir,'plot',string_ +'.png'),dpi = 200)
        plt.show()
    except ValueError:
        print('direct ',len(gdf))
        print('negated ',len(gdfnot))
        print('empty intersection')
    return fig,ax    


def _get_complete_intersection(class2gdf):
    intersection = set(class2gdf[0][0])
    for i in range(len(class2gdf.keys())-1):
        intersection = intersection.intersection(set(class2gdf[i+1][0]))
    return list(intersection)
def initialize_from_config_file(config_):
    working_dir = config_['working_dir']
    file_geojson = config_['file_geojson']
    prefix_name = config_["prefix_name"]
    cartout_basename = config_['cartout_basename']
    subnet_class = config_['subnet_txt_files']
    subnet_complement_class = config_['subnet_txt_files_complement']
    subnet_complete_intersection = config_['subnet_txt_files_complete_intersection']
    geoj = gpd.read_file(file_geojson,driver = 'GeoJSON')
    day_in_sec = config_['day_in_sec']
    dt = config_['dt']
    iterations = day_in_sec/dt
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(config_['start_date'],time_format)
    end_date = datetime.datetime.strptime(config_['end_date'],time_format)
    day_analysis = config_["day_analysis"]
    return working_dir,file_geojson,prefix_name,cartout_basename,subnet_class,subnet_complement_class,subnet_complete_intersection,geoj,day_in_sec,dt,iterations,time_format,start_date,end_date,day_analysis

if __name__ =="__main__":
    combs = combinations("01234",2)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='directory configuaration file', required=True)
    args = parser.parse_args()
    with open(args.config,'r') as f:
         jsonfile = f.read()
    config_ = json.loads(jsonfile)
    working_dir,file_geojson,prefix_name,cartout_basename,subnet_class,subnet_complement_class,subnet_complete_intersection,geoj,day_in_sec,dt,iterations,time_format,start_date,end_date,day_analysis = initialize_from_config_file(config_)
    print('day under analysis: ',day_analysis)
# GET NUMBER OF CLASSES: init: class_int2str: {class_int: class_str,...} WITHOUT CLASS 10 and the quickest if the average velocity is bigger than 50 m/s
    fcm_file = os.path.join(working_dir,prefix_name +"_fcm.csv")
    classes_idx = config_['class_idx']
    classes_names = config_['class_names']
    classes_colors = config_['colors'] 
    class_int2str,dict_vel = _class_int2str(fcm_file)

# PLOT SINGLE SUBNETS 
    #print("dictionary velocities:\n",dict_vel)
    #print("dictionary class2str:\n",class_int2str)
    count = 0   
    for i in range(5):
        if i in list(dict_vel.keys()):
            print('subnet: ',i,' color: ',classes_colors[count],' class from fcm file: ',class_int2str[i])
            gdf_subnet,gdf_subnet_complement = geoj2gdf(subnet_class[count])
            plot(gdf_subnet,gdf_subnet_complement,class_int2str[i],working_dir,classes_colors,count)
            count += 1
#            gdf_subnet_complementary,gdf_subnet_complementary_complement = geoj2gdf(subnet_complement_class[i])
#            plot(gdf_subnet_complementary,gdf_subnet_complementary_complement,'complete_complement_'+class_int2str[i],working_dir,classes_colors,count)
# PLOT COMPLETE INTERSECTION
    print('plot complete intersection')
    gdf_subnet_complete_intersection,gdf_subnet_complete_intersection_complement = geoj2gdf(subnet_complete_intersection)
    plot_complete_intersection(gdf_subnet_complete_intersection,gdf_subnet_complete_intersection_complement,"complete_intersection",working_dir)
 #   class2gdf = geoj2_gdf_each_class(config_['weights_dir'],class_int2str)
#    print('plot subnet for each class separately: ')
#    for i in class2gdf.keys():
#        print('subnet: ',i)
#        plot(class2gdf[i][0],class_int2str[i],working_dir,class2gdf[i][1])

# PLOT: COMPLEMENT completo e INTERSECTION completa
'''    print('list files: ',list_files)
    for f_ in list_files:
        print('geojson file: ',f_)
        string_components_dir = f_.split("/")
        # OBTAIN TITLE PLOT
        for s in string_components_dir:
            if prefix_name in s:
                string_components_name = s.split('_')
                for i in ['0','1','2','3','4']:
                    for component in string_components_name:
                        if component == i:
                            string_class_title_plot = class_int2str[int(i)] 
                        else:
                            pass
            else:
                pass 
        gdf_complete_intersection,gdf_complete_intersection_complement=geoj2gdf(f_)
        print('plotting complete intersection')
        plot(gdf_complete_intersection,prefix_name+string_class_title_plot,working_dir,gdf_complete_intersection_complement)
         
    gdf_complete_intersection,gdf_complete_intersection_complement=geoj2gdf(os.path.join(cartout_basename,prefix_name+"_complete_intersection.csv"))
    vel_sub_complete_intersection=pd.read_csv(os.path.join(cartout_basename,prefix_name +"_complete_intersection_velocity_subnet.csv"))
    plot(gdf_complete_intersection,"complete_intersection",working_dir,gdf_complete_intersection_complement)
        
    for s in ["0","1","2","3"]:
        gdf_complete_complement,gdf_complete_complement_complement=geoj2gdf(os.path.join(cartout_basename,prefix_name+s+"_complete_complement.csv"))
        plot(gdf_complete_complement,class_int2str[int(s)]+"_complete_complement",working_dir,gdf_complete_complement_complement)    

        vel_sub_complete_complement=pd.read_csv(os.path.join(cartout_basename,prefix_name +"_complete_complement_"+s+"_velocity_subnet.csv"))
    # COMPLETE INTERSECTION SUBNET and FILE FLUXES
    intersection = _get_complete_intersection(class2gdf)
    gdf_complete_intersection,gdf_complete_intersection_complement = complete_intersection2gdf(intersection)
    plot(gdf_complete_intersection,"complete_intersection_no_biggest_velocity",working_dir,gdf_complete_intersection_complement)
    
# coppie di sottonetwork    
#    for couple in combs:
#        fcm_comp1 = pd.read_csv(os.path.join(cartout_basename,prefix_name+couple[0]+couple[1]+"_complementary1_fcm.csv"))
#        fcm_comp2 = pd.read_csv(os.path.join(cartout_basename,prefix_name+couple[0]+couple[1]+"_complementary2_fcm.csv"))
'''