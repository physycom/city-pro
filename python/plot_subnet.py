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
def indx_vel(fcm,dict_vel,i):
    '''
    Output: dict: {velocity:'velocity class in words: (slowest,...quickest)]}
    '''
    for f,fc in fcm[i].groupby('class'):
        if f!=10:
            vel = np.mean(fc['av_speed'].to_numpy())
            dict_vel[f] = vel
    dict_vel = dict(sorted(dict_vel.items(),key = lambda item:item[1]))
    dict_name = defaultdict(dict)
    number_classes = len(fcm[i].groupby('class')) - 1
    for i in range(number_classes):
        if i<number_classes/2:
            dict_name[list(dict_vel.keys())[i]] = '{} slowest'.format(i+1)
        elif i<number_classes==2:
            dict_name[list(dict_vel.keys())[i]] = 'middle velocity class'
        else:
            dict_name[list(dict_vel.keys())[i]] = '{} quickest'.format(number_classes - i)             
    return dict_vel,dict_name


def geoj2gdf(dir_file):
    '''
    Read subnet files (associated for each subnet) and extract polies in the geodataframe.
    
    '''
    file_complement1 = open(dir_file,'r')        
    list_complement1 = file_complement1.read().split(" ")   
    mask = [True if str(x) in list_complement1 else False for x in geoj['poly_lid']]    
    gdf = geoj.loc[mask] 
    masknot = [False if str(x) in list_complement1 else True for x in geoj['poly_lid']]    
    gdfnot =  geoj.loc[masknot] 
    return gdf,gdfnot

def plot(gdf,string_,working_dir,gdfnot):
    network = gdf.to_crs({'init': 'epsg:4326'})
#    networknot = gdfnot.to_crs({'init': 'epsg:4326'})    
    fig,ax = plt.subplots(1,1,figsize = (20,15))
    plt.title(string_)
    try:
#        networknot.plot(ax = ax,color = "green")    
        network.plot(ax = ax,color = "blue")
        plt.savefig(os.path.join(working_dir,'plot',string_ +'.png'),dpi = 200)
        plt.show()
    except ValueError:
        print('direct ',len(gdf))
        print('negated ',len(gdfnot))
        print('empty intersection')
    return fig,ax    
if __name__ == '__main__':
    combs = combinations("01234",2)    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='directory configuaration file', required=True)
    args = parser.parse_args()
    with open(args.config,'r') as f:
         jsonfile = f.read()
    config_ = json.loads(jsonfile)
    working_dir = config_['working_dir']
    file_geojson = config_['file_geojson']
    prefix_name = config_["prefix_name"]
    cartout_basename = config_['cartout_basename']
    
    geoj = gpd.read_file(file_geojson,driver = 'GeoJSON')
    day_in_sec = config_['day_in_sec']
    dt = config_['dt']
    iterations = day_in_sec/dt
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(config_['start_date'],time_format)
    end_date = datetime.datetime.strptime(config_['end_date'],time_format)
    day_analysis = config_["day_analysis"]
    

    for couple in combs:
        tok0 = couple[0]
        tok1 = couple[1]
        # PREPARE 
        dir_complement2 = os.path.join(cartout_basename,"bologna_mdt_30-12_without_presence_file" +tok0 + tok1+ "_subnet_complementary2.csv") 
        dir_complement1 = os.path.join(cartout_basename,"bologna_mdt_30-12_without_presence_file" +tok0 + tok1+ "_subnet_complementary1.csv")
        dir_intersection = os.path.join(cartout_basename,"bologna_mdt_30-12_without_presence_file" +tok0 + tok1+ "_subnet_intersection.csv")
        gdf_complement1,gdf_complement1not = geoj2gdf(dir_complement1)
        gdf_complement2,gdf_complement2not = geoj2gdf(dir_complement2)
        gdf_intersection,gdf_intersectionnot = geoj2gdf(dir_intersection)
        # PLOT
        plot(gdf_intersection,"intersection "+tok0+"_"+tok1,working_dir,gdf_intersectionnot)
        plot(gdf_complement1,"complement1 "+tok0+"_"+tok1,working_dir,gdf_complement1not)
        plot(gdf_complement2,"complement2 "+tok0+"_"+tok1,working_dir,gdf_complement2not)
        m_compl1 = folium.Map(location=[36.862317, -76.3151], zoom_start=6)
        m_compl2 = folium.Map(location=[36.862317, -76.3151], zoom_start=6)
        m_intersection = folium.Map(location=[36.862317, -76.3151], zoom_start=6)
        