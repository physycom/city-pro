# CONTROLS THAT TRAJECTORIES ARE FOUND INSIDE SOME POLIGON, trying to assess some density of data points, 
# USELESS in terms of the analysis. USEFUL somwhat in the middle of validation of the analysis.
import pandas as pd
import numpy as np            
import os
import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import movingpandas as mpd
import hvplot.pandas 
from shapely.geometry import Polygon
from shapely.ops import unary_union

def generate_geohash_polygons(min_lat, max_lat, min_lon, max_lon, precision):
    geohash_polygons = []
    
    lat_interval = max_lat - min_lat
    lon_interval = max_lon - min_lon
    
    lat_step = lat_interval / precision
    lon_step = lon_interval / precision
    
    for lat_idx in range(precision):
        for lon_idx in range(precision):
            lat_start = min_lat + (lat_idx * lat_step)
            lon_start = min_lon + (lon_idx * lon_step)
            
            lat_end = lat_start + lat_step
            lon_end = lon_start + lon_step
            
            polygon = Polygon([(lon_start, lat_start),
                               (lon_start, lat_end),
                               (lon_end, lat_end),
                               (lon_end, lat_start)])
            
            geohash_polygons.append(polygon)
    
    return unary_union(geohash_polygons) 
'''
dd = pd.read_csv("/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/2022-11-11/bologna_mdt_2022-11-11_2022-11-11.csv",";")
#gdf = gpd.read_file('/home/aamad/codice/city-pro/bologna-provincia.geojson')
l = list(zip(dd['LAT'],dd['LON'])) 
points = [Point(ll) for ll in l]
dd['geometry'] = points
dd['time_idx']=[datetime.datetime.fromtimestamp(t) for t in dd['timestamp']]
dd = dd.set_index('time_idx')
gdd = gpd.GeoDataFrame(dd[:10000],geometry='geometry')
traj_collection = mpd.TrajectoryCollection(gdd, 'CALL_ID', t='time_idx')
#traj_collection = traj_collection.to_crs()
#print('traj collection:\n',traj_collection.df.head())
detector = mpd.TrajectoryStopDetector(traj_collection)
stop_collection = detector.get_stop_points(min_duration = datetime.timedelta(seconds=300),max_diameter = 50)
#print('stop collection:\n',stop_collection.df.head())
stop_collection.hvplot(geo = True,tiles=True,hover_cols='all') 
plt.savefig('/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/2022-11-11/anomaly_points_all.png',dpi = 200)
traj_collection.hvplot(geo = True,tiles=True,hover_cols='all')
plt.savefig('/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/2022-11-11/anomaly_traj_all.png',dpi = 200)

idx = ''
max_l = 0
for g,dfg in dd.groupby('CALL_ID'):
    if len(dfg)>max_l:
        max_l = len(dfg)
        idx = g
    else:
        pass
df_max = dd.groupby('CALL_ID').get_group(g)
l = list(zip(df_max['LAT'],df_max['LON'])) 
points = [Point(ll) for ll in l]
df_max['geometry'] = points
df_max = df_max.sort_values(by='timestamp')
df_max['time_idx']=[datetime.datetime.fromtimestamp(t) for t in df_max['timestamp']]
df_max = df_max.set_index('time_idx')
print(df_max.head())
#gdf_max = gpd.GeoDataFrame(df_max.set_index('time_idx'), crs=31256)
#plt.subplots(1,1,figsize=(10,8))
traj_collection = mpd.TrajectoryCollection(df_max, 'CALL_ID', t='time_idx', x='LAT', y='LON')
traj_collection.plot(column='CALL_ID', legend=True, figsize=(9,5))
traj_collection.hvplot(line_width=7.0, tiles='OSM')
plt.savefig('/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/2022-11-11/anomaly_traj_py.png',dpi = 200)

df_max = pd.read_csv('/home/aamad/codice/city-pro/output/bologna_mdt_detailed/bologna_mdt_2022-08-05_2022-08-05_traj2plot.csv',';')
l = list(zip(df_max['LAT'],df_max['LON'])) 
points = [Point(ll) for ll in l]
df_max['geometry'] = points
df_max = df_max.sort_values(by='timestamp')
df_max['time_idx']=[datetime.datetime.fromtimestamp(t) for t in df_max['timestamp']]
df_max = df_max.set_index('time_idx')
print(df_max.head())
#gdf_max = gpd.GeoDataFrame(df_max.set_index('time_idx'), crs=31256)
#plt.subplots(1,1,figsize=(10,8))
traj = mpd.Trajectory(df_max, 'CALL_ID', t='time_idx', x='LAT', y='LON')
traj.df.set_crs("EPSG:4326",inplace=True,allow_override=True)
dist = traj.distance(metric='geodesic')
print(dist)
traj.add_distance(overwrite=True,name='distance')
#traj.add_angular_difference(name='distance')
print(traj.df)
traj.plot(column='CALL_ID', legend=True, figsize=(9,5))
traj.hvplot(line_width=7.0, tiles='OSM')
plt.savefig('/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/2022-11-11/anomaly_traj.png',dpi = 200)
#plt.show()
'''
df = pd.read_csv('/home/aamad/codice/city-pro/output/bologna_mdt_detailed/bologna_mdt_2022-08-05_2022-08-05_traj2plot1.csv',';')
idx = ''
max_l = 0
for g,dfg in df.groupby('CALL_ID'):
    if len(dfg)>max_l:
        max_l = len(dfg)
        idx = g
        print(g,type(g))
    else:
        pass
df_max = df[:100000]#.groupby('CALL_ID').get_group(g)

l = list(zip(df_max['LAT'],df_max['LON'])) 
points = [Point(ll) for ll in l]
df_max['geometry'] = points
df_max.sort_values(by='timestamp')
df_max['time_idx']=[datetime.datetime.fromtimestamp(t) for t in df_max['timestamp']]
df_max = df_max.set_index('time_idx')
gdf_max = gpd.GeoDataFrame(df_max,geometry='geometry')
#gdf_max = gpd.GeoDataFrame(df_max.set_index('time_idx'), crs=31256)
#plt.subplots(1,1,figsize=(10,8))
trjectories = []
for g,dfg in gdf_max.groupby('CALL_ID'): 
    traj = mpd.Trajectory(dfg,idx)
    traj.df.set_crs("EPSG:4326",inplace=True,allow_override=True)
    traj.add_speed(overwrite=True)
#    trajectories.append(traj)
    traj.plot(column='speed',with_basemap=True,linewidth=3,legend=True,figsize=(9,9))
    plt.savefig('/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/2022-11-11/traj_velocity{}.png'.format(g),dpi = 150)

traj_collection = mpd.TrajectoryCollection(df_max, 'CALL_ID', t='time_idx', x='LAT', y='LON')
traj_collection.plot(column='CALL_ID', legend=True, figsize=(9,5))
traj_collection.hvplot(c = 'speed',line_width=7.0, tiles='OSM')
plt.savefig('/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/2022-11-11/anomaly_traj1.png',dpi = 200)
df_traj = df.loc[df['CALL_ID']==idx]


# Define the bounding box coordinates
min_latitude = 40.0
max_latitude = 42.0
min_longitude = 10.0
max_longitude = 12.0

# Specify the desired geohash precision (number of divisions per side)
precision_level = 4

# Generate the geohash polygons
geohash_polygons = generate_geohash_polygons(min_latitude, max_latitude, min_longitude, max_longitude, precision_level)

# Create a GeoDataFrame from the geohash polygons
geohash_gdf = gpd.GeoDataFrame(geometry=[geohash_polygons])